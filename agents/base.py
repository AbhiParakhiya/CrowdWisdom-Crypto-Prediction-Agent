"""
agents/base.py
──────────────
Base Hermes-style agent class.

Each concrete agent:
  1. Declares a set of "tools" (plain async methods decorated with @tool)
  2. Calls self.run(prompt) to let the LLM pick and invoke tools in a loop
  3. Returns a structured result

The agent loop mirrors the Hermes ReAct pattern:
  LLM → think → pick tool → observe result → repeat until done
"""

from __future__ import annotations

import json
import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import cfg
from core.logger import get_logger

# Registry of all tools defined on an agent instance
_TOOL_ATTR = "__hermes_tool__"


def tool(description: str):
    """Decorator that marks an async method as an LLM-callable tool."""
    def decorator(fn: Callable) -> Callable:
        fn.__hermes_tool__ = True
        fn.__tool_description__ = description
        return fn
    return decorator


class BaseAgent(ABC):
    """
    Hermes-style ReAct agent base class.

    Subclasses implement:
      - tools decorated with @tool(description="...")
      - run_task() which calls self.llm_loop(system_prompt, user_prompt)
        and returns a typed result
    """

    agent_name: str = "BaseAgent"

    def __init__(self):
        self.log = get_logger(self.agent_name)
        self._client = AsyncOpenAI(
            api_key=cfg.openrouter_api_key,
            base_url=cfg.openrouter_base_url,
        )
        self._tools_schema = self._build_tools_schema()
        self._tool_map: dict[str, Callable] = {
            name: method
            for name, method in inspect.getmembers(self, predicate=inspect.ismethod)
            if getattr(method, _TOOL_ATTR, False)
        }

    def _build_tools_schema(self) -> list[dict]:
        """Build OpenAI-compatible function schema from @tool-decorated methods."""
        schemas = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not getattr(method, _TOOL_ATTR, False):
                continue
            sig = inspect.signature(method)
            properties = {}
            required = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                ptype = "string"
                if param.annotation in (int,):
                    ptype = "integer"
                elif param.annotation in (float,):
                    ptype = "number"
                elif param.annotation in (bool,):
                    ptype = "boolean"
                properties[param_name] = {"type": ptype}
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": method.__tool_description__,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return schemas

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def _chat(self, messages: list[dict], use_tools: bool = True) -> dict:
        """Single LLM call with optional tool use."""
        kwargs: dict[str, Any] = {
            "model": cfg.openrouter_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1024,
        }
        if use_tools and self._tools_schema:
            kwargs["tools"] = self._tools_schema
            kwargs["tool_choice"] = "auto"
        response = await self._client.chat.completions.create(**kwargs)
        return response

    async def llm_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        max_iterations: int = 6,
    ) -> str:
        """
        Hermes ReAct loop:
        1. Send system + user message
        2. If LLM calls a tool → invoke it, append observation, repeat
        3. When LLM returns text without tool call → return that text
        """
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for iteration in range(max_iterations):
            self.log.debug("LLM loop iteration {}/{}", iteration + 1, max_iterations)
            response = await self._chat(messages)
            msg = response.choices[0].message

            # No tool calls — agent is done
            if not msg.tool_calls:
                return msg.content or ""

            # Process all tool calls in this turn
            messages.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments or "{}")
                self.log.info("Tool call: {}({})", fn_name, fn_args)

                if fn_name not in self._tool_map:
                    result = f"Error: unknown tool '{fn_name}'"
                else:
                    try:
                        result = await self._tool_map[fn_name](**fn_args)
                        if not isinstance(result, str):
                            result = json.dumps(result, default=str)
                    except Exception as exc:
                        self.log.error("Tool {} raised: {}", fn_name, exc)
                        result = f"Error executing {fn_name}: {exc}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        self.log.warning("Max iterations reached — returning last message")
        return messages[-1].get("content", "")

    @abstractmethod
    async def run_task(self, **kwargs) -> Any:
        """Each agent implements this as its main entry point."""
        ...
