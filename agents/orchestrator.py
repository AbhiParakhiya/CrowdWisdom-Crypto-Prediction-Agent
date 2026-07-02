"""
agents/orchestrator.py
───────────────────────
Orchestrator Agent — Hermes Feedback Loop

The orchestrator is the "brain" of the system. It:
  1. Kicks off all child agents in the right sequence
  2. Aggregates their signals with confidence-weighted voting
  3. Runs a Hermes-style feedback loop: if consensus confidence is below
     cfg.confidence_threshold, it re-runs lower-confidence agents with
     adjusted prompts until threshold is met or max_feedback_rounds exceeded
  4. Uses the LLM to synthesize a final human-readable narrative
  5. Returns a FinalSignal for each asset

Multi-timeframe scaling strategy:
  ┌──────────────────────────────────────────────────────────────┐
  │  1m bars → predict 1m n+5 → use as leading signal for 5m n+1│
  │  5m bars → predict 5m n+1 → primary signal                  │
  │  15m bars → predict 15m n+1 → arbitrage check vs 3×5m       │
  └──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Optional

from agents.base import BaseAgent, tool
from agents.kelly_risk import KellyRiskAgent
from agents.kronos_predictor import KronosPredictorAgent
from agents.market_search import MarketSearchAgent
from agents.ohlcv_fetcher import OHLCVFetcherAgent
from core.config import cfg
from core.models import (
    AgentSignal,
    Asset,
    Direction,
    FinalSignal,
    KellyResult,
    KronosPrediction,
    MarketSearchResult,
    OHLCVData,
    Timeframe,
)


class OrchestratorAgent(BaseAgent):
    """
    Master orchestrator with Hermes feedback loop.
    Runs all sub-agents, aggregates signals, re-runs if confidence is low.
    """

    agent_name = "OrchestratorAgent"

    def __init__(self):
        super().__init__()
        self._market_search = MarketSearchAgent()
        self._ohlcv_fetcher = OHLCVFetcherAgent()
        self._kronos = KronosPredictorAgent()
        self._kelly = KellyRiskAgent()

    # ── Tools (used by LLM in synthesis loop) ─────────────────────────────────

    @tool(
        "Aggregate multiple directional signals with confidence weights. "
        "Pass signals as JSON array of {direction, confidence, weight}. "
        "Returns consensus direction and confidence."
    )
    async def aggregate_signals(self, signals_json: str) -> str:
        """Weighted vote on direction signals."""
        try:
            signals = json.loads(signals_json)
            up_score = 0.0
            down_score = 0.0
            total_weight = 0.0

            for s in signals:
                direction = s.get("direction", "NEUTRAL")
                confidence = float(s.get("confidence", 0.5))
                weight = float(s.get("weight", 1.0))
                weighted_conf = confidence * weight
                total_weight += weight

                if direction == "UP":
                    up_score += weighted_conf
                elif direction == "DOWN":
                    down_score += weighted_conf

            if total_weight == 0:
                return json.dumps({"direction": "NEUTRAL", "confidence": 0.5})

            up_norm = up_score / total_weight
            down_norm = down_score / total_weight

            if up_norm > down_norm + 0.02:
                direction = "UP"
                confidence = up_norm
            elif down_norm > up_norm + 0.02:
                direction = "DOWN"
                confidence = down_norm
            else:
                direction = "NEUTRAL"
                confidence = 0.50

            return json.dumps({
                "direction": direction,
                "confidence": round(confidence, 4),
                "up_score": round(up_norm, 4),
                "down_score": round(down_norm, 4),
            })
        except Exception as exc:
            return json.dumps({"error": str(exc), "direction": "NEUTRAL", "confidence": 0.5})

    @tool(
        "Check if consensus confidence meets the threshold. "
        "Returns whether a feedback loop re-run is needed."
    )
    async def check_confidence(self, confidence: float) -> str:
        """Determine if confidence meets the threshold."""
        meets = confidence >= cfg.confidence_threshold
        return json.dumps({
            "meets_threshold": meets,
            "confidence": confidence,
            "threshold": cfg.confidence_threshold,
            "action": "proceed" if meets else "re_run",
        })

    # ── Internal pipeline ──────────────────────────────────────────────────────

    async def _run_pipeline(self, asset: Asset) -> tuple[
        Optional[MarketSearchResult],
        Optional[OHLCVData],
        Optional[OHLCVData],
        Optional[OHLCVData],
        Optional[KronosPrediction],
    ]:
        """Run market search and OHLCV fetch in parallel, then predict."""
        self.log.info("Running pipeline for {}", asset.value)

        # Step 1: Parallel — market search + OHLCV fetch (all timeframes)
        (market_result, ohlcv_1m, ohlcv_5m, ohlcv_15m) = await asyncio.gather(
            self._market_search.run_task(asset=asset),
            self._ohlcv_fetcher.run_task(asset=asset, timeframe=Timeframe.M1, limit=300),
            self._ohlcv_fetcher.run_task(asset=asset, timeframe=Timeframe.M5, limit=500),
            self._ohlcv_fetcher.run_task(asset=asset, timeframe=Timeframe.M15, limit=200),
            return_exceptions=True,
        )

        # Handle exceptions from gather
        def _unwrap(result, cls_name: str):
            if isinstance(result, Exception):
                self.log.error("{} failed: {}", cls_name, result)
                return None
            return result

        market_result = _unwrap(market_result, "MarketSearch")
        ohlcv_1m = _unwrap(ohlcv_1m, "OHLCV_1m")
        ohlcv_5m = _unwrap(ohlcv_5m, "OHLCV_5m")
        ohlcv_15m = _unwrap(ohlcv_15m, "OHLCV_15m")

        # Step 2: Kronos prediction (requires OHLCV)
        kronos_prediction = None
        if ohlcv_5m and ohlcv_5m.bars:
            try:
                kronos_prediction = await self._kronos.run_task(
                    ohlcv_5m=ohlcv_5m,
                    ohlcv_1m=ohlcv_1m,
                    ohlcv_15m=ohlcv_15m,
                )
            except Exception as exc:
                self.log.error("Kronos prediction failed: {}", exc)

        return market_result, ohlcv_1m, ohlcv_5m, ohlcv_15m, kronos_prediction

    def _build_signals(
        self,
        market_result: Optional[MarketSearchResult],
        kronos: Optional[KronosPrediction],
    ) -> list[AgentSignal]:
        """Convert agent outputs to weighted AgentSignal list."""
        signals: list[AgentSignal] = []

        if market_result:
            asset = market_result.asset
            if market_result.implied_prob_up > 0.5:
                signals.append(AgentSignal(
                    agent_name="MarketSearchAgent",
                    asset=asset,
                    direction=Direction.UP,
                    confidence=market_result.implied_prob_up,
                    weight=1.2,  # prediction markets are strong signal
                ))
            elif market_result.implied_prob_down > 0.5:
                signals.append(AgentSignal(
                    agent_name="MarketSearchAgent",
                    asset=asset,
                    direction=Direction.DOWN,
                    confidence=market_result.implied_prob_down,
                    weight=1.2,
                ))

        if kronos:
            asset = kronos.asset
            signals.append(AgentSignal(
                agent_name="KronosPredictorAgent",
                asset=asset,
                direction=kronos.direction,
                confidence=kronos.confidence,
                weight=1.5,  # model prediction gets highest weight
            ))

            # 1m leading signal
            if kronos.n_plus_5_1m:
                signals.append(AgentSignal(
                    agent_name="KronosPredictor_1m_n5",
                    asset=asset,
                    direction=kronos.n_plus_5_1m.direction,
                    confidence=kronos.n_plus_5_1m.confidence,
                    weight=0.8,
                ))

            # 15m arbitrage signal
            if kronos.n_plus_1_15m:
                signals.append(AgentSignal(
                    agent_name="KronosPredictor_15m_n1",
                    asset=asset,
                    direction=kronos.n_plus_1_15m.direction,
                    confidence=kronos.n_plus_1_15m.confidence,
                    weight=1.0,
                ))

        return signals

    async def _compute_consensus(
        self, signals: list[AgentSignal]
    ) -> tuple[Direction, float]:
        """Aggregate signals into consensus direction + confidence."""
        signals_json = json.dumps([
            {
                "direction": s.direction.value,
                "confidence": s.confidence,
                "weight": s.weight,
            }
            for s in signals
        ])
        result_json = await self.aggregate_signals(signals_json)
        result = json.loads(result_json)
        direction = Direction(result.get("direction", "NEUTRAL"))
        confidence = float(result.get("confidence", 0.5))
        return direction, confidence

    async def _llm_narrative(
        self,
        asset: Asset,
        direction: Direction,
        confidence: float,
        signals: list[AgentSignal],
        kelly: Optional[KellyResult],
        market_result: Optional[MarketSearchResult],
        kronos: Optional[KronosPrediction],
        feedback_rounds: int,
    ) -> str:
        """Use LLM to write a concise trading narrative."""
        signal_summary = "\n".join([
            f"- {s.agent_name}: {s.direction.value} (conf={s.confidence:.1%}, weight={s.weight})"
            for s in signals
        ])
        kelly_summary = ""
        if kelly:
            kelly_summary = (
                f"\nKelly sizing: {kelly.recommended_fraction:.1%} of portfolio "
                f"= ${kelly.recommended_usd:,.0f}\n"
                f"Arbitrage signal: {kelly.arbitrage_signal or 'none'}"
            )

        market_summary = ""
        if market_result and market_result.markets:
            market_summary = (
                f"\nPrediction markets: {len(market_result.markets)} found | "
                f"implied P(UP)={market_result.implied_prob_up:.1%}"
            )

        try:
            narrative = await self.llm_loop(
                system_prompt=(
                    "You are a crypto trading analyst. Write a concise 2-3 sentence "
                    "trading signal summary. Be direct and factual. "
                    "Include direction, confidence, key supporting signals, and recommended size. "
                    "Do NOT be verbose."
                ),
                user_prompt=(
                    f"Asset: {asset.value}\n"
                    f"Consensus: {direction.value} (confidence={confidence:.1%})\n"
                    f"Feedback rounds needed: {feedback_rounds}\n"
                    f"Agent signals:\n{signal_summary}\n"
                    f"{market_summary}\n"
                    f"{kelly_summary}\n\n"
                    "Write a concise trading signal summary."
                ),
            )
            return narrative
        except Exception as exc:
            self.log.warning("LLM narrative failed: {}", exc)
            return (
                f"{asset.value} {direction.value} signal with {confidence:.0%} confidence. "
                f"Kelly size: ${kelly.recommended_usd:,.0f}." if kelly else ""
            )

    # ── Main entry point ───────────────────────────────────────────────────────

    async def run_task(self, asset: Asset) -> FinalSignal:
        """
        Run the full Hermes feedback loop for a single asset.
        Returns a FinalSignal with direction, confidence, Kelly sizing, and narrative.
        """
        self.log.info("=== Orchestrator starting for {} ===", asset.value)
        feedback_rounds = 0

        # Initial pipeline run
        market_result, ohlcv_1m, ohlcv_5m, ohlcv_15m, kronos = await self._run_pipeline(asset)
        signals = self._build_signals(market_result, kronos)
        direction, confidence = await self._compute_consensus(signals)

        # Hermes feedback loop: re-run if confidence is below threshold
        while (
            confidence < cfg.confidence_threshold
            and feedback_rounds < cfg.max_feedback_rounds
        ):
            feedback_rounds += 1
            self.log.info(
                "Feedback round {} — confidence {:.1%} below threshold {:.1%}",
                feedback_rounds, confidence, cfg.confidence_threshold
            )

            # Re-run with more bars for better signal quality
            extra_bars = cfg.ohlcv_bars + (feedback_rounds * 200)
            if ohlcv_5m and ohlcv_5m.bars:
                self.log.debug("Re-fetching {} 5m bars for better prediction", extra_bars)
                try:
                    ohlcv_5m = await self._ohlcv_fetcher.run_task(
                        asset=asset, timeframe=Timeframe.M5, limit=extra_bars
                    )
                    kronos = await self._kronos.run_task(
                        ohlcv_5m=ohlcv_5m,
                        ohlcv_1m=ohlcv_1m,
                        ohlcv_15m=ohlcv_15m,
                    )
                    signals = self._build_signals(market_result, kronos)
                    direction, confidence = await self._compute_consensus(signals)
                except Exception as exc:
                    self.log.error("Feedback round {} failed: {}", feedback_rounds, exc)
                    break

        self.log.info(
            "Consensus after {} feedback rounds: {} (conf={:.1%})",
            feedback_rounds, direction.value, confidence
        )

        # Kelly risk sizing
        kelly_result = None
        try:
            kelly_result = await self._kelly.run_task(
                asset=asset,
                direction=direction,
                win_probability=confidence,
                market_search=market_result,
                kronos=kronos,
            )
        except Exception as exc:
            self.log.error("Kelly computation failed: {}", exc)

        # LLM narrative synthesis
        narrative = await self._llm_narrative(
            asset=asset,
            direction=direction,
            confidence=confidence,
            signals=signals,
            kelly=kelly_result,
            market_result=market_result,
            kronos=kronos,
            feedback_rounds=feedback_rounds,
        )

        final = FinalSignal(
            asset=asset,
            direction=direction,
            consensus_confidence=round(confidence, 4),
            agent_signals=signals,
            kelly=kelly_result,
            market_search=market_result,
            kronos_prediction=kronos,
            llm_synthesis=narrative,
            feedback_rounds=feedback_rounds,
        )

        self.log.info(
            "=== FINAL SIGNAL for {}: {} ({:.1%}) | Action: {} ===",
            asset.value, direction.value, confidence, final.action
        )
        return final

    async def run_all_assets(self) -> dict[str, FinalSignal]:
        """Run orchestrator for all configured assets in parallel."""
        self.log.info("Running full pipeline for assets: {}", cfg.assets)
        tasks = {
            asset: self.run_task(Asset(asset))
            for asset in cfg.assets
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        output = {}
        for asset, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                self.log.error("Pipeline failed for {}: {}", asset, result)
            else:
                output[asset] = result
        return output
