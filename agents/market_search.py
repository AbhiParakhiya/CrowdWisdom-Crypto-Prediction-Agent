"""
agents/market_search.py
────────────────────────
Market Search Agent

Responsibilities:
  - Search Polymarket (Gamma API) for BTC/ETH 5-min prediction markets
  - Search Kalshi for equivalent crypto direction markets
  - Extract implied probability of UP / DOWN for each asset
  - Return a MarketSearchResult with aggregated probabilities

The agent uses a Hermes ReAct loop: the LLM decides which API to call,
interprets results, and synthesises a final probability estimate.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.base import BaseAgent, tool
from core.config import cfg
from core.models import (
    Asset,
    Direction,
    MarketSearchResult,
    PredictionMarket,
    Timeframe,
)

# Keyword sets for matching crypto markets
_BTC_KEYWORDS = {"bitcoin", "btc", "btcusd", "bitcoin price"}
_ETH_KEYWORDS = {"ethereum", "eth", "ethusd", "ethereum price"}


def _asset_from_text(text: str) -> Optional[Asset]:
    t = text.lower()
    if any(k in t for k in _BTC_KEYWORDS):
        return Asset.BTC
    if any(k in t for k in _ETH_KEYWORDS):
        return Asset.ETH
    return None


def _direction_from_text(text: str) -> Direction:
    t = text.lower()
    up_words = {"above", "higher", "up", "rise", "bull", "exceed", "over", "+"}
    down_words = {"below", "lower", "down", "fall", "bear", "under", "-"}
    up_score = sum(1 for w in up_words if w in t)
    down_score = sum(1 for w in down_words if w in t)
    if up_score > down_score:
        return Direction.UP
    if down_score > up_score:
        return Direction.DOWN
    return Direction.UP  # default assumption for "will X reach Y" style markets


class MarketSearchAgent(BaseAgent):
    """
    Searches Polymarket and Kalshi for crypto directional prediction markets,
    then synthesises an implied probability for BTC/ETH up/down moves.
    """

    agent_name = "MarketSearchAgent"

    # ── Tools ──────────────────────────────────────────────────────────────────

    @tool("Search Polymarket Gamma API for crypto prediction markets. "
          "Pass 'btc' or 'eth' as the query.")
    async def search_polymarket(self, query: str) -> str:
        """Fetch open Polymarket markets matching the query."""
        self.log.info("Searching Polymarket for '{}'", query)
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{cfg.polymarket_gamma_url}/markets",
                    params={
                        "keyword": query,
                        "active": "true",
                        "closed": "false",
                        "limit": 20,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            # data is a list of market dicts
            markets = data if isinstance(data, list) else data.get("data", [])
            simplified = []
            for m in markets[:10]:
                simplified.append({
                    "id": m.get("id", ""),
                    "question": m.get("question", ""),
                    "outcomePrices": m.get("outcomePrices", []),
                    "volume": m.get("volume", 0),
                    "endDate": m.get("endDate", ""),
                    "slug": m.get("slug", ""),
                })
            return json.dumps(simplified, default=str)
        except Exception as exc:
            self.log.warning("Polymarket search failed: {}", exc)
            return json.dumps({"error": str(exc), "markets": []})

    @tool("Search Kalshi prediction markets for crypto events. "
          "Pass 'bitcoin' or 'ethereum' as the query.")
    async def search_kalshi(self, query: str) -> str:
        """Fetch Kalshi markets matching the query (read-only, no auth needed)."""
        self.log.info("Searching Kalshi for '{}'", query)
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{cfg.kalshi_base_url}/markets",
                    params={
                        "keyword": query,
                        "limit": 20,
                        "status": "open",
                    },
                    headers={"Accept": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()

            markets = data.get("markets", [])
            simplified = []
            for m in markets[:10]:
                simplified.append({
                    "id": m.get("ticker", ""),
                    "title": m.get("title", ""),
                    "yes_bid": m.get("yes_bid", 0),
                    "yes_ask": m.get("yes_ask", 0),
                    "volume": m.get("volume", 0),
                    "close_time": m.get("close_time", ""),
                })
            return json.dumps(simplified, default=str)
        except Exception as exc:
            self.log.warning("Kalshi search failed: {}", exc)
            return json.dumps({"error": str(exc), "markets": []})

    @tool("Get the current best-bid/ask from Polymarket CLOB for a specific condition_id. "
          "Returns the mid-price (0-1) as the implied probability.")
    async def get_polymarket_price(self, condition_id: str) -> str:
        """Fetch Polymarket CLOB order book price for a token."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{cfg.polymarket_clob_url}/book",
                    params={"token_id": condition_id},
                )
                resp.raise_for_status()
                book = resp.json()
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            best_bid = float(bids[0]["price"]) if bids else 0.5
            best_ask = float(asks[0]["price"]) if asks else 0.5
            mid = (best_bid + best_ask) / 2
            return json.dumps({"mid_price": mid, "best_bid": best_bid, "best_ask": best_ask})
        except Exception as exc:
            return json.dumps({"error": str(exc), "mid_price": 0.5})

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _parse_polymarket_markets(
        self, raw_json: str, asset: Asset
    ) -> list[PredictionMarket]:
        """Parse Polymarket JSON into PredictionMarket objects."""
        markets = []
        try:
            data = json.loads(raw_json)
            if isinstance(data, dict) and "error" in data:
                return markets
            for m in data:
                q = m.get("question", "")
                if _asset_from_text(q) != asset:
                    continue
                direction = _direction_from_text(q)
                prices = m.get("outcomePrices", ["0.5", "0.5"])
                try:
                    yes_price = float(prices[0]) if prices else 0.5
                    no_price = float(prices[1]) if len(prices) > 1 else 1 - yes_price
                except (ValueError, TypeError):
                    yes_price, no_price = 0.5, 0.5
                markets.append(
                    PredictionMarket(
                        platform="polymarket",
                        market_id=m.get("id", ""),
                        question=q,
                        asset=asset,
                        direction=direction,
                        yes_price=yes_price,
                        no_price=no_price,
                        volume_24h=float(m.get("volume", 0) or 0),
                        url=f"https://polymarket.com/event/{m.get('slug', '')}",
                    )
                )
        except Exception as exc:
            self.log.warning("Error parsing Polymarket markets: {}", exc)
        return markets

    def _parse_kalshi_markets(
        self, raw_json: str, asset: Asset
    ) -> list[PredictionMarket]:
        """Parse Kalshi JSON into PredictionMarket objects."""
        markets = []
        try:
            data = json.loads(raw_json)
            if isinstance(data, dict) and "error" in data:
                return markets
            for m in data:
                title = m.get("title", "")
                if _asset_from_text(title) != asset:
                    continue
                direction = _direction_from_text(title)
                yes_bid = float(m.get("yes_bid", 50)) / 100
                yes_ask = float(m.get("yes_ask", 50)) / 100
                yes_price = (yes_bid + yes_ask) / 2
                markets.append(
                    PredictionMarket(
                        platform="kalshi",
                        market_id=m.get("id", ""),
                        question=title,
                        asset=asset,
                        direction=direction,
                        yes_price=yes_price,
                        no_price=1.0 - yes_price,
                        volume_24h=float(m.get("volume", 0) or 0),
                        url=f"https://kalshi.com/markets/{m.get('id', '')}",
                    )
                )
        except Exception as exc:
            self.log.warning("Error parsing Kalshi markets: {}", exc)
        return markets

    def _aggregate_probability(self, markets: list[PredictionMarket]) -> tuple[float, float]:
        """Volume-weighted average implied probability of UP and DOWN."""
        if not markets:
            return 0.5, 0.5

        total_vol = sum(m.volume_24h or 1.0 for m in markets)
        if total_vol == 0:
            total_vol = len(markets)

        weighted_up = 0.0
        weighted_down = 0.0
        for m in markets:
            w = (m.volume_24h or 1.0) / total_vol
            if m.direction == Direction.UP:
                weighted_up += m.yes_price * w
                weighted_down += m.no_price * w
            else:
                weighted_down += m.yes_price * w
                weighted_up += m.no_price * w

        total = weighted_up + weighted_down
        if total == 0:
            return 0.5, 0.5
        return weighted_up / total, weighted_down / total

    # ── Main entry point ───────────────────────────────────────────────────────

    async def run_task(self, asset: Asset) -> MarketSearchResult:
        """
        Run the market search agent for the given asset.
        Uses the LLM loop to decide which searches to run and interpret results,
        then parses raw API responses into structured objects.
        """
        self.log.info("Starting market search for {}", asset.value)

        query_btc = "bitcoin" if asset == Asset.BTC else "ethereum"
        query_eth = asset.value.lower()

        # Run both platform searches in parallel via direct tool calls
        # (no need to burn LLM tokens for simple parallel fetches)
        poly_json = await self.search_polymarket(query_eth)
        kalshi_json = await self.search_kalshi(query_btc)

        poly_markets = self._parse_polymarket_markets(poly_json, asset)
        kalshi_markets = self._parse_kalshi_markets(kalshi_json, asset)
        all_markets = poly_markets + kalshi_markets

        self.log.info(
            "Found {} Polymarket + {} Kalshi markets for {}",
            len(poly_markets), len(kalshi_markets), asset.value
        )

        # Use LLM only for synthesis if we have real markets
        if all_markets:
            market_summary = json.dumps(
                [
                    {
                        "platform": m.platform,
                        "question": m.question,
                        "direction": m.direction,
                        "yes_price": round(m.yes_price, 3),
                        "volume_24h": m.volume_24h,
                    }
                    for m in all_markets
                ],
                indent=2,
            )
            llm_response = await self.llm_loop(
                system_prompt=(
                    "You are a prediction market analyst. "
                    "Given a list of crypto prediction markets, estimate the consensus "
                    "implied probability of the asset going UP in the next 5 minutes. "
                    "Return ONLY a JSON object: "
                    '{\"prob_up\": 0.XX, \"prob_down\": 0.XX, \"reasoning\": \"...\"}'
                ),
                user_prompt=(
                    f"Asset: {asset.value}\n"
                    f"Current prediction markets:\n{market_summary}\n\n"
                    "What is the implied probability of UP vs DOWN for the next 5 minutes?"
                ),
            )
            try:
                # Extract JSON from LLM response
                import re
                json_match = re.search(r"\{.*\}", llm_response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    prob_up = float(parsed.get("prob_up", 0.5))
                    prob_down = float(parsed.get("prob_down", 1 - prob_up))
                else:
                    prob_up, prob_down = self._aggregate_probability(all_markets)
            except Exception:
                prob_up, prob_down = self._aggregate_probability(all_markets)
        else:
            # Fallback: no markets found, return neutral
            self.log.warning("No markets found for {}, using neutral 50/50", asset.value)
            prob_up, prob_down = 0.5, 0.5

        return MarketSearchResult(
            asset=asset,
            markets=all_markets,
            implied_prob_up=round(prob_up, 4),
            implied_prob_down=round(prob_down, 4),
        )
