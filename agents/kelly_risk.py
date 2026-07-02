"""
agents/kelly_risk.py
─────────────────────
Kelly Risk Management Agent

Responsibilities:
  - Apply Kelly Criterion: f* = (b*p - q) / b
      where p = P(win), q = P(loss) = 1-p, b = reward/risk ratio
  - Compute fractional Kelly (conservative: 0.25× by default)
  - Detect internal arbitrage opportunities:
      * 15m UP signal but 3×5m composite disagrees → arbitrage opportunity
      * 1m n+5 disagrees with 5m n+1 → leading indicator divergence
  - Return recommended position size in USD

Kelly formula reference:
  https://mintlify.wiki/joicodev/polymarket-bot/risk/kelly-criterion
  https://managebankroll.com/blog/polymarket-kelly-criterion-position-sizing
"""

from __future__ import annotations

import json
import math
from typing import Optional

from agents.base import BaseAgent, tool
from core.config import cfg
from core.models import (
    Asset,
    Direction,
    KellyResult,
    KronosPrediction,
    MarketSearchResult,
)


class KellyRiskAgent(BaseAgent):
    """
    Computes Kelly-optimal position sizes and detects arbitrage.
    """

    agent_name = "KellyRiskAgent"

    # ── Tools ──────────────────────────────────────────────────────────────────

    @tool(
        "Compute the Kelly fraction given win probability p, "
        "payout ratio b (reward/risk), and kelly_scale (e.g. 0.25 for quarter Kelly). "
        "Returns the optimal fraction of portfolio to risk."
    )
    async def compute_kelly(
        self, win_prob: float, payout_ratio: float, kelly_scale: float
    ) -> str:
        """Apply Kelly Criterion formula."""
        try:
            p = max(0.01, min(0.99, win_prob))
            q = 1.0 - p
            b = max(0.01, payout_ratio)

            # Full Kelly: f* = (b*p - q) / b
            f_star = (b * p - q) / b
            f_star = max(0.0, f_star)  # Never bet negative

            # Fractional Kelly
            f_recommended = f_star * kelly_scale

            # Cap at 20% of portfolio (hard risk limit)
            f_recommended = min(f_recommended, 0.20)

            return json.dumps({
                "full_kelly": round(f_star, 4),
                "fractional_kelly": round(f_recommended, 4),
                "expected_growth_rate": round(
                    p * math.log(1 + b * f_recommended) + q * math.log(1 - f_recommended), 6
                ) if f_recommended > 0 else 0,
                "ruin_probability_estimate": round(
                    ((1 - f_recommended) / (1 + b * f_recommended)) ** (1 / (2 * f_recommended + 1e-8)), 4
                ) if f_recommended > 0 else 0,
            })
        except Exception as exc:
            return json.dumps({"error": str(exc), "fractional_kelly": 0})

    @tool(
        "Detect internal arbitrage between 15m and compound 5m predictions. "
        "Pass 15m direction, list of 5m directions (JSON array), and their confidences "
        "(JSON array). Returns arbitrage signal as string."
    )
    async def detect_arbitrage(
        self,
        direction_15m: str,
        directions_5m_json: str,
        confidences_5m_json: str,
    ) -> str:
        """
        Check if 15m direction conflicts with compound 3×5m prediction.
        Internal arbitrage exists when:
          - 15m says UP but 2+/3 of 5m predictions say DOWN (or vice versa)
          - 1m n+5 leading signal contradicts 5m n+1
        """
        try:
            dirs_5m = json.loads(directions_5m_json)
            confs_5m = json.loads(confidences_5m_json)

            up_votes = sum(1 for d in dirs_5m if d == "UP")
            down_votes = sum(1 for d in dirs_5m if d == "DOWN")
            avg_conf = sum(confs_5m) / len(confs_5m) if confs_5m else 0.5

            # Compound 5m signal
            compound_dir = "UP" if up_votes > down_votes else "DOWN"
            compound_conf = avg_conf * (max(up_votes, down_votes) / max(len(dirs_5m), 1))

            # Check for arbitrage
            if direction_15m != compound_dir and avg_conf > 0.58:
                return json.dumps({
                    "arbitrage": True,
                    "signal": (
                        f"DIVERGENCE: 15m={direction_15m} vs 5m_compound={compound_dir} "
                        f"(conf={compound_conf:.2f}). "
                        f"Consider fading the 15m signal or waiting for convergence."
                    ),
                    "recommended_action": "REDUCE_SIZE",
                    "size_multiplier": 0.5,
                })
            elif direction_15m == compound_dir and avg_conf > 0.62:
                return json.dumps({
                    "arbitrage": False,
                    "signal": (
                        f"CONVERGENCE: 15m and 5m_compound both={direction_15m} "
                        f"(conf={compound_conf:.2f}). High-conviction signal."
                    ),
                    "recommended_action": "FULL_SIZE",
                    "size_multiplier": 1.0,
                })
            else:
                return json.dumps({
                    "arbitrage": False,
                    "signal": "NEUTRAL — insufficient divergence for arbitrage",
                    "recommended_action": "NORMAL_SIZE",
                    "size_multiplier": 0.75,
                })
        except Exception as exc:
            return json.dumps({"error": str(exc), "arbitrage": False})

    @tool(
        "Estimate the payout ratio (b) for a given market. "
        "Pass market_type as 'binary' (Polymarket/Kalshi YES/NO) or 'crypto' (spot/futures). "
        "Pass yes_price as the current market price (0-1 for binary). "
        "Returns the reward/risk ratio b."
    )
    async def estimate_payout_ratio(
        self, market_type: str, yes_price: float
    ) -> str:
        """Compute b (reward/risk ratio) from market price."""
        try:
            if market_type == "binary":
                # Binary market: win (1 - yes_price) per yes_price risked
                b = (1.0 - yes_price) / max(yes_price, 0.01)
            else:
                # Crypto spot/futures: assume 1:1 for simplicity (can be adjusted)
                b = 1.0
            return json.dumps({"payout_ratio": round(b, 4)})
        except Exception as exc:
            return json.dumps({"error": str(exc), "payout_ratio": 1.0})

    # ── Main entry point ───────────────────────────────────────────────────────

    async def run_task(
        self,
        asset: Asset,
        direction: Direction,
        win_probability: float,
        market_search: Optional[MarketSearchResult] = None,
        kronos: Optional[KronosPrediction] = None,
    ) -> KellyResult:
        """
        Compute Kelly position size and detect arbitrage opportunities.

        win_probability: combined probability from market search + Kronos
        """
        self.log.info(
            "Kelly risk computation for {} | dir={} | p_win={:.1%}",
            asset.value, direction.value, win_probability
        )

        # Determine payout ratio from prediction market prices
        yes_price = 0.5  # Default
        if market_search and market_search.markets:
            # Use average yes_price from markets where direction matches
            matching = [
                m for m in market_search.markets
                if m.direction == direction
            ]
            if matching:
                yes_price = sum(m.yes_price for m in matching) / len(matching)

        payout_json = await self.estimate_payout_ratio("binary", yes_price)
        payout_ratio = json.loads(payout_json).get("payout_ratio", 1.0)

        # Compute Kelly fraction
        kelly_json = await self.compute_kelly(
            win_probability, payout_ratio, cfg.kelly_fraction
        )
        kelly_data = json.loads(kelly_json)
        full_kelly = kelly_data.get("full_kelly", 0)
        recommended_fraction = kelly_data.get("fractional_kelly", 0)
        recommended_usd = recommended_fraction * cfg.portfolio_usd

        self.log.info(
            "Kelly: full={:.1%}, recommended={:.1%} (${:.0f})",
            full_kelly, recommended_fraction, recommended_usd
        )

        # Detect internal arbitrage if we have multi-timeframe data
        arbitrage_signal = None
        if kronos and kronos.n_plus_1_15m:
            dirs_5m = [direction.value]  # Current 5m prediction
            if kronos.n_plus_5_1m:
                dirs_5m.append(kronos.n_plus_5_1m.direction.value)

            arb_json = await self.detect_arbitrage(
                direction_15m=kronos.n_plus_1_15m.direction.value,
                directions_5m_json=json.dumps(dirs_5m),
                confidences_5m_json=json.dumps([
                    kronos.confidence,
                    kronos.n_plus_5_1m.confidence if kronos.n_plus_5_1m else kronos.confidence,
                ]),
            )
            arb_data = json.loads(arb_json)
            arbitrage_signal = arb_data.get("signal")

            # Adjust position size based on arbitrage
            size_mult = arb_data.get("size_multiplier", 1.0)
            recommended_fraction *= size_mult
            recommended_usd *= size_mult
            self.log.info("Arbitrage signal: {}", arbitrage_signal)

        return KellyResult(
            asset=asset,
            direction=direction,
            win_probability=round(win_probability, 4),
            win_payout_ratio=round(payout_ratio, 4),
            kelly_fraction=round(full_kelly, 4),
            recommended_fraction=round(recommended_fraction, 4),
            recommended_usd=round(recommended_usd, 2),
            arbitrage_signal=arbitrage_signal,
        )
