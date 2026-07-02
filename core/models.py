"""
core/models.py
──────────────
Shared Pydantic data models passed between agents.
Using Pydantic v2 syntax throughout.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class Asset(str, Enum):
    BTC = "BTC"
    ETH = "ETH"


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"


# ── OHLCV ─────────────────────────────────────────────────────────────────────

class OHLCVBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class OHLCVData(BaseModel):
    asset: Asset
    timeframe: Timeframe
    bars: list[OHLCVBar]
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    source: str = "apify"

    @property
    def closes(self) -> list[float]:
        return [b.close for b in self.bars]

    @property
    def latest_price(self) -> float:
        return self.bars[-1].close if self.bars else 0.0


# ── Market Search Results ──────────────────────────────────────────────────────

class PredictionMarket(BaseModel):
    platform: str                    # "polymarket" | "kalshi"
    market_id: str
    question: str
    asset: Asset
    direction: Direction             # UP or DOWN
    yes_price: float                 # 0-1, implied probability of direction=UP
    no_price: float
    volume_24h: Optional[float] = None
    expiry: Optional[datetime] = None
    url: Optional[str] = None


class MarketSearchResult(BaseModel):
    asset: Asset
    markets: list[PredictionMarket]
    implied_prob_up: float           # Weighted average from all found markets
    implied_prob_down: float
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def consensus_direction(self) -> Direction:
        if self.implied_prob_up > self.implied_prob_down + 0.02:
            return Direction.UP
        elif self.implied_prob_down > self.implied_prob_up + 0.02:
            return Direction.DOWN
        return Direction.NEUTRAL


# ── Kronos Prediction ─────────────────────────────────────────────────────────

class KronosPrediction(BaseModel):
    asset: Asset
    timeframe: Timeframe
    direction: Direction
    confidence: float                # 0.0 – 1.0
    predicted_price: Optional[float] = None
    price_change_pct: Optional[float] = None
    method: str = "kronos"           # "kronos" | "fallback_trend" | "fallback_lstm"
    predicted_at: datetime = Field(default_factory=datetime.utcnow)

    # Multi-timeframe cascade predictions
    n_plus_5_1m: Optional["KronosPrediction"] = None   # 1m × 5 ahead
    n_plus_1_15m: Optional["KronosPrediction"] = None  # 15m × 1 ahead


# ── Kelly Risk ────────────────────────────────────────────────────────────────

class KellyResult(BaseModel):
    asset: Asset
    direction: Direction
    win_probability: float           # p
    win_payout_ratio: float          # b (reward/risk)
    kelly_fraction: float            # f* = (bp - q) / b
    recommended_fraction: float      # f* × kelly_scale_factor
    recommended_usd: float           # recommended_fraction × portfolio_usd
    arbitrage_signal: Optional[str] = None
    computed_at: datetime = Field(default_factory=datetime.utcnow)


# ── Orchestrator / Final Signal ───────────────────────────────────────────────

class AgentSignal(BaseModel):
    agent_name: str
    asset: Asset
    direction: Direction
    confidence: float
    weight: float = 1.0


class FinalSignal(BaseModel):
    asset: Asset
    direction: Direction
    consensus_confidence: float
    agent_signals: list[AgentSignal]
    kelly: Optional[KellyResult] = None
    market_search: Optional[MarketSearchResult] = None
    kronos_prediction: Optional[KronosPrediction] = None
    llm_synthesis: Optional[str] = None
    feedback_rounds: int = 0
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def action(self) -> str:
        if self.consensus_confidence < 0.55:
            return "HOLD — low confidence"
        return f"{'LONG' if self.direction == Direction.UP else 'SHORT'} {self.asset.value}"
