"""
core/config.py
──────────────
Central configuration loaded from .env (or environment variables).
All agents import from here — never import os.environ directly elsewhere.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── LLM ──────────────────────────────────────────────────────────────────
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
    )

    # ── Apify ─────────────────────────────────────────────────────────────────
    apify_api_token: str = field(default_factory=lambda: os.getenv("APIFY_API_TOKEN", ""))

    # ── Polymarket (read-only, no auth needed) ────────────────────────────────
    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"
    polymarket_clob_url: str = "https://clob.polymarket.com"

    # ── Kalshi (read-only, no auth needed for public endpoints) ───────────────
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"

    # ── Kronos model ──────────────────────────────────────────────────────────
    kronos_model: str = field(
        default_factory=lambda: os.getenv("KRONOS_MODEL", "mock")
    )

    # ── Risk / Kelly ──────────────────────────────────────────────────────────
    kelly_fraction: float = field(
        default_factory=lambda: float(os.getenv("KELLY_FRACTION", "0.25"))
    )
    min_confidence: float = field(
        default_factory=lambda: float(os.getenv("MIN_CONFIDENCE", "0.55"))
    )
    portfolio_usd: float = field(
        default_factory=lambda: float(os.getenv("PORTFOLIO_USD", "10000"))
    )

    # ── Orchestrator feedback loop ─────────────────────────────────────────────
    max_feedback_rounds: int = 3          # Re-run loop if confidence is below threshold
    confidence_threshold: float = 0.60   # Target consensus confidence

    # ── Supported assets ──────────────────────────────────────────────────────
    assets: list = field(default_factory=lambda: ["BTC", "ETH"])

    # ── Apify actor IDs ───────────────────────────────────────────────────────
    # CryptoCompare scraper actor (free on Apify)
    apify_crypto_actor: str = "lulzasaur/cryptocompare-scraper"
    # OHLCV bars to fetch
    ohlcv_bars: int = 1000

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = config is valid)."""
        errors = []
        if not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY is not set")
        if not self.apify_api_token:
            errors.append("APIFY_API_TOKEN is not set")
        return errors


# Singleton instance used across the entire app
cfg = Config()
