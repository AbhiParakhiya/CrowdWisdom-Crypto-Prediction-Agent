"""
agents/ohlcv_fetcher.py
────────────────────────
OHLCV Fetcher Agent

Responsibilities:
  - Fetch last N OHLCV bars for a given crypto asset using Apify
  - Primary: Apify actor 'lulzasaur/cryptocompare-scraper'
  - Fallback: ccxt (Binance public API, no auth needed)
  - Return structured OHLCVData

The Apify actor runs in the cloud and returns JSON.
We poll the run status until it completes, then read the dataset.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from apify_client import ApifyClient
from apify_client.errors import ApifyApiError

from agents.base import BaseAgent, tool
from core.config import cfg
from core.models import Asset, OHLCVBar, OHLCVData, Timeframe

# Mapping of Apify timeframe codes to our Timeframe enum
_TIMEFRAME_MAP = {
    Timeframe.M1: "histominute",
    Timeframe.M5: "histominute",   # We'll request 5x bars and resample
    Timeframe.M15: "histominute",  # Same approach
}

# ccxt symbol map
_CCXT_SYMBOL = {
    Asset.BTC: "BTC/USDT",
    Asset.ETH: "ETH/USDT",
}

# CryptoCompare symbol map
_CC_SYMBOL = {
    Asset.BTC: "BTC",
    Asset.ETH: "ETH",
}


class OHLCVFetcherAgent(BaseAgent):
    """
    Fetches historical OHLCV bars for a given asset.
    Uses Apify as primary source, ccxt as fallback.
    """

    agent_name = "OHLCVFetcherAgent"

    # ── Tools ──────────────────────────────────────────────────────────────────

    @tool(
        "Fetch OHLCV data from Apify using the CryptoCompare scraper actor. "
        "Pass symbol as 'BTC' or 'ETH' and limit as number of bars (e.g. 1000)."
    )
    async def fetch_via_apify(self, symbol: str, limit: int) -> str:
        """Run Apify CryptoCompare actor and return OHLCV bars as JSON."""
        self.log.info("Fetching {} bars for {} via Apify", limit, symbol)

        if not cfg.apify_api_token:
            return json.dumps({"error": "APIFY_API_TOKEN not set", "bars": []})

        try:
            client = ApifyClient(cfg.apify_api_token)

            # Run the actor synchronously (wait for it to finish)
            run_input = {
                "fsym": symbol.upper(),
                "tsym": "USD",
                "limit": min(limit, 2000),
                "aggregate": 1,
                "type": "histominute",
            }

            self.log.debug("Starting Apify actor with input: {}", run_input)
            run = client.actor(cfg.apify_crypto_actor).call(run_input=run_input)

            if not run:
                return json.dumps({"error": "Actor run returned None", "bars": []})

            # Read from default dataset
            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                return json.dumps({"error": "No dataset ID in run result", "bars": []})

            items = list(
                client.dataset(dataset_id).iterate_items()
            )
            self.log.info("Apify returned {} raw items for {}", len(items), symbol)
            return json.dumps({"bars": items[:limit]})

        except ApifyApiError as exc:
            self.log.error("Apify API error: {}", exc)
            return json.dumps({"error": str(exc), "bars": []})
        except Exception as exc:
            self.log.error("Unexpected Apify error: {}", exc)
            return json.dumps({"error": str(exc), "bars": []})

    @tool(
        "Fetch OHLCV data from Binance via ccxt (fallback, no API key needed). "
        "Pass symbol as 'BTC' or 'ETH' and limit as number of 1-minute bars."
    )
    async def fetch_via_ccxt(self, symbol: str, limit: int) -> str:
        """Fallback OHLCV fetch using ccxt Binance public API."""
        self.log.info("Fetching {} bars for {} via ccxt (Binance)", limit, symbol)
        try:
            import ccxt.async_support as ccxt

            exchange = ccxt.binance({"enableRateLimit": True})
            try:
                ccxt_symbol = f"{symbol.upper()}/USDT"
                bars = await exchange.fetch_ohlcv(
                    ccxt_symbol, timeframe="1m", limit=min(limit, 1000)
                )
                # bars: [[timestamp_ms, open, high, low, close, volume], ...]
                result = [
                    {
                        "time": b[0] // 1000,
                        "open": b[1],
                        "high": b[2],
                        "low": b[3],
                        "close": b[4],
                        "volumefrom": b[5],
                    }
                    for b in bars
                ]
                self.log.info("ccxt returned {} bars for {}", len(result), symbol)
                return json.dumps({"bars": result})
            finally:
                await exchange.close()

        except Exception as exc:
            self.log.error("ccxt fetch failed: {}", exc)
            return json.dumps({"error": str(exc), "bars": []})

    # ── Parsing helpers ────────────────────────────────────────────────────────

    def _parse_bars(self, raw_json: str) -> list[OHLCVBar]:
        """Parse raw JSON (from Apify or ccxt) into a list of OHLCVBar."""
        bars: list[OHLCVBar] = []
        try:
            data = json.loads(raw_json)
            raw_bars = data.get("bars", [])
            for b in raw_bars:
                try:
                    # Handle both Apify CryptoCompare and ccxt format
                    ts_raw = b.get("time") or b.get("timestamp") or b.get("t")
                    if ts_raw is None:
                        continue
                    # CryptoCompare returns Unix seconds; ccxt returns ms
                    ts = int(ts_raw)
                    if ts > 10_000_000_000:  # milliseconds
                        ts = ts // 1000
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)

                    bars.append(
                        OHLCVBar(
                            timestamp=dt,
                            open=float(b.get("open") or b.get("o") or 0),
                            high=float(b.get("high") or b.get("h") or 0),
                            low=float(b.get("low") or b.get("l") or 0),
                            close=float(b.get("close") or b.get("c") or 0),
                            volume=float(b.get("volumefrom") or b.get("v") or 0),
                        )
                    )
                except (ValueError, TypeError, KeyError) as e:
                    self.log.debug("Skipping malformed bar: {}", e)
                    continue
        except Exception as exc:
            self.log.error("Failed to parse bars JSON: {}", exc)
        return bars

    def _resample_to_timeframe(
        self, bars: list[OHLCVBar], timeframe: Timeframe
    ) -> list[OHLCVBar]:
        """Resample 1-minute bars into 5m or 15m bars."""
        if timeframe == Timeframe.M1:
            return bars

        n = 5 if timeframe == Timeframe.M5 else 15
        resampled = []
        for i in range(0, len(bars) - n + 1, n):
            chunk = bars[i : i + n]
            if not chunk:
                continue
            resampled.append(
                OHLCVBar(
                    timestamp=chunk[0].timestamp,
                    open=chunk[0].open,
                    high=max(b.high for b in chunk),
                    low=min(b.low for b in chunk),
                    close=chunk[-1].close,
                    volume=sum(b.volume for b in chunk),
                )
            )
        return resampled

    # ── Main entry point ───────────────────────────────────────────────────────

    async def run_task(
        self,
        asset: Asset,
        timeframe: Timeframe = Timeframe.M5,
        limit: int = None,
    ) -> OHLCVData:
        """
        Fetch OHLCV bars for the given asset and timeframe.
        Tries Apify first; falls back to ccxt if Apify fails or token is missing.
        """
        if limit is None:
            limit = cfg.ohlcv_bars

        symbol = _CC_SYMBOL[asset]
        self.log.info("Fetching {} {} bars for {}", limit, timeframe.value, asset.value)

        # Try Apify first
        apify_json = await self.fetch_via_apify(symbol, limit)
        bars = self._parse_bars(apify_json)

        # Fallback to ccxt if Apify returned nothing
        if not bars:
            self.log.warning("Apify returned no bars — falling back to ccxt")
            ccxt_json = await self.fetch_via_ccxt(symbol, limit)
            bars = self._parse_bars(ccxt_json)

        if not bars:
            self.log.error("Both Apify and ccxt failed to return bars for {}", asset.value)
            # Return empty data — prediction agents will handle this gracefully
            return OHLCVData(asset=asset, timeframe=timeframe, bars=[], source="error")

        # Sort by timestamp ascending
        bars.sort(key=lambda b: b.timestamp)

        # Resample to requested timeframe
        if timeframe != Timeframe.M1:
            bars = self._resample_to_timeframe(bars, timeframe)

        # Keep only the last `limit` bars
        bars = bars[-limit:]

        source = "apify" if len(self._parse_bars(apify_json)) > 0 else "ccxt"
        self.log.info(
            "Returning {} {} bars for {} (source: {})",
            len(bars), timeframe.value, asset.value, source
        )
        return OHLCVData(
            asset=asset,
            timeframe=timeframe,
            bars=bars,
            source=source,
        )
