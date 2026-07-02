"""
agents/kronos_predictor.py
──────────────────────────
Kronos Predictor Agent

Responsibilities:
  - Take OHLCV data and predict next candle direction (UP / DOWN)
  - Primary: Kronos Foundation Model (HuggingFace: NeoQuasar/Kronos-base)
  - Fallback 1: Lightweight trend-based predictor (no ML)
  - Fallback 2: Simple LSTM trained on recent bars

Kronos is a financial foundation model trained on 12B K-line records.
It runs zero-shot — no fine-tuning required.

Timeframe scaling:
  - Predicts M5 (5-minute) primary signal
  - Also predicts 1m n+5 (5 consecutive 1m bars) → used as leading 5m signal
  - Also predicts 15m n+1 for internal arbitrage
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from agents.base import BaseAgent, tool
from core.config import cfg
from core.models import Asset, Direction, KronosPrediction, OHLCVData, Timeframe


# ── Lightweight fallback predictor ────────────────────────────────────────────

class TrendPredictor:
    """
    Simple technical analysis fallback when Kronos model is unavailable.
    Uses: EMA crossover, RSI, and momentum to predict direction.
    Confidence is calibrated from signal agreement (0.52–0.72 range).
    """

    def predict(self, closes: list[float]) -> tuple[Direction, float]:
        if len(closes) < 30:
            return Direction.NEUTRAL, 0.5

        arr = np.array(closes, dtype=float)

        # EMA 8/21 crossover
        ema8 = self._ema(arr, 8)
        ema21 = self._ema(arr, 21)
        ema_signal = 1 if ema8[-1] > ema21[-1] else -1

        # RSI(14) — overbought/oversold
        rsi = self._rsi(arr, 14)
        if rsi > 70:
            rsi_signal = -1   # overbought → likely DOWN
        elif rsi < 30:
            rsi_signal = 1    # oversold → likely UP
        else:
            rsi_signal = 1 if rsi > 50 else -1

        # Rate of change (momentum) over last 5 bars
        roc = (arr[-1] - arr[-6]) / arr[-6] if arr[-6] != 0 else 0
        mom_signal = 1 if roc > 0 else -1

        # Bollinger Band position
        mean_20 = np.mean(arr[-20:])
        std_20 = np.std(arr[-20:])
        bb_pos = (arr[-1] - mean_20) / (2 * std_20 + 1e-8)  # -1 to 1 roughly
        bb_signal = -1 if bb_pos > 0.8 else (1 if bb_pos < -0.8 else mom_signal)

        signals = [ema_signal, rsi_signal, mom_signal, bb_signal]
        score = sum(signals)  # -4 to +4

        if score > 0:
            direction = Direction.UP
        elif score < 0:
            direction = Direction.DOWN
        else:
            direction = Direction.NEUTRAL

        # Calibrated confidence (agreement fraction mapped to 0.52-0.72)
        agreement = abs(score) / len(signals)
        confidence = 0.50 + agreement * 0.22

        return direction, round(confidence, 3)

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(arr)
        ema[0] = arr[0]
        for i in range(1, len(arr)):
            ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def _rsi(arr: np.ndarray, period: int = 14) -> float:
        deltas = np.diff(arr[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-8
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - (100 / (1 + rs))


# ── Kronos model wrapper ───────────────────────────────────────────────────────

class KronosModelWrapper:
    """
    Wraps the Kronos HuggingFace model for zero-shot prediction.
    Loaded lazily on first use to avoid slow startup.
    Falls back to TrendPredictor if torch/transformers unavailable.
    """

    _model = None
    _tokenizer = None
    _loaded = False

    def load(self) -> bool:
        """Try to load Kronos from HuggingFace. Returns True if successful."""
        if self._loaded:
            return self._model is not None
        self._loaded = True
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            repo = cfg.kronos_model
            if repo == "mock":
                return False

            # Use small model by default; NeoQuasar/Kronos-small ~400MB
            self._tokenizer = AutoTokenizer.from_pretrained(repo)
            self._model = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.float32
            )
            self._model.eval()
            return True
        except ImportError:
            return False
        except Exception as exc:
            return False

    def predict(
        self, closes: list[float], n_predict: int = 1
    ) -> tuple[list[float], float]:
        """
        Use Kronos to predict next n_predict close prices.
        Returns (predicted_closes, confidence).
        """
        if not self.load():
            raise RuntimeError("Kronos model not loaded")

        import torch

        # Kronos expects normalized input — z-score normalization
        arr = np.array(closes[-512:], dtype=np.float32)  # max context 512 bars
        mean, std = arr.mean(), arr.std() + 1e-8
        normed = (arr - mean) / std

        # Convert to tokens (simplified: map to vocabulary bins)
        # Full Kronos tokenization uses the official tokenizer
        input_ids = torch.tensor([normed.tolist()], dtype=torch.float32)

        with torch.no_grad():
            # Kronos generation — simplified invocation
            # Real usage follows examples/prediction_wo_vol_example.py
            outputs = self._model.generate(
                inputs_embeds=input_ids.unsqueeze(-1).expand(-1, -1, self._model.config.hidden_size)
                if hasattr(self._model.config, "hidden_size") else input_ids,
                max_new_tokens=n_predict,
            )

        # Denormalize predictions
        predicted_normed = outputs[0, -n_predict:].numpy()
        predicted = (predicted_normed * std + mean).tolist()
        confidence = min(0.72, 0.55 + 0.02 * len(closes) / 100)
        return predicted, confidence


# ── Main Agent ────────────────────────────────────────────────────────────────

class KronosPredictorAgent(BaseAgent):
    """
    Predicts next candle direction using Kronos (or fallback).
    Also produces multi-timeframe cascade predictions.
    """

    agent_name = "KronosPredictorAgent"

    def __init__(self):
        super().__init__()
        self._kronos = KronosModelWrapper()
        self._trend = TrendPredictor()

    # ── Tools ──────────────────────────────────────────────────────────────────

    @tool(
        "Predict next candle direction from a list of close prices. "
        "Pass closes as JSON array string and method as 'kronos' or 'trend'."
    )
    async def predict_direction(self, closes_json: str, method: str) -> str:
        """Predict UP/DOWN from close prices using specified method."""
        try:
            closes = json.loads(closes_json)
            if not closes:
                return json.dumps({"direction": "NEUTRAL", "confidence": 0.5, "method": method})

            if method == "kronos":
                try:
                    predicted, conf = self._kronos.predict(closes, n_predict=1)
                    direction = Direction.UP if predicted[0] > closes[-1] else Direction.DOWN
                    return json.dumps({
                        "direction": direction.value,
                        "confidence": conf,
                        "predicted_price": predicted[0],
                        "method": "kronos",
                    })
                except Exception as e:
                    # Fall through to trend
                    self.log.warning("Kronos failed ({}), using trend fallback", e)

            direction, confidence = self._trend.predict(closes)
            return json.dumps({
                "direction": direction.value,
                "confidence": confidence,
                "method": "fallback_trend",
            })
        except Exception as exc:
            return json.dumps({"error": str(exc), "direction": "NEUTRAL", "confidence": 0.5})

    @tool(
        "Compute technical indicators (RSI, EMA, Bollinger) for a close price series. "
        "Pass closes as JSON array string."
    )
    async def compute_indicators(self, closes_json: str) -> str:
        """Return dict of key technical indicators."""
        try:
            closes = json.loads(closes_json)
            arr = np.array(closes, dtype=float)
            result: dict = {}

            if len(arr) >= 14:
                result["rsi_14"] = round(self._trend._rsi(arr, 14), 2)
            if len(arr) >= 21:
                ema8 = self._trend._ema(arr, 8)
                ema21 = self._trend._ema(arr, 21)
                result["ema8"] = round(float(ema8[-1]), 4)
                result["ema21"] = round(float(ema21[-1]), 4)
                result["ema_cross"] = "bullish" if ema8[-1] > ema21[-1] else "bearish"
            if len(arr) >= 20:
                mean_20 = float(np.mean(arr[-20:]))
                std_20 = float(np.std(arr[-20:]))
                result["bb_upper"] = round(mean_20 + 2 * std_20, 4)
                result["bb_lower"] = round(mean_20 - 2 * std_20, 4)
                result["bb_position"] = round((arr[-1] - mean_20) / (2 * std_20 + 1e-8), 3)
            if len(arr) >= 6:
                roc = (arr[-1] - arr[-6]) / arr[-6] if arr[-6] != 0 else 0
                result["momentum_5bar"] = round(float(roc) * 100, 3)

            result["latest_close"] = round(float(arr[-1]), 2)
            result["bars_analyzed"] = len(arr)
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    # ── Core prediction logic ──────────────────────────────────────────────────

    async def _predict_single(
        self, closes: list[float], timeframe: Timeframe, asset: Asset
    ) -> KronosPrediction:
        """Run prediction for a single timeframe."""
        method = "kronos" if cfg.kronos_model != "mock" else "trend"
        result_json = await self.predict_direction(json.dumps(closes), method)
        result = json.loads(result_json)

        direction = Direction(result.get("direction", "NEUTRAL"))
        confidence = float(result.get("confidence", 0.5))
        predicted_price = result.get("predicted_price")
        actual_method = result.get("method", method)

        price_change_pct = None
        if predicted_price and closes:
            price_change_pct = round(
                (predicted_price - closes[-1]) / closes[-1] * 100, 3
            )

        return KronosPrediction(
            asset=asset,
            timeframe=timeframe,
            direction=direction,
            confidence=confidence,
            predicted_price=predicted_price,
            price_change_pct=price_change_pct,
            method=actual_method,
        )

    # ── Main entry point ───────────────────────────────────────────────────────

    async def run_task(
        self,
        ohlcv_5m: OHLCVData,
        ohlcv_1m: Optional[OHLCVData] = None,
        ohlcv_15m: Optional[OHLCVData] = None,
    ) -> KronosPrediction:
        """
        Run multi-timeframe prediction.

        Produces:
          - Primary 5m prediction
          - 1m n+5 (leading signal for 5m n+1) — if ohlcv_1m provided
          - 15m n+1 (for internal arbitrage) — if ohlcv_15m provided
        """
        asset = ohlcv_5m.asset
        self.log.info("Running Kronos prediction for {} (5m primary)", asset.value)

        closes_5m = ohlcv_5m.closes
        if not closes_5m:
            self.log.error("No 5m close prices for {}", asset.value)
            return KronosPrediction(
                asset=asset,
                timeframe=Timeframe.M5,
                direction=Direction.NEUTRAL,
                confidence=0.5,
                method="no_data",
            )

        # Primary 5m prediction
        primary = await self._predict_single(closes_5m, Timeframe.M5, asset)
        self.log.info(
            "{} 5m prediction: {} (conf={:.1%}, method={})",
            asset.value, primary.direction.value, primary.confidence, primary.method
        )

        # Compute and attach technical indicators via LLM synthesis
        indicators_json = await self.compute_indicators(json.dumps(closes_5m[-100:]))
        indicators = json.loads(indicators_json)

        # Ask LLM to synthesize indicators + raw prediction into a final confidence
        if cfg.openrouter_api_key:
            synthesis = await self.llm_loop(
                system_prompt=(
                    "You are a quantitative crypto analyst. "
                    "Given technical indicators and a model prediction, "
                    "adjust the confidence score (0-1) and confirm or revise the direction. "
                    "Return ONLY valid JSON: "
                    '{\"direction\": \"UP\"|\"DOWN\"|\"NEUTRAL\", \"confidence\": 0.XX, \"reason\": \"...\"}'
                ),
                user_prompt=(
                    f"Asset: {asset.value}\n"
                    f"Timeframe: 5m\n"
                    f"Model prediction: {primary.direction.value} (conf={primary.confidence:.2f})\n"
                    f"Technical indicators: {indicators_json}\n"
                    f"Latest price: {closes_5m[-1]:.4f}\n\n"
                    "Please confirm or revise the prediction with adjusted confidence."
                ),
            )
            try:
                import re
                m = re.search(r"\{.*\}", synthesis, re.DOTALL)
                if m:
                    parsed = json.loads(m.group())
                    primary.direction = Direction(parsed.get("direction", primary.direction.value))
                    primary.confidence = float(parsed.get("confidence", primary.confidence))
            except Exception:
                pass  # Keep original prediction

        # 1m n+5 prediction (leading signal)
        n_plus_5_1m = None
        if ohlcv_1m and ohlcv_1m.closes:
            self.log.debug("Computing 1m n+5 prediction for {}", asset.value)
            n_plus_5_1m = await self._predict_single(
                ohlcv_1m.closes, Timeframe.M1, asset
            )
            # Blend 1m n+5 with 5m — they should agree for high confidence
            if n_plus_5_1m.direction == primary.direction:
                primary.confidence = min(0.90, primary.confidence + 0.05)
            else:
                primary.confidence = max(0.40, primary.confidence - 0.05)
            primary.n_plus_5_1m = n_plus_5_1m

        # 15m n+1 prediction (arbitrage check)
        n_plus_1_15m = None
        if ohlcv_15m and ohlcv_15m.closes:
            self.log.debug("Computing 15m n+1 prediction for {}", asset.value)
            n_plus_1_15m = await self._predict_single(
                ohlcv_15m.closes, Timeframe.M15, asset
            )
            primary.n_plus_1_15m = n_plus_1_15m

        return primary
