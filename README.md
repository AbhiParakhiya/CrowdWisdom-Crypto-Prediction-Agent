# CrowdWisdom Trading — Crypto Prediction Agent System

A multi-agent crypto prediction system using Hermes-style agents, Kronos foundation model,
Polymarket/Kalshi prediction market search, Apify data scraping, and a Streamlit UI.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Streamlit UI (ui/app.py)                 │
└─────────────────────────────┬────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────┐
│              Orchestrator Agent (agents/orchestrator.py)     │
│   Hermes feedback loop: runs agents, collects results,       │
│   re-prompts based on confidence, scales multi-timeframe     │
└──┬──────────────┬───────────────┬──────────────┬─────────────┘
   │              │               │              │
┌──▼──────┐ ┌────▼──────┐ ┌─────▼────┐ ┌───────▼──────┐
│ Market  │ │   OHLCV   │ │ Kronos   │ │    Kelly     │
│ Search  │ │  Fetcher  │ │Predictor │ │ Risk Manager │
│  Agent  │ │   Agent   │ │  Agent   │ │    Agent     │
└─────────┘ └───────────┘ └──────────┘ └──────────────┘
  Polymarket   OHLCV bars   Up/Down      Kelly f*
  + Kalshi     last 1000    5min pred    position size
  crypto mkts  via Apify    via Kronos   + arbitrage
```

## Agents

1. **Market Search Agent** — Queries Polymarket CLOB + Kalshi APIs for BTC/ETH markets,
   extracts 5-minute directional implied probability
2. **OHLCV Fetcher Agent** — Uses Apify to scrape last 1000 OHLCV bars (CryptoCompare actor)
3. **Kronos Predictor Agent** — Feeds OHLCV into Kronos foundation model to predict next
   5-min up/down; falls back to lightweight LSTM/trend model if Kronos unavailable
4. **Kelly Risk Manager Agent** — Applies Kelly Criterion for optimal position sizing;
   runs internal arbitrage between 15min signal vs 3x 5min signals
5. **Orchestrator Agent** — Hermes feedback loop, aggregates all signals, scales across
   timeframes (1min→5min n+5, 5min→15min n+1)

## Scaling / Arbitrage Features
- Multi-timeframe internal arbitrage: 15min prediction vs compound of 3x 5min predictions
- 1min n+5 prediction used as 5min n+1 leading signal
- Confidence-weighted signal aggregation with LLM synthesis via OpenRouter
- Feedback loop: re-runs if consensus confidence < threshold (default 60%)
- Per-asset Kelly sizing with half-Kelly safety cap

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
streamlit run ui/app.py
```

## Environment Variables

See `.env.example` for all keys required.

## Data Sources
- OHLCV: Apify CryptoCompare scraper (`lulzasaur/cryptocompare-scraper`) + ccxt fallback
- Predictions: Kronos Foundation Model (HuggingFace: `NeoQuasar/Kronos-base`)
- Market odds: Polymarket Gamma API + Kalshi REST API (read-only, no auth needed)
- Risk sizing: Kelly Criterion with 0.25× fractional Kelly
- LLM synthesis: OpenRouter (free model: `mistralai/mistral-7b-instruct:free`)
