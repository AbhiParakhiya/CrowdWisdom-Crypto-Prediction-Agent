"""
ui/app.py
──────────
CrowdWisdom Trading — Streamlit Dashboard

Features:
  - Live prediction pipeline with real-time status updates
  - OHLCV candlestick charts with technical indicators
  - Prediction market implied probabilities (Polymarket + Kalshi)
  - Kronos prediction confidence gauge
  - Kelly position sizing display
  - Multi-timeframe arbitrage signal
  - Agent signal breakdown table
  - Auto-refresh every 5 minutes

Run: streamlit run ui/app.py
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.orchestrator import OrchestratorAgent
from core.config import cfg
from core.models import Asset, Direction, FinalSignal, OHLCVBar

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrowdWisdom Crypto Predictions",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d4aa;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .signal-card {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .signal-up {
        background: linear-gradient(135deg, #0d3320 0%, #0a2518 100%);
        border: 1px solid #00ff88;
    }
    .signal-down {
        background: linear-gradient(135deg, #330d0d 0%, #250a0a 100%);
        border: 1px solid #ff4444;
    }
    .signal-neutral {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #444;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
    }
    .up-color { color: #00ff88; }
    .down-color { color: #ff4444; }
    .neutral-color { color: #aaa; }
    .agent-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .stProgress > div > div > div { background: #00d4aa !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state helpers ──────────────────────────────────────────────────────

def _init_state():
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "last_run" not in st.session_state:
        st.session_state.last_run = None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "run_logs" not in st.session_state:
        st.session_state.run_logs = []


# ── Async runner ──────────────────────────────────────────────────────────────

def run_async(coro):
    """Run async coroutine from sync Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=300)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


async def run_pipeline(selected_assets: list[str]) -> dict:
    """Run the orchestrator for selected assets."""
    orchestrator = OrchestratorAgent()
    results = {}
    for asset_str in selected_assets:
        try:
            signal = await orchestrator.run_task(Asset(asset_str))
            results[asset_str] = signal
        except Exception as exc:
            results[asset_str] = {"error": str(exc)}
    return results


# ── Chart builders ────────────────────────────────────────────────────────────

def build_candlestick(bars: list[OHLCVBar], asset: str, signal: FinalSignal) -> go.Figure:
    """Build OHLCV candlestick with EMA overlays and volume."""
    if not bars:
        fig = go.Figure()
        fig.add_annotation(text="No OHLCV data available", showarrow=False)
        return fig

    df = pd.DataFrame([
        {
            "timestamp": b.timestamp,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in bars[-200:]  # Show last 200 bars
    ])

    # EMA calculations
    df["ema8"] = df["close"].ewm(span=8, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="OHLCV",
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4444",
        ),
        row=1, col=1,
    )

    # EMA overlays
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["ema8"],
            name="EMA8", line=dict(color="#ffa500", width=1.5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["ema21"],
            name="EMA21", line=dict(color="#00aaff", width=1.5),
        ),
        row=1, col=1,
    )

    # Add prediction annotation
    if signal and signal.direction != Direction.NEUTRAL:
        color = "#00ff88" if signal.direction == Direction.UP else "#ff4444"
        arrow = "▲" if signal.direction == Direction.UP else "▼"
        fig.add_annotation(
            x=df["timestamp"].iloc[-1],
            y=df["close"].iloc[-1],
            text=f"  {arrow} {signal.direction.value} ({signal.consensus_confidence:.0%})",
            showarrow=True,
            arrowhead=2,
            arrowcolor=color,
            font=dict(color=color, size=14, family="monospace"),
            row=1, col=1,
        )

    # Volume
    colors = ["#00ff88" if c >= o else "#ff4444"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(
            x=df["timestamp"], y=df["volume"],
            name="Volume", marker_color=colors, opacity=0.6,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"{asset} / USDT — 5m Bars",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#ddd"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(t=50, b=30, l=50, r=20),
    )
    fig.update_xaxes(gridcolor="#222", showgrid=True)
    fig.update_yaxes(gridcolor="#222", showgrid=True)

    return fig


def build_probability_gauge(prob: float, label: str, color: str) -> go.Figure:
    """Build a gauge chart for probability display."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        title={"text": label, "font": {"size": 13, "color": "#aaa"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1a1a2e",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 45], "color": "#1a0a0a"},
                {"range": [45, 55], "color": "#1a1a1a"},
                {"range": [55, 100], "color": "#0a1a0a"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(
        height=200,
        paper_bgcolor="#0e1117",
        font=dict(color="#ddd"),
        margin=dict(t=30, b=10, l=10, r=10),
    )
    return fig


def build_signal_radar(signals: list) -> go.Figure:
    """Radar chart showing agent signal strengths."""
    if not signals:
        return go.Figure()

    names = [s.agent_name.replace("Agent", "").replace("Predictor", "") for s in signals]
    confidences = [s.confidence * 100 for s in signals]
    colors_map = {"UP": "#00ff88", "DOWN": "#ff4444", "NEUTRAL": "#888888"}
    colors = [colors_map.get(s.direction.value, "#888") for s in signals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=confidences,
        marker_color=colors,
        text=[f"{c:.0f}%" for c in confidences],
        textposition="outside",
        name="Agent Confidence",
    ))
    fig.update_layout(
        title="Agent Signal Breakdown",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#ddd"),
        yaxis=dict(range=[0, 110], gridcolor="#222"),
        xaxis=dict(gridcolor="#222"),
        height=280,
        margin=dict(t=40, b=20, l=30, r=20),
    )
    return fig


# ── Signal display helpers ─────────────────────────────────────────────────────

def direction_emoji(direction: Direction) -> str:
    return {"UP": "🟢 UP", "DOWN": "🔴 DOWN", "NEUTRAL": "⚪ NEUTRAL"}.get(
        direction.value, "⚪"
    )


def confidence_bar(confidence: float) -> str:
    filled = int(confidence * 20)
    return "█" * filled + "░" * (20 - filled)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[list[str], bool, int]:
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # Validation
        errors = cfg.validate()
        if errors:
            st.error("**Config errors:**\n" + "\n".join(f"• {e}" for e in errors))
        else:
            st.success("✅ Config valid")

        st.divider()

        # Asset selection
        st.markdown("### Assets")
        selected = []
        if st.checkbox("₿ Bitcoin (BTC)", value=True):
            selected.append("BTC")
        if st.checkbox("Ξ Ethereum (ETH)", value=True):
            selected.append("ETH")

        st.divider()

        # Settings display
        st.markdown("### Settings")
        st.markdown(f"**Model:** `{cfg.openrouter_model.split('/')[-1]}`")
        st.markdown(f"**Kelly fraction:** {cfg.kelly_fraction:.0%}")
        st.markdown(f"**Min confidence:** {cfg.min_confidence:.0%}")
        st.markdown(f"**Portfolio:** ${cfg.portfolio_usd:,.0f}")
        st.markdown(f"**Kronos:** `{cfg.kronos_model}`")

        st.divider()

        # Auto-refresh
        st.markdown("### Auto-refresh")
        auto_refresh = st.checkbox("Enable (5 min)", value=False)
        refresh_interval = st.slider("Interval (seconds)", 60, 600, 300, 30)

        st.divider()

        # Info
        st.markdown("""
        ### Data Sources
        - 📊 OHLCV: Apify → ccxt fallback
        - 🎯 Markets: Polymarket + Kalshi
        - 🤖 Model: Kronos foundation model
        - 📐 Risk: Kelly Criterion (fractional)

        ### Agent Loop
        1. Market Search Agent
        2. OHLCV Fetcher Agent
        3. Kronos Predictor Agent
        4. Kelly Risk Agent
        5. Orchestrator (feedback loop)
        """)

        return selected, auto_refresh, refresh_interval


# ── Main dashboard renderer ───────────────────────────────────────────────────

def render_signal_card(asset: str, signal: FinalSignal):
    """Render the main signal card for one asset."""
    dir_val = signal.direction.value
    card_class = f"signal-{dir_val.lower()}"
    color_class = f"{dir_val.lower()}-color"
    dir_arrow = "▲" if dir_val == "UP" else ("▼" if dir_val == "DOWN" else "▬")
    dir_color = "#00ff88" if dir_val == "UP" else ("#ff4444" if dir_val == "DOWN" else "#aaa")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

    with col1:
        st.metric(
            label=f"{'₿' if asset == 'BTC' else 'Ξ'} {asset} Signal",
            value=f"{dir_arrow} {dir_val}",
            delta=f"{signal.consensus_confidence:.1%} confidence",
            delta_color="normal" if dir_val != "NEUTRAL" else "off",
        )

    with col2:
        kelly = signal.kelly
        if kelly:
            st.metric(
                label="Kelly Position",
                value=f"${kelly.recommended_usd:,.0f}",
                delta=f"{kelly.recommended_fraction:.1%} of portfolio",
            )
        else:
            st.metric(label="Kelly Position", value="N/A")

    with col3:
        if signal.market_search:
            ms = signal.market_search
            st.metric(
                label="Market Implied P(UP)",
                value=f"{ms.implied_prob_up:.1%}",
                delta=f"{len(ms.markets)} markets found",
            )
        else:
            st.metric(label="Market Implied P(UP)", value="N/A")

    with col4:
        if signal.kronos_prediction:
            kp = signal.kronos_prediction
            st.metric(
                label="Kronos Prediction",
                value=f"{kp.direction.value}",
                delta=f"{kp.confidence:.1%} conf ({kp.method})",
            )
        else:
            st.metric(label="Kronos Prediction", value="N/A")


def render_asset_section(asset: str, signal: FinalSignal):
    """Render full section for one asset."""
    dir_val = signal.direction.value
    dir_color = "#00ff88" if dir_val == "UP" else ("#ff4444" if dir_val == "DOWN" else "#aaa")

    st.markdown(f"""
    <div style='border-left: 4px solid {dir_color}; padding-left: 1rem; margin-bottom: 0.5rem;'>
        <h2 style='color: {dir_color}; margin: 0;'>
            {'₿' if asset == 'BTC' else 'Ξ'} {asset} — 
            {'▲' if dir_val == 'UP' else '▼' if dir_val == 'DOWN' else '▬'} {dir_val}
        </h2>
        <p style='color: #888; margin: 0; font-size: 0.85rem;'>
            Confidence: {signal.consensus_confidence:.1%} | 
            Action: {signal.action} | 
            Feedback rounds: {signal.feedback_rounds}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    render_signal_card(asset, signal)

    # LLM synthesis
    if signal.llm_synthesis:
        with st.expander("💬 AI Analysis", expanded=True):
            st.markdown(f"_{signal.llm_synthesis}_")

    # Charts row
    col_left, col_right = st.columns([3, 1])

    with col_left:
        # OHLCV chart
        if signal.kronos_prediction and hasattr(signal, '_ohlcv_bars'):
            fig = build_candlestick(signal._ohlcv_bars, asset, signal)
            st.plotly_chart(fig, use_container_width=True)
        elif signal.market_search:
            # Show probability gauges when no chart data
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                ms = signal.market_search
                fig = build_probability_gauge(
                    ms.implied_prob_up, "P(UP) — Market Implied", "#00ff88"
                )
                st.plotly_chart(fig, use_container_width=True)
            with g_col2:
                fig = build_probability_gauge(
                    signal.consensus_confidence,
                    "Consensus Confidence", dir_color
                )
                st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Agent breakdown
        if signal.agent_signals:
            fig = build_signal_radar(signal.agent_signals)
            st.plotly_chart(fig, use_container_width=True)

    # Kelly detail expander
    if signal.kelly:
        with st.expander("📐 Kelly Risk Details"):
            k = signal.kelly
            c1, c2, c3 = st.columns(3)
            c1.metric("P(Win)", f"{k.win_probability:.1%}")
            c2.metric("Payout Ratio (b)", f"{k.win_payout_ratio:.2f}×")
            c3.metric("Full Kelly f*", f"{k.kelly_fraction:.1%}")

            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Fractional Kelly ({cfg.kelly_fraction:.0%}) | {k.recommended_fraction:.2%} |
            | Recommended size | ${k.recommended_usd:,.2f} |
            | Portfolio base | ${cfg.portfolio_usd:,.0f} |
            """)

            if k.arbitrage_signal:
                st.info(f"🔀 **Arbitrage Signal:** {k.arbitrage_signal}")

    # Multi-timeframe breakdown
    if signal.kronos_prediction:
        kp = signal.kronos_prediction
        with st.expander("⏱ Multi-Timeframe Signals"):
            tf_data = {"5m (Primary)": (kp.direction.value, kp.confidence, kp.method)}
            if kp.n_plus_5_1m:
                tf_data["1m × n+5 (Leading)"] = (
                    kp.n_plus_5_1m.direction.value,
                    kp.n_plus_5_1m.confidence,
                    kp.n_plus_5_1m.method,
                )
            if kp.n_plus_1_15m:
                tf_data["15m × n+1 (Arb Check)"] = (
                    kp.n_plus_1_15m.direction.value,
                    kp.n_plus_1_15m.confidence,
                    kp.n_plus_1_15m.method,
                )

            for tf, (dir_, conf, method) in tf_data.items():
                col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 2])
                col_a.markdown(f"**{tf}**")
                arrow = "▲" if dir_ == "UP" else "▼"
                color = "green" if dir_ == "UP" else "red"
                col_b.markdown(f":{color}[{arrow} {dir_}]")
                col_c.markdown(f"`{conf:.1%}`")
                col_d.markdown(f"_{method}_")

    # Prediction markets table
    if signal.market_search and signal.market_search.markets:
        with st.expander(f"🎯 Prediction Markets ({len(signal.market_search.markets)})"):
            markets_data = []
            for m in signal.market_search.markets:
                markets_data.append({
                    "Platform": m.platform.title(),
                    "Direction": m.direction.value,
                    "P(YES)": f"{m.yes_price:.1%}",
                    "Volume 24h": f"${m.volume_24h:,.0f}" if m.volume_24h else "N/A",
                    "Question": m.question[:60] + "..." if len(m.question) > 60 else m.question,
                })
            st.dataframe(pd.DataFrame(markets_data), use_container_width=True)

    st.divider()


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    _init_state()

    # Header
    st.markdown(
        '<p class="main-header">📈 CrowdWisdom Crypto Prediction Agent</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Multi-agent crypto signal system | '
        'Polymarket + Kalshi + Kronos + Kelly | Hermes feedback loop</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    selected_assets, auto_refresh, refresh_interval = render_sidebar()

    # Auto-refresh
    if auto_refresh:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")

    # Run button + status
    col_btn, col_status = st.columns([1, 4])
    with col_btn:
        run_btn = st.button(
            "🚀 Run Prediction Pipeline",
            type="primary",
            disabled=st.session_state.running or not selected_assets,
            use_container_width=True,
        )
    with col_status:
        if st.session_state.last_run:
            st.markdown(
                f"Last run: `{st.session_state.last_run.strftime('%H:%M:%S')}` | "
                f"Assets: `{', '.join(st.session_state.results.keys())}`"
            )
        if not cfg.openrouter_api_key:
            st.warning("⚠️ OPENROUTER_API_KEY not set in .env")
        if not cfg.apify_api_token:
            st.warning("⚠️ APIFY_API_TOKEN not set in .env")

    # Run pipeline
    if run_btn and selected_assets:
        st.session_state.running = True
        st.session_state.run_logs = []

        progress_bar = st.progress(0, text="Initializing agents...")
        status_placeholder = st.empty()

        try:
            steps = [
                (10, "🔍 Searching Polymarket + Kalshi markets..."),
                (30, "📊 Fetching OHLCV data via Apify..."),
                (60, "🤖 Running Kronos predictions..."),
                (80, "📐 Computing Kelly position sizes..."),
                (95, "🔄 Running Hermes feedback loop..."),
            ]

            for pct, msg in steps:
                progress_bar.progress(pct, text=msg)
                status_placeholder.info(msg)
                time.sleep(0.3)

            results = run_async(run_pipeline(selected_assets))

            progress_bar.progress(100, text="✅ Pipeline complete!")
            status_placeholder.success("✅ Prediction pipeline completed successfully")
            time.sleep(0.5)

            st.session_state.results = results
            st.session_state.last_run = datetime.now()

        except Exception as exc:
            progress_bar.progress(100, text="❌ Pipeline failed")
            status_placeholder.error(f"❌ Pipeline error: {exc}")
            st.exception(exc)
        finally:
            st.session_state.running = False
            progress_bar.empty()

    # Render results
    if st.session_state.results:
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")

        for asset, result in st.session_state.results.items():
            if isinstance(result, dict) and "error" in result:
                st.error(f"❌ {asset}: {result['error']}")
                continue
            if not isinstance(result, FinalSignal):
                st.warning(f"⚠️ {asset}: Unexpected result type")
                continue
            render_asset_section(asset, result)

    else:
        # Empty state
        st.markdown("---")
        with st.container():
            st.markdown("""
            ### 👆 Click "Run Prediction Pipeline" to start

            The system will:
            1. **Search Polymarket + Kalshi** for BTC/ETH prediction markets
            2. **Fetch 1000 OHLCV bars** via Apify (CryptoCompare actor)
            3. **Run Kronos predictions** across 1m, 5m, 15m timeframes
            4. **Apply Kelly Criterion** for optimal position sizing
            5. **Orchestrate** everything with a Hermes feedback loop
            6. **Display** unified signals with multi-timeframe arbitrage detection

            ---
            #### Multi-Timeframe Scaling Strategy
            ```
            1m × n+5  ──→  Leading signal for 5m n+1
            5m × n+1  ──→  Primary trading signal
            15m × n+1 ──→  Arbitrage check vs 3× compound 5m
            ```

            #### Kelly Position Sizing
            ```
            f* = (b×p - q) / b      where p = P(win), b = payout ratio
            Recommended = f* × {kelly_fraction:.0%}  (fractional Kelly)
            ```
            """.format(kelly_fraction=cfg.kelly_fraction))

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="color: #444; font-size: 0.8rem; text-align: center;">'
        "CrowdWisdom Trading Prediction System | "
        "For research and educational purposes only. Not financial advice. "
        f"| Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
