"""
main.py
────────
CLI entry point for the CrowdWisdom Trading prediction system.
Runs the full pipeline for all configured assets and prints results.

Usage:
    python main.py                    # run for all assets (BTC + ETH)
    python main.py --asset BTC        # run for BTC only
    python main.py --asset ETH --json # output JSON
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import cfg
from core.logger import get_logger
from agents.orchestrator import OrchestratorAgent
from core.models import Asset, Direction

log = get_logger("main")


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║       CrowdWisdom Trading — Crypto Prediction System        ║
║  Polymarket + Kalshi + Kronos + Kelly | Hermes Agent Loop   ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_signal(asset: str, signal):
    """Pretty-print a FinalSignal to console."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    console = Console()

    dir_val = signal.direction.value
    color = "green" if dir_val == "UP" else ("red" if dir_val == "DOWN" else "white")
    arrow = "▲" if dir_val == "UP" else ("▼" if dir_val == "DOWN" else "▬")

    # Main panel
    console.print(Panel(
        f"[bold {color}]{arrow}  {asset} — {dir_val}[/]\n"
        f"[white]Confidence: [bold]{signal.consensus_confidence:.1%}[/bold] | "
        f"Action: [bold]{signal.action}[/bold][/]\n"
        f"[dim]Feedback rounds: {signal.feedback_rounds}[/]",
        title=f"[bold]SIGNAL — {asset}[/]",
        border_style=color,
        padding=(1, 2),
    ))

    # Kelly table
    if signal.kelly:
        k = signal.kelly
        table = Table(title="Kelly Position Sizing", box=box.SIMPLE_HEAVY)
        table.add_column("Parameter", style="dim")
        table.add_column("Value", style="bold")
        table.add_row("Direction", f"[{color}]{dir_val}[/]")
        table.add_row("P(Win)", f"{k.win_probability:.1%}")
        table.add_row("Payout ratio (b)", f"{k.win_payout_ratio:.2f}×")
        table.add_row("Full Kelly f*", f"{k.kelly_fraction:.1%}")
        table.add_row(
            f"Recommended ({cfg.kelly_fraction:.0%} Kelly)",
            f"[bold]{k.recommended_fraction:.2%}[/]"
        )
        table.add_row(
            "Position size",
            f"[bold green]${k.recommended_usd:,.2f}[/]"
        )
        if k.arbitrage_signal:
            table.add_row("Arbitrage", f"[yellow]{k.arbitrage_signal[:60]}...[/]"
                          if len(k.arbitrage_signal) > 60 else k.arbitrage_signal)
        console.print(table)

    # Agent signals
    if signal.agent_signals:
        sig_table = Table(title="Agent Signals", box=box.SIMPLE)
        sig_table.add_column("Agent", style="cyan")
        sig_table.add_column("Direction")
        sig_table.add_column("Confidence")
        sig_table.add_column("Weight", style="dim")
        for s in signal.agent_signals:
            sc = "green" if s.direction.value == "UP" else "red"
            sig_table.add_row(
                s.agent_name,
                f"[{sc}]{s.direction.value}[/]",
                f"{s.confidence:.1%}",
                f"{s.weight:.1f}",
            )
        console.print(sig_table)

    # Multi-timeframe
    kp = signal.kronos_prediction
    if kp:
        console.print(f"\n[bold]Multi-Timeframe Predictions:[/]")
        c = "green" if kp.direction.value == "UP" else "red"
        console.print(f"  5m n+1:  [{c}]{kp.direction.value}[/] ({kp.confidence:.1%}) via {kp.method}")
        if kp.n_plus_5_1m:
            c2 = "green" if kp.n_plus_5_1m.direction.value == "UP" else "red"
            console.print(
                f"  1m n+5:  [{c2}]{kp.n_plus_5_1m.direction.value}[/] "
                f"({kp.n_plus_5_1m.confidence:.1%}) — leading signal"
            )
        if kp.n_plus_1_15m:
            c3 = "green" if kp.n_plus_1_15m.direction.value == "UP" else "red"
            console.print(
                f"  15m n+1: [{c3}]{kp.n_plus_1_15m.direction.value}[/] "
                f"({kp.n_plus_1_15m.confidence:.1%}) — arbitrage check"
            )

    # Prediction markets
    if signal.market_search and signal.market_search.markets:
        console.print(
            f"\n[dim]Prediction markets: {len(signal.market_search.markets)} found | "
            f"Implied P(UP) = {signal.market_search.implied_prob_up:.1%}[/]"
        )

    # LLM narrative
    if signal.llm_synthesis:
        console.print(Panel(
            f"[italic]{signal.llm_synthesis}[/]",
            title="AI Analysis",
            border_style="dim",
            padding=(0, 1),
        ))

    console.print()


async def run(assets: list[str], json_output: bool = False):
    """Main async runner."""
    log.info("Starting prediction pipeline for assets: {}", assets)

    # Validate config
    errors = cfg.validate()
    if errors:
        log.error("Configuration errors:")
        for e in errors:
            log.error("  - {}", e)
        print("\n⚠️  Missing configuration. Please check your .env file.\n")
        print("Required:\n  OPENROUTER_API_KEY=...\n  APIFY_API_TOKEN=...\n")
        sys.exit(1)

    orchestrator = OrchestratorAgent()
    results = {}

    for asset_str in assets:
        try:
            log.info("Running pipeline for {}", asset_str)
            signal = await orchestrator.run_task(Asset(asset_str))
            results[asset_str] = signal
        except Exception as exc:
            log.error("Pipeline failed for {}: {}", asset_str, exc)
            results[asset_str] = {"error": str(exc)}

    if json_output:
        output = {}
        for asset, signal in results.items():
            if hasattr(signal, "model_dump"):
                output[asset] = signal.model_dump(mode="json", exclude_none=True)
            else:
                output[asset] = signal
        print(json.dumps(output, indent=2, default=str))
    else:
        print_banner()
        for asset, signal in results.items():
            if isinstance(signal, dict) and "error" in signal:
                print(f"❌ {asset}: {signal['error']}\n")
            else:
                print_signal(asset, signal)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CrowdWisdom Crypto Prediction Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     # All assets (BTC + ETH)
  python main.py --asset BTC         # BTC only
  python main.py --asset ETH --json  # ETH, JSON output
  streamlit run ui/app.py            # Launch Streamlit dashboard
        """,
    )
    parser.add_argument(
        "--asset",
        choices=["BTC", "ETH"],
        help="Asset to analyze (default: all configured assets)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    assets = [args.asset] if args.asset else cfg.assets
    asyncio.run(run(assets, json_output=args.json))


if __name__ == "__main__":
    main()
