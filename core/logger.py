"""
core/logger.py
──────────────
Loguru-based structured logger.
Each agent gets a named child logger so logs are easy to filter.

Usage:
    from core.logger import get_logger
    log = get_logger("MarketSearchAgent")
    log.info("Fetched {} markets", n)
"""

import sys
from loguru import logger

# Remove default handler and configure a clean one
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[agent]: <24}</cyan> | "
        "{message}"
    ),
    level="DEBUG",
    colorize=True,
)

# Also write to a rotating file for session replay
logger.add(
    "logs/session.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[agent]: <24} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    enqueue=True,  # thread-safe
)


def get_logger(agent_name: str):
    """Return a logger bound to the given agent name."""
    return logger.bind(agent=agent_name)
