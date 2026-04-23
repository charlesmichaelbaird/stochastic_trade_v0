"""Configuration models for data, fitting, and backtesting."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Storage locations for raw and processed artifacts."""

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")


class DataFetchConfig(BaseModel):
    """Data fetch settings."""

    provider: Literal["yfinance", "twelvedata"] = "yfinance"
    symbol: str
    interval: str = "1d"
    start: str | None = None
    end: str | None = None
    output_name: str | None = None


class RollingFitConfig(BaseModel):
    """Rolling estimation settings."""

    window: int = Field(200, gt=10)
    step: int = Field(10, gt=0)
    dt: float = Field(1.0, gt=0)


class BacktestConfig(BaseModel):
    """Simple no-lookahead backtest settings."""

    transaction_cost_bps: float = 2.0
    slippage_bps: float = 1.0
    max_abs_position: float = 1.0
