"""Data fetching orchestration and local caching."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stochastic_trade.data.providers import (
    DataProvider,
    TwelveDataProvider,
    YFinanceProvider,
)


def get_provider(name: str, twelvedata_api_key: str | None = None) -> DataProvider:
    """Create provider instance by name."""

    if name == "yfinance":
        return YFinanceProvider()
    if name == "twelvedata":
        if not twelvedata_api_key:
            raise ValueError("twelvedata_api_key is required for TwelveData provider")
        return TwelveDataProvider(api_key=twelvedata_api_key)
    raise ValueError(f"Unknown provider: {name}")


def fetch_and_cache(
    provider_name: str,
    symbol: str,
    interval: str,
    start: str | None,
    end: str | None,
    raw_dir: Path,
    processed_dir: Path,
    twelvedata_api_key: str | None = None,
) -> Path:
    """Fetch normalized data and cache into processed CSV."""

    provider = get_provider(provider_name, twelvedata_api_key=twelvedata_api_key)
    df = provider.fetch_ohlcv(symbol=symbol, interval=interval, start=start, end=end)
    if df.empty:
        raise ValueError("No bars returned from provider")

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Keep a normalized copy in processed data for downstream model fitting.
    filename = f"{symbol}_{interval}_{(start or 'min')}_{(end or 'max')}.csv".replace(":", "-")
    path = processed_dir / filename
    df.to_csv(path, index=False)
    return path


def update_latest_bars(
    provider_name: str,
    symbol: str,
    interval: str,
    processed_dir: Path,
    lookback_days: int = 10,
    twelvedata_api_key: str | None = None,
) -> Path:
    """Update latest bars by fetching a short recent lookback slice."""

    end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    return fetch_and_cache(
        provider_name=provider_name,
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        raw_dir=processed_dir.parent / "raw",
        processed_dir=processed_dir,
        twelvedata_api_key=twelvedata_api_key,
    )
