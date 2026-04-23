"""Data providers and normalization helpers."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "symbol", "interval"]


class DataProvider(ABC):
    """Base data provider interface."""

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, interval: str, start: str | None, end: str | None) -> pd.DataFrame:
        """Fetch and normalize OHLCV bars."""


class YFinanceProvider(DataProvider):
    """yfinance historical data provider."""

    def fetch_ohlcv(self, symbol: str, interval: str, start: str | None, end: str | None) -> pd.DataFrame:
        raw = yf.download(symbol, interval=interval, start=start, end=end, progress=False, auto_adjust=False)
        if raw.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df = raw.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )[["open", "high", "low", "close", "volume"]].copy()
        df = df.reset_index(names="timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["symbol"] = symbol
        df["interval"] = interval
        return df[OHLCV_COLUMNS]


class TwelveDataProvider(DataProvider):
    """TwelveData time-series provider (requires API key)."""

    BASE_URL = "https://api.twelvedata.com/time_series"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def fetch_ohlcv(self, symbol: str, interval: str, start: str | None, end: str | None) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": 5000,
            "order": "ASC",
            "format": "JSON",
        }
        if start:
            params["start_date"] = start
        if end:
            params["end_date"] = end

        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        values = payload.get("values", [])
        if not values:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df = pd.DataFrame(values)
        df["timestamp"] = pd.to_datetime(df["datetime"], utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["symbol"] = symbol
        df["interval"] = interval
        return df[OHLCV_COLUMNS]


def cache_key(provider: str, symbol: str, interval: str, start: str | None, end: str | None) -> str:
    """Stable cache key for raw API payloads."""

    raw = json.dumps(
        {"provider": provider, "symbol": symbol, "interval": interval, "start": start, "end": end},
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def write_raw_cache(raw_dir: Path, key: str, payload: dict) -> Path:
    """Write raw response payload to local cache."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{key}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
