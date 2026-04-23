"""Modeled series construction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_modeled_series(df: pd.DataFrame, mode: str, rolling_window: int = 50) -> pd.Series:
    """Construct a model target series from normalized OHLCV data."""

    close = df["close"].astype(float)

    if mode == "raw_close":
        return close
    if mode == "log_close":
        return np.log(close.clip(lower=1e-12))
    if mode == "rolling_demeaned_log_close":
        log_close = np.log(close.clip(lower=1e-12))
        return log_close - log_close.rolling(rolling_window).mean()
    if mode == "rolling_zscore_close":
        mean = close.rolling(rolling_window).mean()
        std = close.rolling(rolling_window).std(ddof=0).replace(0.0, np.nan)
        return (close - mean) / std
    if mode == "realized_vol_proxy":
        rets = np.log(close.clip(lower=1e-12)).diff()
        return rets.pow(2).rolling(rolling_window).sum().pow(0.5)

    raise ValueError(f"Unknown modeled series mode: {mode}")
