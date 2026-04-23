"""No-lookahead backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    bars: pd.DataFrame,
    signal_df: pd.DataFrame,
    transaction_cost_bps: float = 2.0,
    slippage_bps: float = 1.0,
    max_abs_position: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a simple close-to-close no-lookahead backtest."""

    px = bars[["timestamp", "close"]].copy()
    px["timestamp"] = pd.to_datetime(px["timestamp"], utc=True)
    px = px.set_index("timestamp").sort_index()

    sig = signal_df.copy()
    sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True)
    sig = sig.set_index("timestamp").sort_index()

    joined = px.join(sig[["signal"]], how="left").fillna(0.0)
    joined["ret"] = joined["close"].pct_change().fillna(0.0)
    joined["position"] = joined["signal"].clip(-max_abs_position, max_abs_position)
    joined["position_lag"] = joined["position"].shift(1).fillna(0.0)

    turnover = (joined["position"] - joined["position_lag"]).abs()
    cost = turnover * (transaction_cost_bps + slippage_bps) / 1e4

    joined["strategy_ret"] = joined["position_lag"] * joined["ret"] - cost
    joined["equity"] = (1 + joined["strategy_ret"]).cumprod()

    trades = joined.loc[turnover > 0, ["close", "position", "strategy_ret"]].copy()
    trades["turnover"] = turnover[turnover > 0]
    trades = trades.reset_index().rename(columns={"index": "timestamp"})

    curve = joined.reset_index()[["timestamp", "close", "position", "ret", "strategy_ret", "equity"]]
    return trades, curve
