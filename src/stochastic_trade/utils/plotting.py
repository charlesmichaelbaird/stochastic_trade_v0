"""Plot helpers for research diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_signal_overlay(bars: pd.DataFrame, signals: pd.DataFrame, out_path: Path) -> Path:
    """Plot close price with signal markers."""

    price = bars.copy()
    price["timestamp"] = pd.to_datetime(price["timestamp"], utc=True)
    price = price.set_index("timestamp")

    sig = signals.copy()
    sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True)
    sig = sig.set_index("timestamp")

    merged = price.join(sig[["signal"]], how="left").fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(merged.index, merged["close"], label="Close", linewidth=1.2)
    ax.scatter(merged.index[merged["signal"] > 0], merged.loc[merged["signal"] > 0, "close"], marker="^", label="Long", s=20)
    ax.scatter(merged.index[merged["signal"] < 0], merged.loc[merged["signal"] < 0, "close"], marker="v", label="Short", s=20)
    ax.legend(loc="best")
    ax.set_title("Signal Overlay")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
