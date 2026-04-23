"""Experimental signal generation from fitted dynamics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _apply_hysteresis_and_cooldown(raw: pd.Series, threshold: float, cooldown: int) -> pd.Series:
    signal = pd.Series(0.0, index=raw.index)
    cool = 0
    pos = 0.0
    for i, val in enumerate(raw):
        if cool > 0:
            signal.iloc[i] = pos
            cool -= 1
            continue
        if pos == 0 and abs(val) >= threshold:
            pos = float(np.sign(val))
            cool = cooldown
        elif pos != 0 and abs(val) < threshold * 0.5:
            pos = 0.0
        signal.iloc[i] = pos
    return signal


def generate_ou_signals(
    modeled_series: pd.Series,
    ou_params: pd.DataFrame,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """Generate OU-like signals from mean deviation and fit quality."""

    df = ou_params.copy().set_index("timestamp")
    aligned = modeled_series.reindex(df.index).dropna()
    df = df.reindex(aligned.index)

    z = (aligned - df["mu_euler"]) / df["sigma_euler"].replace(0.0, np.nan)
    raw = -z
    quality_mask = (df["theta_euler"] > 0) & (df["r2_euler"] > 0.02)
    raw = raw.where(quality_mask, 0.0).fillna(0.0)

    signal = _apply_hysteresis_and_cooldown(raw, threshold=threshold, cooldown=3)
    return pd.DataFrame({"timestamp": signal.index, "raw_score": raw.values, "signal": signal.values})


def generate_generic_signals(
    generic_params: pd.DataFrame,
    drift_strength_threshold: float = 0.05,
    high_diffusion_quantile: float = 0.8,
) -> pd.DataFrame:
    """Generate generic SDE signals from local drift and diffusion levels."""

    df = generic_params.copy().set_index("timestamp")
    diffusion_cut = df["local_diffusion"].quantile(high_diffusion_quantile)
    raw = df["local_drift"] / df["local_diffusion"].replace(0.0, np.nan)
    raw = raw.where(df["local_diffusion"] <= diffusion_cut, raw * 0.25).fillna(0.0)

    signal = _apply_hysteresis_and_cooldown(raw, threshold=drift_strength_threshold, cooldown=5)
    return pd.DataFrame({"timestamp": signal.index, "raw_score": raw.values, "signal": signal.values})
