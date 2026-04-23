"""Rolling window model fitting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stochastic_trade.models.generic_sde import fit_generic_sde
from stochastic_trade.models.ou import fit_ou_euler, fit_ou_exact


def rolling_fit_ou(series: pd.Series, window: int, step: int, dt: float = 1.0) -> pd.DataFrame:
    """Fit OU models over rolling windows."""

    rows: list[dict] = []
    s = series.dropna()
    for end in range(window, len(s) + 1, step):
        chunk = s.iloc[end - window : end]
        euler = fit_ou_euler(chunk, dt=dt)
        exact = fit_ou_exact(chunk, dt=dt)
        rows.append(
            {
                "timestamp": chunk.index[-1],
                "theta_euler": euler.theta,
                "mu_euler": euler.mu,
                "sigma_euler": euler.sigma,
                "r2_euler": euler.r2,
                "half_life_euler": euler.half_life,
                "warnings_euler": ";".join(euler.warnings),
                "theta_exact": exact.theta,
                "mu_exact": exact.mu,
                "sigma_exact": exact.sigma,
                "r2_exact": exact.r2,
                "half_life_exact": exact.half_life,
                "warnings_exact": ";".join(exact.warnings),
            }
        )
    return pd.DataFrame(rows)


def rolling_fit_generic(
    series: pd.Series,
    window: int,
    step: int,
    dt: float = 1.0,
    drift_degree: int = 2,
    diff_degree: int = 1,
) -> pd.DataFrame:
    """Fit generic SDE model over rolling windows."""

    rows: list[dict] = []
    s = series.dropna()
    for end in range(window, len(s) + 1, step):
        chunk = s.iloc[end - window : end]
        fit = fit_generic_sde(chunk, dt=dt, drift_degree=drift_degree, diff_degree=diff_degree)
        x_last = float(chunk.iloc[-1])
        rows.append(
            {
                "timestamp": chunk.index[-1],
                "drift_coeffs": np.array2string(fit.drift_coeffs, precision=6, separator=","),
                "diff_coeffs": np.array2string(fit.diff_coeffs, precision=6, separator=","),
                "mse_drift": fit.mse_drift,
                "mse_var": fit.mse_var,
                "local_drift": fit.local_drift(x_last),
                "local_diffusion": fit.local_diffusion(x_last),
                "warnings": ";".join(fit.warnings),
            }
        )
    return pd.DataFrame(rows)
