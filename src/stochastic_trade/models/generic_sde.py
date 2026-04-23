"""Generic 1D Itô SDE conditional moment estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass
class GenericSDEFitResult:
    drift_coeffs: np.ndarray
    diff_coeffs: np.ndarray
    diffusion_link: Literal["exp", "softplus"]
    mse_drift: float
    mse_var: float
    warnings: list[str]

    def local_drift(self, x: float) -> float:
        return float(np.polyval(self.drift_coeffs, x))

    def local_diffusion(self, x: float) -> float:
        p = float(np.polyval(self.diff_coeffs, x))
        if self.diffusion_link == "exp":
            return float(np.exp(np.clip(p, -30, 30)))
        return float(np.log1p(np.exp(np.clip(p, -30, 30))) + 1e-8)


def _positive_transform(z: np.ndarray, link: Literal["exp", "softplus"]) -> np.ndarray:
    if link == "exp":
        return np.exp(np.clip(z, -30, 30))
    return np.log1p(np.exp(np.clip(z, -30, 30))) + 1e-8


def fit_generic_sde(
    series: pd.Series,
    dt: float = 1.0,
    drift_degree: int = 2,
    diff_degree: int = 1,
    diffusion_link: Literal["exp", "softplus"] = "softplus",
) -> GenericSDEFitResult:
    """Fit polynomial drift and positive diffusion from one-step conditional moments."""

    x = series.dropna().to_numpy()
    if len(x) < 30:
        raise ValueError("Need at least 30 points for generic SDE fit")

    x_prev = x[:-1]
    dx = np.diff(x)
    y_drift = dx / dt
    y_var = (dx**2) / dt

    d_cols = np.vstack([x_prev**k for k in range(drift_degree, -1, -1)]).T
    q_cols = np.vstack([x_prev**k for k in range(diff_degree, -1, -1)]).T

    drift_init, *_ = np.linalg.lstsq(d_cols, y_drift, rcond=None)
    var_floor = np.maximum(y_var, 1e-10)
    latent_var = np.log(var_floor)
    diff_init, *_ = np.linalg.lstsq(q_cols, latent_var / 2.0, rcond=None)

    init = np.concatenate([drift_init, diff_init])

    def residuals(params: np.ndarray) -> np.ndarray:
        a = params[: drift_degree + 1]
        q = params[drift_degree + 1 :]
        drift_hat = d_cols @ a
        b = _positive_transform(q_cols @ q, diffusion_link)
        var_hat = b**2
        r1 = (drift_hat - y_drift)
        r2 = (var_hat - y_var)
        return np.concatenate([r1, 0.5 * r2])

    result = least_squares(residuals, init, max_nfev=2000)
    params = result.x
    drift_coeffs = params[: drift_degree + 1]
    diff_coeffs = params[drift_degree + 1 :]

    drift_hat = d_cols @ drift_coeffs
    b_hat = _positive_transform(q_cols @ diff_coeffs, diffusion_link)
    var_hat = b_hat**2

    mse_drift = float(np.mean((drift_hat - y_drift) ** 2))
    mse_var = float(np.mean((var_hat - y_var) ** 2))

    warnings: list[str] = []
    if not result.success:
        warnings.append(f"Optimizer did not fully converge: {result.message}")
    if np.quantile(b_hat, 0.9) > 4 * max(np.quantile(np.abs(dx), 0.9), 1e-8):
        warnings.append("Elevated diffusion estimate; model may be unstable")

    return GenericSDEFitResult(
        drift_coeffs=drift_coeffs,
        diff_coeffs=diff_coeffs,
        diffusion_link=diffusion_link,
        mse_drift=mse_drift,
        mse_var=mse_var,
        warnings=warnings,
    )
