"""Synthetic process generators for validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ou(n: int, theta: float, mu: float, sigma: float, dt: float = 1.0, x0: float | None = None, seed: int = 7) -> pd.Series:
    """Simulate OU process with Euler-Maruyama."""

    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = mu if x0 is None else x0
    for t in range(1, n):
        dW = rng.normal(0.0, np.sqrt(dt))
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * dW
    idx = pd.RangeIndex(start=0, stop=n, step=1)
    return pd.Series(x, index=idx, name="x")


def generate_poly_const_diff(
    n: int,
    drift_coeffs: np.ndarray,
    sigma: float,
    dt: float = 1.0,
    x0: float = 0.0,
    seed: int = 11,
) -> pd.Series:
    """Simulate dX = a(X)dt + sigma dW with polynomial drift."""

    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        dW = rng.normal(0.0, np.sqrt(dt))
        drift = np.polyval(drift_coeffs, x[t - 1])
        x[t] = x[t - 1] + drift * dt + sigma * dW
    return pd.Series(x, index=pd.RangeIndex(n), name="x")


def generate_poly_state_diff(
    n: int,
    drift_coeffs: np.ndarray,
    diff_coeffs: np.ndarray,
    diffusion_link: str = "softplus",
    dt: float = 1.0,
    x0: float = 0.0,
    seed: int = 13,
) -> pd.Series:
    """Simulate dX = a(X)dt + b(X)dW with positive state-dependent diffusion."""

    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = x0

    for t in range(1, n):
        xt = x[t - 1]
        drift = np.polyval(drift_coeffs, xt)
        latent = np.polyval(diff_coeffs, xt)
        if diffusion_link == "exp":
            b = np.exp(np.clip(latent, -30, 30))
        else:
            b = np.log1p(np.exp(np.clip(latent, -30, 30))) + 1e-8
        dW = rng.normal(0.0, np.sqrt(dt))
        x[t] = xt + drift * dt + b * dW
    return pd.Series(x, index=pd.RangeIndex(n), name="x")
