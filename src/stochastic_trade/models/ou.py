"""Ornstein-Uhlenbeck estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class OUFitResult:
    theta: float
    mu: float
    sigma: float
    half_life: float | None
    r2: float
    warnings: list[str]


def fit_ou_euler(series: pd.Series, dt: float = 1.0) -> OUFitResult:
    """Estimate OU parameters using Euler-discretized regression."""

    x = series.dropna().to_numpy()
    if len(x) < 20:
        raise ValueError("Need at least 20 points for OU fit")

    dx = np.diff(x)
    x_prev = x[:-1]
    y = dx / dt
    X = sm.add_constant(x_prev)
    model = sm.OLS(y, X).fit()

    alpha, beta = model.params
    theta = -beta
    mu = alpha / theta if abs(theta) > 1e-10 else float(np.mean(x))
    resid_var = float(np.var(model.resid, ddof=1))
    sigma = float(np.sqrt(max(resid_var * dt, 1e-12)))
    half_life = (np.log(2) / theta) if theta > 1e-8 else None

    warnings = []
    if theta <= 0:
        warnings.append("Non-mean-reverting fit: theta <= 0")
    if model.rsquared < 0.02:
        warnings.append("Low explanatory power (R^2)")

    return OUFitResult(theta=theta, mu=mu, sigma=sigma, half_life=half_life, r2=float(model.rsquared), warnings=warnings)


def fit_ou_exact(series: pd.Series, dt: float = 1.0) -> OUFitResult:
    """Estimate OU via exact AR(1)-equivalent discretization."""

    x = series.dropna().to_numpy()
    if len(x) < 20:
        raise ValueError("Need at least 20 points for OU fit")

    x_t = x[1:]
    x_tm1 = x[:-1]
    X = sm.add_constant(x_tm1)
    model = sm.OLS(x_t, X).fit()

    c, phi = model.params
    phi = float(np.clip(phi, -0.999999, 0.999999))
    theta = -np.log(max(phi, 1e-8)) / dt if phi > 0 else -1.0
    mu = c / (1 - phi) if abs(1 - phi) > 1e-10 else float(np.mean(x))

    var_eps = float(np.var(model.resid, ddof=1))
    if theta > 1e-8:
        sigma = float(np.sqrt(var_eps * (2 * theta) / (1 - np.exp(-2 * theta * dt))))
        half_life = float(np.log(2) / theta)
    else:
        sigma = float(np.sqrt(max(var_eps / dt, 1e-12)))
        half_life = None

    warnings = []
    if theta <= 0:
        warnings.append("Non-mean-reverting fit in exact estimator")
    if model.rsquared < 0.02:
        warnings.append("Low AR(1) explanatory power (R^2)")

    return OUFitResult(theta=float(theta), mu=float(mu), sigma=sigma, half_life=half_life, r2=float(model.rsquared), warnings=warnings)
