"""Tests for generic SDE estimator."""

from __future__ import annotations

import numpy as np

from stochastic_trade.models.generic_sde import fit_generic_sde
from stochastic_trade.synthetic.generators import generate_poly_const_diff, generate_poly_state_diff


def test_generic_constant_diffusion_recovery() -> None:
    drift_true = np.array([-0.2, 0.4])
    series = generate_poly_const_diff(n=2500, drift_coeffs=drift_true, sigma=0.25, dt=0.05)

    fit = fit_generic_sde(series, dt=0.05, drift_degree=1, diff_degree=0, diffusion_link="exp")

    assert fit.mse_drift < 1.5
    assert fit.mse_var < 2.0
    assert len(fit.drift_coeffs) == 2
    assert len(fit.diff_coeffs) == 1


def test_generic_state_dependent_diffusion_positive() -> None:
    drift = np.array([-0.15, 0.1])
    diff = np.array([0.1, -0.4])
    series = generate_poly_state_diff(n=2000, drift_coeffs=drift, diff_coeffs=diff, dt=0.05)

    fit = fit_generic_sde(series, dt=0.05, drift_degree=1, diff_degree=1, diffusion_link="softplus")

    x0 = float(series.iloc[-1])
    assert fit.local_diffusion(x0) > 0
    assert np.isfinite(fit.local_drift(x0))
