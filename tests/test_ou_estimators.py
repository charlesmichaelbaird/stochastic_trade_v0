"""Tests for OU estimators."""

from __future__ import annotations

from stochastic_trade.models.ou import fit_ou_euler, fit_ou_exact
from stochastic_trade.synthetic.generators import generate_ou


def test_ou_parameter_recovery() -> None:
    series = generate_ou(n=3000, theta=0.25, mu=1.5, sigma=0.4, dt=0.1)

    fit_e = fit_ou_euler(series, dt=0.1)
    fit_x = fit_ou_exact(series, dt=0.1)

    assert abs(fit_e.theta - 0.25) < 0.08
    assert abs(fit_x.theta - 0.25) < 0.08
    assert abs(fit_e.mu - 1.5) < 0.2
    assert abs(fit_x.mu - 1.5) < 0.2
    assert fit_e.sigma > 0
    assert fit_x.sigma > 0
