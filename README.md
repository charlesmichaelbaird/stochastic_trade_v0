# Stochastic Trade Research (Inverse SDE)

A **research-oriented** Python repository for inverse stochastic differential equation (SDE) modeling of stock time series.

## Core philosophy

- Raw stock price is **not assumed** to follow OU.
- OU is implemented as a **baseline sanity check** only.
- Primary model family is a general 1D Itô SDE:

\[
  dX_t = a(X_t)dt + b(X_t)dW_t
\]

with configurable drift and strictly-positive diffusion parameterizations.

## Repository layout

- `src/stochastic_trade/` - all reusable logic (data, features, models, signals, backtest)
- `tests/` - synthetic-data validation and parameter recovery tests
- `data/raw/` - raw payload cache location
- `data/processed/` - normalized bars and model outputs
- `configs/` - research config examples
- `notebooks/` - optional exploratory notebooks (logic should stay in `src/`)

## Dependencies

Python 3.11+ and:
- pandas
- numpy
- scipy
- statsmodels
- matplotlib
- pydantic
- requests
- pytest
- yfinance (free data in v1)

## Data layer

Provider interface implemented with two providers:
1. `yfinance` for easy historical research access
2. `twelvedata` (optional, API key required)

All provider outputs are normalized to:
`timestamp, open, high, low, close, volume, symbol, interval`

## Modeled series options

`build_modeled_series()` supports:
- `raw_close`
- `log_close`
- `rolling_demeaned_log_close`
- `rolling_zscore_close`
- `realized_vol_proxy`

## Modeling

### OU baseline
- Euler-Maruyama inversion estimator (`fit_ou_euler`)
- Exact discrete OU estimator (`fit_ou_exact`)
- Rolling fitting utilities in `models/rolling.py`
- OU half-life reported where applicable

### Generic inverse SDE
- Conditional-moment estimator (`fit_generic_sde`)
- Polynomial drift `a(x)`
- Positive diffusion `b(x)` via `exp` or `softplus` link
- Diagnostics include parameter vectors, MSE metrics, local drift/diffusion, and warnings

## Signal research layer

Experimental signals only (not production trading):
- OU-like: mean-deviation + positive mean-reversion + quality gate
- Generic SDE: local drift scaled by diffusion, with high-diffusion suppression
- Hysteresis and cooldown logic included

## Backtesting

No-lookahead close-to-close backtest with:
- position sizing stub (bounded position)
- transaction cost parameter (bps)
- slippage parameter (bps)

Outputs:
- trades table
- equity curve
- parameter time series (from rolling fits)
- signal overlay chart

## CLI

Single entrypoint:

```bash
stoch-research fetch-data --provider yfinance --symbol AAPL --interval 1d --start 2020-01-01 --end 2024-12-31
stoch-research update-latest --provider yfinance --symbol AAPL --interval 1h
stoch-research fit-ou --input data/processed/AAPL_1d_2020-01-01_2024-12-31.csv --series-mode log_close --rolling-window 252 --step 5 --output data/processed/ou_params.csv
stoch-research fit-generic-sde --input data/processed/AAPL_1d_2020-01-01_2024-12-31.csv --series-mode log_close --drift-degree 2 --diff-degree 1 --rolling-window 252 --step 5 --output data/processed/generic_params.csv
stoch-research generate-signals --model ou --bars data/processed/AAPL_1d_2020-01-01_2024-12-31.csv --params data/processed/ou_params.csv --output data/processed/signals_ou.csv
stoch-research backtest --bars data/processed/AAPL_1d_2020-01-01_2024-12-31.csv --signals data/processed/signals_ou.csv --trades-output data/processed/trades.csv --equity-output data/processed/equity.csv
```

## Validation approach

Validate on synthetic data first:
- OU process generator
- Polynomial drift + constant diffusion generator
- Polynomial drift + state-dependent diffusion generator

Tests verify parameter recovery behavior and positivity/stability constraints.

## Warnings & risk notes

- This project is for **research and education**, not financial advice.
- Inverse SDE estimation is sensitive to sampling interval, regime shifts, and microstructure noise.
- Strong backtest performance on one symbol/period can fail out-of-sample.
- Use robust out-of-sample evaluation and stress testing before any live use.
