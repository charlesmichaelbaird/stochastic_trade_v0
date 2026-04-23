"""CLI entrypoint for stochastic SDE research workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stochastic_trade.backtest.engine import run_backtest
from stochastic_trade.data.fetcher import fetch_and_cache, update_latest_bars
from stochastic_trade.features.series import build_modeled_series
from stochastic_trade.models.generic_sde import fit_generic_sde
from stochastic_trade.models.ou import fit_ou_euler, fit_ou_exact
from stochastic_trade.models.rolling import rolling_fit_generic, rolling_fit_ou
from stochastic_trade.signals.generate import generate_generic_signals, generate_ou_signals
from stochastic_trade.utils.plotting import plot_signal_overlay


def _load_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")


def cmd_fetch_data(args: argparse.Namespace) -> None:
    path = fetch_and_cache(
        provider_name=args.provider,
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        twelvedata_api_key=args.twelvedata_api_key,
    )
    print(path)


def cmd_update_latest(args: argparse.Namespace) -> None:
    path = update_latest_bars(
        provider_name=args.provider,
        symbol=args.symbol,
        interval=args.interval,
        processed_dir=Path(args.processed_dir),
        twelvedata_api_key=args.twelvedata_api_key,
    )
    print(path)


def cmd_fit_ou(args: argparse.Namespace) -> None:
    bars = _load_bars(args.input)
    series = build_modeled_series(bars, mode=args.series_mode).dropna()

    if args.rolling_window:
        out = rolling_fit_ou(series, window=args.rolling_window, step=args.step, dt=args.dt)
        out.to_csv(args.output, index=False)
        print(args.output)
        return

    euler = fit_ou_euler(series, dt=args.dt)
    exact = fit_ou_exact(series, dt=args.dt)
    print(json.dumps({"euler": euler.__dict__, "exact": exact.__dict__}, indent=2, default=str))


def cmd_fit_generic(args: argparse.Namespace) -> None:
    bars = _load_bars(args.input)
    series = build_modeled_series(bars, mode=args.series_mode).dropna()

    if args.rolling_window:
        out = rolling_fit_generic(
            series,
            window=args.rolling_window,
            step=args.step,
            dt=args.dt,
            drift_degree=args.drift_degree,
            diff_degree=args.diff_degree,
        )
        out.to_csv(args.output, index=False)
        print(args.output)
        return

    fit = fit_generic_sde(
        series,
        dt=args.dt,
        drift_degree=args.drift_degree,
        diff_degree=args.diff_degree,
        diffusion_link=args.diffusion_link,
    )
    print(
        json.dumps(
            {
                "drift_coeffs": fit.drift_coeffs.tolist(),
                "diff_coeffs": fit.diff_coeffs.tolist(),
                "mse_drift": fit.mse_drift,
                "mse_var": fit.mse_var,
                "warnings": fit.warnings,
            },
            indent=2,
        )
    )


def cmd_generate_signals(args: argparse.Namespace) -> None:
    bars = _load_bars(args.bars)
    params = pd.read_csv(args.params)
    params["timestamp"] = pd.to_datetime(params["timestamp"], utc=True)

    if args.model == "ou":
        series = build_modeled_series(bars, mode=args.series_mode).dropna()
        out = generate_ou_signals(series, params)
    else:
        out = generate_generic_signals(params)

    out.to_csv(args.output, index=False)
    plot_signal_overlay(bars, out, Path(args.output).with_suffix(".png"))
    print(args.output)


def cmd_backtest(args: argparse.Namespace) -> None:
    bars = _load_bars(args.bars)
    signals = pd.read_csv(args.signals)
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)

    trades, curve = run_backtest(
        bars,
        signals,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )
    trades.to_csv(args.trades_output, index=False)
    curve.to_csv(args.equity_output, index=False)
    print(json.dumps({"trades": args.trades_output, "equity": args.equity_output}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    p = argparse.ArgumentParser(prog="stoch-research")
    sp = p.add_subparsers(dest="command", required=True)

    fetch = sp.add_parser("fetch-data")
    fetch.add_argument("--provider", default="yfinance")
    fetch.add_argument("--symbol", required=True)
    fetch.add_argument("--interval", default="1d")
    fetch.add_argument("--start")
    fetch.add_argument("--end")
    fetch.add_argument("--raw-dir", default="data/raw")
    fetch.add_argument("--processed-dir", default="data/processed")
    fetch.add_argument("--twelvedata-api-key")
    fetch.set_defaults(func=cmd_fetch_data)

    update = sp.add_parser("update-latest")
    update.add_argument("--provider", default="yfinance")
    update.add_argument("--symbol", required=True)
    update.add_argument("--interval", default="1d")
    update.add_argument("--processed-dir", default="data/processed")
    update.add_argument("--twelvedata-api-key")
    update.set_defaults(func=cmd_update_latest)

    fit_ou = sp.add_parser("fit-ou")
    fit_ou.add_argument("--input", required=True)
    fit_ou.add_argument("--series-mode", default="log_close")
    fit_ou.add_argument("--dt", type=float, default=1.0)
    fit_ou.add_argument("--rolling-window", type=int)
    fit_ou.add_argument("--step", type=int, default=10)
    fit_ou.add_argument("--output", default="data/processed/ou_params.csv")
    fit_ou.set_defaults(func=cmd_fit_ou)

    fit_generic = sp.add_parser("fit-generic-sde")
    fit_generic.add_argument("--input", required=True)
    fit_generic.add_argument("--series-mode", default="log_close")
    fit_generic.add_argument("--dt", type=float, default=1.0)
    fit_generic.add_argument("--drift-degree", type=int, default=2)
    fit_generic.add_argument("--diff-degree", type=int, default=1)
    fit_generic.add_argument("--diffusion-link", choices=["exp", "softplus"], default="softplus")
    fit_generic.add_argument("--rolling-window", type=int)
    fit_generic.add_argument("--step", type=int, default=10)
    fit_generic.add_argument("--output", default="data/processed/generic_params.csv")
    fit_generic.set_defaults(func=cmd_fit_generic)

    sig = sp.add_parser("generate-signals")
    sig.add_argument("--model", choices=["ou", "generic"], required=True)
    sig.add_argument("--bars", required=True)
    sig.add_argument("--params", required=True)
    sig.add_argument("--series-mode", default="log_close")
    sig.add_argument("--output", default="data/processed/signals.csv")
    sig.set_defaults(func=cmd_generate_signals)

    bt = sp.add_parser("backtest")
    bt.add_argument("--bars", required=True)
    bt.add_argument("--signals", required=True)
    bt.add_argument("--transaction-cost-bps", type=float, default=2.0)
    bt.add_argument("--slippage-bps", type=float, default=1.0)
    bt.add_argument("--trades-output", default="data/processed/trades.csv")
    bt.add_argument("--equity-output", default="data/processed/equity.csv")
    bt.set_defaults(func=cmd_backtest)

    return p


def main() -> None:
    """CLI main."""

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
