"""Walk-forward backtest of the GMM regime classifier + per-regime XGBoost models.

To avoid the lookahead bias flagged in Phase 1 (fitting the GMM once on full
history leaks future information into past regime labels), this script
retrains the *entire* pipeline -- GMM regime detector and all per-regime
XGBoost models -- inside its own expanding-window walk-forward loop:

  - Every `--retrain-freq` trading days, refit the GMM + per-regime XGBoost
    models using only data strictly before the current day.
  - Between retrains, use the frozen models to classify each day's regime
    and generate that day's trading signal.
  - Convention: a signal computed from day i's features (which use data
    through close_i) is "traded" at close_i and its return is realized over
    close_i -> close_{i+1}. This is the standard sign-at-close/hold-to-next-
    close daily backtest convention and avoids intraday lookahead.

This intentionally duplicates (rather than reuses the saved artifacts of)
train_gmm_regimes.py / train_regime_models.py: those scripts fit once on the
full history for descriptive analysis and live/current-regime inference;
this script needs point-in-time-only models for a fair historical
simulation, so it imports their fitting functions and calls them fresh at
each retrain checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from train_gmm_regimes import GMM_FEATURES, fit_gmm, load_data, map_clusters_to_labels
from train_regime_models import PREDICTIVE_FEATURES, REGIMES, XGB_PARAMS

MIN_REGIME_TRAIN_SAMPLES = 30


def build_dataset(ticker: str, features_dir: str) -> pd.DataFrame:
    df = load_data(ticker, features_dir)  # GMM_FEATURES + PREDICTIVE_FEATURES overlap + Close, causal
    df = df.sort_index()
    df["fwd_return"] = df["log_return"].shift(-1)
    df["next_day_up"] = (df["fwd_return"] > 0).astype(int)
    df = df.iloc[:-1]  # last row's forward return is unknown
    return df


def fit_pipeline(train_df: pd.DataFrame, n_components: int, random_state: int):
    """Fit GMM regime detector + one XGBoost classifier per well-populated regime, using only train_df."""
    gmm, scaler, cluster_ids = fit_gmm(train_df, n_components, random_state)
    label_map, _ = map_clusters_to_labels(train_df, cluster_ids, n_components)

    regime_labels = pd.Series(cluster_ids, index=train_df.index).map(label_map)

    xgb_models = {}
    for regime in REGIMES:
        mask = regime_labels == regime
        if mask.sum() < MIN_REGIME_TRAIN_SAMPLES:
            continue
        X = train_df.loc[mask, PREDICTIVE_FEATURES].to_numpy()
        y = train_df.loc[mask, "next_day_up"].to_numpy()
        if len(set(y)) < 2:
            continue
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X, y)
        xgb_models[regime] = model

    return gmm, scaler, label_map, xgb_models


def classify_and_predict(row: pd.Series, gmm, scaler, label_map, xgb_models):
    x_gmm = scaler.transform(row[GMM_FEATURES].to_numpy().reshape(1, -1))
    cluster_id = int(gmm.predict(x_gmm)[0])
    regime_label = label_map.get(cluster_id, "Unknown")

    model = xgb_models.get(regime_label)
    if model is None:
        return regime_label, None
    x_pred = row[PREDICTIVE_FEATURES].to_numpy().reshape(1, -1)
    proba_up = float(model.predict_proba(x_pred)[0, 1])
    return regime_label, proba_up


def position_from_proba(proba_up, threshold: float, allow_short: bool) -> int:
    if proba_up is None:
        return 0
    if proba_up > threshold:
        return 1
    if allow_short and proba_up < (1 - threshold):
        return -1
    return 0


def run_backtest(
    df: pd.DataFrame,
    initial_train_size: int,
    retrain_freq: int,
    n_components: int,
    threshold: float,
    allow_short: bool,
    cost_bps: float,
    slippage_bps: float,
    random_state: int,
):
    n = len(df)
    if initial_train_size >= n:
        raise ValueError("initial_train_size must be smaller than the dataset length")

    records = []
    gmm = scaler = label_map = xgb_models = None
    prev_position = 0

    for i in range(initial_train_size, n):
        if (i - initial_train_size) % retrain_freq == 0:
            train_df = df.iloc[:i]
            gmm, scaler, label_map, xgb_models = fit_pipeline(train_df, n_components, random_state)

        row = df.iloc[i]
        regime_label, proba_up = classify_and_predict(row, gmm, scaler, label_map, xgb_models)
        position = position_from_proba(proba_up, threshold, allow_short)

        turnover = abs(position - prev_position)
        cost = turnover * (cost_bps + slippage_bps) / 10000.0
        fwd_return = float(row["fwd_return"])
        strategy_return = position * fwd_return - cost

        records.append(
            {
                "date": df.index[i],
                "regime_label": regime_label,
                "proba_up": proba_up,
                "position": position,
                "turnover": turnover,
                "cost": cost,
                "market_return": fwd_return,
                "strategy_return": strategy_return,
            }
        )
        prev_position = position

    results = pd.DataFrame(records).set_index("date")
    results["equity"] = np.exp(results["strategy_return"].cumsum())
    results["benchmark_equity"] = np.exp(results["market_return"].cumsum())
    return results


def compute_metrics(results: pd.DataFrame, risk_free_annual: float, periods_per_year: int = 252) -> dict:
    r = results["strategy_return"]
    rf_daily = risk_free_annual / periods_per_year
    excess = r - rf_daily

    equity = results["equity"]
    cumulative_return = float(equity.iloc[-1] - 1)
    n_years = len(r) / periods_per_year
    annualized_return = float(equity.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else float("nan")
    annualized_vol = float(r.std() * np.sqrt(periods_per_year))

    sharpe = float(excess.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else float("nan")
    downside = r[r < 0]
    sortino = (
        float(excess.mean() / downside.std() * np.sqrt(periods_per_year))
        if len(downside) > 1 and downside.std() > 0
        else float("nan")
    )

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = float(drawdown.min())

    active = results[results["position"] != 0]
    win_rate = float((active["strategy_return"] > 0).mean()) if len(active) > 0 else float("nan")

    turnover = float(results["turnover"].mean())
    num_regime_transitions = int((results["regime_label"] != results["regime_label"].shift(1)).sum() - 1)

    benchmark_cumulative_return = float(results["benchmark_equity"].iloc[-1] - 1)

    per_regime = {}
    for regime, g in results.groupby("regime_label"):
        per_regime[regime] = {
            "n_days": int(len(g)),
            "mean_daily_return": float(g["strategy_return"].mean()),
            "win_rate": float((g["strategy_return"] > 0).mean()),
            "cumulative_contribution": float(np.exp(g["strategy_return"].sum()) - 1),
        }

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_daily_turnover": turnover,
        "num_regime_transitions": num_regime_transitions,
        "n_trading_days": int(len(results)),
        "benchmark_cumulative_return": benchmark_cumulative_return,
        "per_regime": per_regime,
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the GMM + per-regime XGBoost strategy")
    parser.add_argument("--ticker", default="^GSPC")
    parser.add_argument("--features-dir", default="app/data/features")
    parser.add_argument("--output-dir", default="app/results")
    parser.add_argument("--initial-train-size", type=int, default=504, help="trading days of history before the first signal is generated (~2y)")
    parser.add_argument("--retrain-freq", type=int, default=21, help="trading days between pipeline refits (~1 month)")
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5, help="P(up) above this -> long; symmetric deadband if --allow-short")
    parser.add_argument("--allow-short", action="store_true", help="allow short positions; default is long/flat only")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="transaction cost in bps of notional per unit turnover")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="slippage in bps of notional per unit turnover")
    parser.add_argument("--risk-free-annual", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = build_dataset(args.ticker, args.features_dir)
    print(f"Loaded {len(df)} rows; generating signals for {len(df) - args.initial_train_size} out-of-sample days "
          f"(retraining every {args.retrain_freq} days, {'long/short' if args.allow_short else 'long/flat'})")

    results = run_backtest(
        df,
        initial_train_size=args.initial_train_size,
        retrain_freq=args.retrain_freq,
        n_components=args.n_components,
        threshold=args.threshold,
        allow_short=args.allow_short,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        random_state=args.random_state,
    )
    metrics = compute_metrics(results, args.risk_free_annual)
    metrics["config"] = vars(args)

    print("\nBacktest metrics:")
    for k, v in metrics.items():
        if k in ("per_regime", "config"):
            continue
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("Per-regime breakdown:")
    for regime, stats in metrics["per_regime"].items():
        print(f"  {regime}: {stats}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"backtest_{args.ticker}.csv"
    results.to_csv(results_path)

    metrics_path = output_dir / f"backtest_metrics_{args.ticker}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\nSaved daily results to {results_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
