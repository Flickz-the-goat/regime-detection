"""Train one XGBoost classifier per regime to predict next-day return direction.

Walk-forward validation uses sklearn's TimeSeriesSplit, which implements an
*expanding* window by default (each fold trains on all data before the test
fold) and a *rolling* window when --max-train-size is set (train set is
capped at a fixed size, so old data ages out). Expanding window makes better
use of data for regimes with few observations (e.g. Bear) but assumes the
return-generating process is stationary over the whole sample; rolling window
adapts faster to drift within a regime but throws away older, potentially
still-relevant, observations. Default here is expanding since some regimes
(Bear) are sample-starved.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

PREDICTIVE_FEATURES = [
    "trend_20d",
    "price_vs_sma20d",
    "mean_reversion_20d",
    "volatility_10_annualized",
    "momentum_12_1",
    "amihud_20d",
    "log_return",
]

REGIMES = ["Bull", "Bear", "Low Volatility", "High Volatility"]

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=4,  # fits are tiny (hundreds of rows); n_jobs=-1 thrashes on thread spawn overhead
    eval_metric="logloss",
)


def load_regime_dataset(ticker: str, features_dir: str, regimes_dir: str) -> pd.DataFrame:
    features_df = pd.read_csv(Path(features_dir) / f"{ticker}_features.csv", index_col=0, parse_dates=True)
    regimes_df = pd.read_csv(Path(regimes_dir) / f"{ticker}_regimes.csv", index_col=0, parse_dates=True)

    df = features_df.join(regimes_df[["cluster_id", "regime_label"]], how="inner")
    df["log_return"] = regimes_df["log_return"]

    df["next_day_up"] = (df["log_return"].shift(-1) > 0).astype(int)
    df = df.iloc[:-1]  # last row has an unknown next-day target
    return df


def make_splits(n_samples: int, n_splits: int, max_train_size: int | None):
    n_splits = min(n_splits, max(2, n_samples // 15 - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)
    return list(tscv.split(np.arange(n_samples)))


def evaluate_fold(y_true, y_pred, y_proba) -> dict:
    metrics = {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = None
    return metrics


def train_and_validate_regime(
    regime_name: str,
    df_regime: pd.DataFrame,
    n_splits: int = 5,
    max_train_size: int | None = None,
    min_samples: int = 40,
):
    df_regime = df_regime.sort_index()
    X = df_regime[PREDICTIVE_FEATURES].to_numpy()
    y = df_regime["next_day_up"].to_numpy()

    result = {"regime": regime_name, "n_samples": int(len(df_regime)), "folds": [], "model": None, "feature_importances": None}

    if len(df_regime) < min_samples:
        result["warning"] = f"Only {len(df_regime)} samples (< {min_samples}); skipping walk-forward validation and model fit."
        return result

    splits = make_splits(len(df_regime), n_splits, max_train_size)
    for train_idx, test_idx in splits:
        if len(set(y[train_idx])) < 2:
            continue  # can't train a classifier on a single class
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X[train_idx], y[train_idx])
        y_proba = model.predict_proba(X[test_idx])[:, 1]
        y_pred = (y_proba > 0.5).astype(int)
        result["folds"].append(evaluate_fold(y[test_idx], y_pred, y_proba))

    # final model trained on the full in-regime history (for live/current-regime
    # inference only -- NOT used to backtest this same history, see Phase 3
    # which retrains inside its own walk-forward loop to avoid lookahead)
    final_model = XGBClassifier(**XGB_PARAMS)
    final_model.fit(X, y)
    result["model"] = final_model
    result["feature_importances"] = dict(zip(PREDICTIVE_FEATURES, final_model.feature_importances_.tolist()))

    return result


def summarize_folds(folds: list[dict]) -> dict:
    if not folds:
        return {}
    summary = {}
    for key in ["accuracy", "precision", "recall", "f1"]:
        vals = [f[key] for f in folds]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
    auc_vals = [f["roc_auc"] for f in folds if f["roc_auc"] is not None]
    if auc_vals:
        summary["roc_auc_mean"] = float(np.mean(auc_vals))
        summary["roc_auc_std"] = float(np.std(auc_vals))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train per-regime XGBoost next-day-direction classifiers")
    parser.add_argument("--ticker", default="^GSPC")
    parser.add_argument("--features-dir", default="app/data/features")
    parser.add_argument("--regimes-dir", default="app/results")
    parser.add_argument("--models-dir", default="app/models")
    parser.add_argument("--output-dir", default="app/results")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--window-type", choices=["expanding", "rolling"], default="expanding")
    parser.add_argument("--rolling-size", type=int, default=252, help="train window size in days when --window-type rolling")
    args = parser.parse_args()

    max_train_size = args.rolling_size if args.window_type == "rolling" else None

    df = load_regime_dataset(args.ticker, args.features_dir, args.regimes_dir)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for regime in REGIMES:
        df_regime = df[df["regime_label"] == regime]
        print(f"\n=== {regime} ({len(df_regime)} samples) ===")
        result = train_and_validate_regime(regime, df_regime, n_splits=args.n_splits, max_train_size=max_train_size)

        if result.get("warning"):
            print(result["warning"])
            all_metrics[regime] = {"n_samples": result["n_samples"], "warning": result["warning"]}
            continue

        fold_summary = summarize_folds(result["folds"])
        print(f"Walk-forward validation ({len(result['folds'])} folds, {args.window_type} window):")
        for k, v in fold_summary.items():
            print(f"  {k}: {v:.4f}")
        print("Feature importances:")
        for feat, imp in sorted(result["feature_importances"].items(), key=lambda x: -x[1]):
            print(f"  {feat}: {imp:.4f}")

        model_path = models_dir / f"xgb_{regime.replace(' ', '_').lower()}_{args.ticker}.pkl"
        joblib.dump({"model": result["model"], "features": PREDICTIVE_FEATURES}, model_path)
        print(f"Saved model to {model_path}")

        all_metrics[regime] = {
            "n_samples": result["n_samples"],
            "n_folds": len(result["folds"]),
            "window_type": args.window_type,
            "fold_summary": fold_summary,
            "folds": result["folds"],
            "feature_importances": result["feature_importances"],
        }

    metrics_path = output_dir / f"xgb_metrics_{args.ticker}.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
