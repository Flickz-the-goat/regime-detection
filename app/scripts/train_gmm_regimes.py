import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Validated (dataviz skill, dark-surface categorical checks) 4-color set --
# keep in sync with any other regime color usage (dashboard) for consistency.
REGIME_COLORS = {
    "Bull": "#199e70",
    "Bear": "#e66767",
    "Low Volatility": "#3987e5",
    "High Volatility": "#c98500",
}

# Features chosen to span the return, volatility, and trend/momentum axes that
# define a market regime. price_vs_sma20d / mean_reversion_20d / amihud_20d are
# left out here since they're mean-reversion / liquidity signals more useful as
# XGBoost predictors than as regime-clustering inputs.
GMM_FEATURES = ["log_return", "volatility_10_annualized", "trend_20d", "momentum_12_1"]


def load_data(ticker: str, features_dir: str) -> pd.DataFrame:
    features_path = Path(features_dir) / f"{ticker}_features.csv"
    raw_path = Path(features_dir).parent / "raw" / f"{ticker}_raw.csv"

    features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    raw_df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

    log_return = np.log(raw_df["Close"] / raw_df["Close"].shift(1))
    features_df["log_return"] = log_return.reindex(features_df.index)
    features_df = features_df.dropna(subset=GMM_FEATURES)
    return features_df


def fit_gmm(features_df: pd.DataFrame, n_components: int, random_state: int):
    X = features_df[GMM_FEATURES].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        n_init=10,
    )
    cluster_ids = gmm.fit_predict(X_scaled)
    return gmm, scaler, cluster_ids


def map_clusters_to_labels(features_df: pd.DataFrame, cluster_ids: np.ndarray, n_components: int):
    """Deterministic cluster -> regime label mapping.

    Return sorts first (defines Bull/Bear, the dominant economic axis), then
    volatility separates the two remaining clusters into High/Low Volatility.
    """
    if n_components != 4:
        raise ValueError("Cluster-to-label mapping requires exactly 4 components")

    df = features_df.copy()
    df["cluster_id"] = cluster_ids
    stats = df.groupby("cluster_id").agg(
        mean_return=("log_return", "mean"),
        mean_vol=("volatility_10_annualized", "mean"),
        n_obs=("log_return", "size"),
    )

    remaining = set(stats.index)
    label_map = {}

    bull_id = stats.loc[list(remaining), "mean_return"].idxmax()
    label_map[bull_id] = "Bull"
    remaining.remove(bull_id)

    bear_id = stats.loc[list(remaining), "mean_return"].idxmin()
    label_map[bear_id] = "Bear"
    remaining.remove(bear_id)

    high_vol_id = stats.loc[list(remaining), "mean_vol"].idxmax()
    label_map[high_vol_id] = "High Volatility"
    remaining.remove(high_vol_id)

    low_vol_id = remaining.pop()
    label_map[low_vol_id] = "Low Volatility"

    stats["regime_label"] = stats.index.map(label_map)
    return label_map, stats


def plot_regimes(df: pd.DataFrame, ticker: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["Close"], color="black", linewidth=1, zorder=3)

    regime_series = df["regime_label"]
    start_idx = df.index[0]
    current_label = regime_series.iloc[0]
    for i in range(1, len(df)):
        if regime_series.iloc[i] != current_label:
            ax.axvspan(start_idx, df.index[i], color=REGIME_COLORS.get(current_label, "gray"), alpha=0.25, zorder=1)
            start_idx = df.index[i]
            current_label = regime_series.iloc[i]
    ax.axvspan(start_idx, df.index[-1], color=REGIME_COLORS.get(current_label, "gray"), alpha=0.25, zorder=1)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.25) for c in REGIME_COLORS.values()]
    ax.legend(handles, REGIME_COLORS.keys(), loc="upper left")
    ax.set_title(f"{ticker} Price with GMM-Detected Regimes")
    ax.set_ylabel("Close")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fit a Gaussian Mixture Model to detect market regimes")
    parser.add_argument("--ticker", default="^GSPC")
    parser.add_argument("--features-dir", default="app/data/features")
    parser.add_argument("--output-dir", default="app/results")
    parser.add_argument("--models-dir", default="app/models")
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    features_df = load_data(args.ticker, args.features_dir)
    gmm, scaler, cluster_ids = fit_gmm(features_df, args.n_components, args.random_state)
    label_map, stats = map_clusters_to_labels(features_df, cluster_ids, args.n_components)

    print("Cluster statistics:")
    print(stats.to_string())

    features_df["cluster_id"] = cluster_ids
    features_df["regime_label"] = features_df["cluster_id"].map(label_map)

    print("\nRegime label counts:")
    print(features_df["regime_label"].value_counts().to_string())

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"gmm_regime_{args.ticker}.pkl"
    joblib.dump(
        {"gmm": gmm, "scaler": scaler, "features": GMM_FEATURES, "label_map": label_map},
        model_path,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    regimes_path = output_dir / f"{args.ticker}_regimes.csv"
    out_cols = GMM_FEATURES + ["Close", "cluster_id", "regime_label"]
    features_df[out_cols].to_csv(regimes_path)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"{args.ticker}_regimes.png"
    plot_regimes(features_df, args.ticker, plot_path)

    print(f"\nSaved model to {model_path}")
    print(f"Saved regime labels to {regimes_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
