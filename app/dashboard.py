"""Streamlit dashboard for the regime-detection research pipeline.

Design notes:
  - Phases 1-3 are run as **subprocesses**, not imported as modules. Each
    script is a standalone, argparse-driven CLI that writes its outputs to
    disk (app/models/, app/results/). Subprocess isolation means a training
    run's memory/state never leaks into the long-lived Streamlit process
    (which reruns this whole file on every widget interaction), and it keeps
    "add a new regime/trader model" to "add a script + a registry entry" --
    no dashboard internals to touch.
  - Regime detection model and trader model type are both looked up from a
    small registry dict so a second entry (e.g. an HMM regime detector, or a
    LightGBM trader) is a registry addition, not a refactor.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "app" / "scripts"
FEATURES_DIR = REPO_ROOT / "app" / "data" / "features"
RESULTS_DIR = REPO_ROOT / "app" / "results"
MODELS_DIR = REPO_ROOT / "app" / "models"
PYTHON = sys.executable

REGIME_COLORS = {
    "Bull": "#199e70",
    "Bear": "#e66767",
    "Low Volatility": "#3987e5",
    "High Volatility": "#c98500",
}

# --- Extensibility registries -------------------------------------------------
# Add a new regime detector or trader model by adding a key here; the sidebar
# selectors and pipeline runner pick it up automatically.

REGIME_MODEL_REGISTRY = {
    "GMM": {
        "label": "Gaussian Mixture Model",
        "script": SCRIPTS_DIR / "train_gmm_regimes.py",
    },
}

TRADER_MODEL_REGISTRY = {
    "XGBoost": {
        "label": "XGBoost (per-regime)",
        "train_script": SCRIPTS_DIR / "train_regime_models.py",
        "backtest_script": SCRIPTS_DIR / "backtest_regime_strategy.py",
    },
}


def discover_tickers() -> list[str]:
    if not FEATURES_DIR.exists():
        return []
    return sorted(p.stem.replace("_features", "") for p in FEATURES_DIR.glob("*_features.csv"))


def run_command(cmd: list[str], log_lines: list[str]) -> bool:
    log_lines.append(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.stdout:
        log_lines.append(proc.stdout)
    if proc.stderr:
        log_lines.append(proc.stderr)
    return proc.returncode == 0


def run_pipeline(ticker: str, regime_model: str, trader_model: str, backtest_args: dict) -> tuple[bool, str]:
    log_lines = []
    regime_cfg = REGIME_MODEL_REGISTRY[regime_model]
    trader_cfg = TRADER_MODEL_REGISTRY[trader_model]

    with st.status("Running pipeline...", expanded=True) as status:
        status.write(f"[1/3] Fitting {regime_model} regime detector...")
        ok = run_command([PYTHON, str(regime_cfg["script"]), "--ticker", ticker], log_lines)
        if not ok:
            status.update(label="Pipeline failed at regime detection", state="error")
            return False, "\n".join(log_lines)

        status.write(f"[2/3] Training per-regime {trader_model} models (walk-forward validation)...")
        ok = run_command([PYTHON, str(trader_cfg["train_script"]), "--ticker", ticker], log_lines)
        if not ok:
            status.update(label="Pipeline failed at model training", state="error")
            return False, "\n".join(log_lines)

        status.write("[3/3] Running walk-forward backtest...")
        cmd = [PYTHON, str(trader_cfg["backtest_script"]), "--ticker", ticker]
        for flag, value in backtest_args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd += [flag, str(value)]
        ok = run_command(cmd, log_lines)
        if not ok:
            status.update(label="Pipeline failed at backtest", state="error")
            return False, "\n".join(log_lines)

        status.update(label="Pipeline complete", state="complete")

    return True, "\n".join(log_lines)


def load_results(ticker: str):
    regimes_path = RESULTS_DIR / f"{ticker}_regimes.csv"
    backtest_path = RESULTS_DIR / f"backtest_{ticker}.csv"
    backtest_metrics_path = RESULTS_DIR / f"backtest_metrics_{ticker}.json"
    xgb_metrics_path = RESULTS_DIR / f"xgb_metrics_{ticker}.json"

    regimes_df = pd.read_csv(regimes_path, index_col=0, parse_dates=True) if regimes_path.exists() else None
    backtest_df = pd.read_csv(backtest_path, index_col=0, parse_dates=True) if backtest_path.exists() else None
    backtest_metrics = json.loads(backtest_metrics_path.read_text()) if backtest_metrics_path.exists() else None
    xgb_metrics = json.loads(xgb_metrics_path.read_text()) if xgb_metrics_path.exists() else None

    return regimes_df, backtest_df, backtest_metrics, xgb_metrics


# --- Chart builders -------------------------------------------------------

def regime_timeline_chart(regimes_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    labels = regimes_df["regime_label"]
    start_idx = regimes_df.index[0]
    current = labels.iloc[0]
    shapes = []
    for i in range(1, len(regimes_df)):
        if labels.iloc[i] != current:
            shapes.append(dict(
                type="rect", xref="x", yref="paper", x0=start_idx, x1=regimes_df.index[i], y0=0, y1=1,
                fillcolor=REGIME_COLORS.get(current, "#555555"), opacity=0.25, line_width=0, layer="below",
            ))
            start_idx = regimes_df.index[i]
            current = labels.iloc[i]
    shapes.append(dict(
        type="rect", xref="x", yref="paper", x0=start_idx, x1=regimes_df.index[-1], y0=0, y1=1,
        fillcolor=REGIME_COLORS.get(current, "#555555"), opacity=0.25, line_width=0, layer="below",
    ))

    fig.add_trace(go.Scatter(
        x=regimes_df.index, y=regimes_df["Close"], mode="lines",
        line=dict(color="#d8f5d8", width=2), name="Close", hovertemplate="%{y:.1f}<extra>Close</extra>",
    ))
    for label, color in REGIME_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", marker=dict(size=10, color=color, opacity=0.6),
            name=label, showlegend=True,
        ))

    fig.update_layout(
        shapes=shapes, hovermode="x unified", template="plotly_dark",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        font=dict(family="monospace", color="#d8f5d8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(gridcolor="#2c2c2a"), yaxis=dict(gridcolor="#2c2c2a", title="Close"),
    )
    return fig


def equity_chart(backtest_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=backtest_df.index, y=(backtest_df["equity"] - 1) * 100, mode="lines",
        line=dict(color="#199e70", width=2), name="Strategy",
        hovertemplate="%{y:.2f}%<extra>Strategy</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=backtest_df.index, y=(backtest_df["benchmark_equity"] - 1) * 100, mode="lines",
        line=dict(color="#c98500", width=2), name="Buy & Hold",
        hovertemplate="%{y:.2f}%<extra>Buy &amp; Hold</extra>",
    ))
    fig.update_layout(
        hovermode="x unified", template="plotly_dark",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        font=dict(family="monospace", color="#d8f5d8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(gridcolor="#2c2c2a"), yaxis=dict(gridcolor="#2c2c2a", title="Cumulative return (%)"),
    )
    return fig


def drawdown_chart(backtest_df: pd.DataFrame) -> go.Figure:
    running_max = backtest_df["equity"].cummax()
    drawdown = (backtest_df["equity"] / running_max - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=backtest_df.index, y=drawdown, mode="lines", fill="tozeroy",
        line=dict(color="#e66767", width=2), fillcolor="rgba(230,103,103,0.15)",
        name="Drawdown", hovertemplate="%{y:.2f}%<extra>Drawdown</extra>", showlegend=False,
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        font=dict(family="monospace", color="#d8f5d8"),
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(gridcolor="#2c2c2a"), yaxis=dict(gridcolor="#2c2c2a", title="Drawdown (%)"),
    )
    return fig


def regime_performance_chart(per_regime: dict) -> go.Figure:
    regimes = [r for r in REGIME_COLORS if r in per_regime]
    values = [per_regime[r]["cumulative_contribution"] * 100 for r in regimes]
    colors = [REGIME_COLORS[r] for r in regimes]
    fig = go.Figure(go.Bar(
        x=regimes, y=values, marker_color=colors, width=0.5,
        text=[f"{v:+.1f}%" for v in values], textposition="outside",
        hovertemplate="%{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        font=dict(family="monospace", color="#d8f5d8"),
        margin=dict(l=40, r=20, t=30, b=40),
        yaxis=dict(gridcolor="#2c2c2a", title="Cumulative contribution (%)"),
        xaxis=dict(gridcolor="#2c2c2a"),
    )
    return fig


# --- Hacker-terminal CSS ----------------------------------------------------

HACKER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace !important;
}

.stApp {
    background: #0a0d0a;
    color: #d8f5d8;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background: repeating-linear-gradient(
        to bottom,
        rgba(51, 255, 102, 0.015) 0px,
        rgba(51, 255, 102, 0.015) 1px,
        transparent 1px,
        transparent 3px
    );
    z-index: 9999;
}

h1, h2, h3 {
    color: #39ff6e !important;
    text-shadow: 0 0 6px rgba(57, 255, 110, 0.45);
    letter-spacing: 0.03em;
}

.term-header {
    border: 1px solid #2c4a2c;
    border-radius: 4px;
    padding: 14px 18px;
    background: #0d120d;
    margin-bottom: 1rem;
}

.term-header .prompt { color: #ffb000; }

[data-testid="stMetric"] {
    background: #0d120d;
    border: 1px solid #2c4a2c;
    border-radius: 4px;
    padding: 10px 14px;
}
[data-testid="stMetricLabel"] { color: #8fae8f !important; }
[data-testid="stMetricValue"] { color: #39ff6e !important; text-shadow: 0 0 4px rgba(57,255,110,0.35); }

.stButton > button {
    background: #0d120d;
    color: #39ff6e;
    border: 1px solid #39ff6e;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}
.stButton > button:hover {
    background: #39ff6e;
    color: #0a0d0a;
    box-shadow: 0 0 12px rgba(57, 255, 110, 0.5);
}

section[data-testid="stSidebar"] {
    background: #0d100d;
    border-right: 1px solid #2c4a2c;
}

code, .stCode {
    color: #ffb000 !important;
}
</style>
"""


def main():
    st.set_page_config(page_title="Regime Detection Terminal", layout="wide", initial_sidebar_state="expanded")
    st.markdown(HACKER_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="term-header">
        <span class="prompt">root@regime-lab</span>:~$ <span style="color:#d8f5d8;">./run_pipeline.sh --mode research</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("REGIME DETECTION TERMINAL")

    tickers = discover_tickers()
    if not tickers:
        st.error(f"No feature files found in {FEATURES_DIR}. Run ingest.py + build_features.py first.")
        return

    with st.sidebar:
        st.markdown("### CONFIG")
        ticker = st.selectbox("TICKER", tickers, index=0)
        regime_model = st.selectbox(
            "REGIME DETECTION MODEL",
            list(REGIME_MODEL_REGISTRY.keys()),
            format_func=lambda k: REGIME_MODEL_REGISTRY[k]["label"],
        )
        trader_model = st.selectbox(
            "TRADER MODEL",
            list(TRADER_MODEL_REGISTRY.keys()),
            format_func=lambda k: TRADER_MODEL_REGISTRY[k]["label"],
        )

        with st.expander("BACKTEST PARAMETERS", expanded=False):
            initial_train_size = st.number_input("Initial train size (days)", value=504, min_value=100, step=21)
            retrain_freq = st.number_input("Retrain frequency (days)", value=21, min_value=1, step=1)
            threshold = st.slider("Signal threshold P(up)", 0.5, 0.7, 0.5, 0.01)
            allow_short = st.checkbox("Allow short positions", value=False)
            cost_bps = st.number_input("Transaction cost (bps)", value=5.0, min_value=0.0, step=0.5)
            slippage_bps = st.number_input("Slippage (bps)", value=2.0, min_value=0.0, step=0.5)
            risk_free_annual = st.number_input("Risk-free rate (annual)", value=0.0, step=0.005, format="%.3f")

        run_clicked = st.button("► RUN PIPELINE", width='stretch', type="primary")

    if run_clicked:
        backtest_args = {
            "--initial-train-size": int(initial_train_size),
            "--retrain-freq": int(retrain_freq),
            "--threshold": float(threshold),
            "--allow-short": bool(allow_short),
            "--cost-bps": float(cost_bps),
            "--slippage-bps": float(slippage_bps),
            "--risk-free-annual": float(risk_free_annual),
        }
        ok, logs = run_pipeline(ticker, regime_model, trader_model, backtest_args)
        st.session_state["last_logs"] = logs
        if not ok:
            st.error("Pipeline failed -- see logs below.")
        with st.expander("SHOW LOGS", expanded=not ok):
            st.code(logs or "(no output)", language="bash")

    regimes_df, backtest_df, backtest_metrics, xgb_metrics = load_results(ticker)

    if regimes_df is None:
        st.info("No results yet for this ticker. Configure parameters in the sidebar and click RUN PIPELINE.")
        return

    tab_timeline, tab_equity, tab_regime_perf, tab_metrics = st.tabs(
        ["REGIME TIMELINE", "EQUITY / DRAWDOWN", "REGIME PERFORMANCE", "METRICS"]
    )

    with tab_timeline:
        st.plotly_chart(regime_timeline_chart(regimes_df), width='stretch')
        counts = regimes_df["regime_label"].value_counts()
        st.caption("Regime day counts: " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    with tab_equity:
        if backtest_df is not None:
            st.plotly_chart(equity_chart(backtest_df), width='stretch')
            st.plotly_chart(drawdown_chart(backtest_df), width='stretch')
        else:
            st.info("No backtest results yet -- run the pipeline.")

    with tab_regime_perf:
        if backtest_metrics is not None:
            st.plotly_chart(regime_performance_chart(backtest_metrics["per_regime"]), width='stretch')
            st.dataframe(pd.DataFrame(backtest_metrics["per_regime"]).T, width='stretch')
        else:
            st.info("No backtest results yet -- run the pipeline.")

    with tab_metrics:
        if backtest_metrics is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CUMULATIVE RETURN", f"{backtest_metrics['cumulative_return']*100:.1f}%")
            c2.metric("ANNUALIZED RETURN", f"{backtest_metrics['annualized_return']*100:.1f}%")
            c3.metric("SHARPE", f"{backtest_metrics['sharpe_ratio']:.2f}")
            c4.metric("SORTINO", f"{backtest_metrics['sortino_ratio']:.2f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("MAX DRAWDOWN", f"{backtest_metrics['max_drawdown']*100:.1f}%")
            c6.metric("WIN RATE", f"{backtest_metrics['win_rate']*100:.1f}%")
            c7.metric("AVG DAILY TURNOVER", f"{backtest_metrics['avg_daily_turnover']:.2f}")
            c8.metric("REGIME TRANSITIONS", f"{backtest_metrics['num_regime_transitions']}")

            st.caption(
                f"Benchmark (buy & hold) cumulative return: {backtest_metrics['benchmark_cumulative_return']*100:.1f}%"
                f"  |  {backtest_metrics['n_trading_days']} OOS trading days"
            )

            if xgb_metrics is not None:
                st.markdown("#### Per-regime walk-forward validation (in-sample model dev, see Phase 2)")
                rows = []
                for regime, m in xgb_metrics.items():
                    if "warning" in m:
                        rows.append({"regime": regime, "n_samples": m["n_samples"], "note": m["warning"]})
                    else:
                        fs = m["fold_summary"]
                        rows.append({
                            "regime": regime, "n_samples": m["n_samples"],
                            "accuracy": fs.get("accuracy_mean"), "roc_auc": fs.get("roc_auc_mean"),
                        })
                st.dataframe(pd.DataFrame(rows), width='stretch')
        else:
            st.info("No backtest results yet -- run the pipeline.")


if __name__ == "__main__":
    main()
