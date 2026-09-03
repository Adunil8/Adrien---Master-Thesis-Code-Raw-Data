"""
07_forecast_evaluation.py - Phase 4: Rolling out-of-sample forecast validation.

FORECAST DESIGN - Direct Local Projection (Jordà 2005):
  For each NS factor i and each horizon h, one OLS regression is estimated:

    β_{i,t+h} = α + Σ_{k=0}^{p-1} φ_{i,k} · [β₀_{t-k}, β₁_{t-k}, β₂_{t-k}]
                  + Σ_{j=0}^{q-1} γ_{i,j} · macro_{t-j}
                  + ε_{i,t+h}

  Macro variables enter at their OBSERVED values at time t (and t-1).
  No iterative compounding - macro conditions today directly adjust
  the h-step NS factor prediction without any intermediate forecasting.
  Residuals ε_{i,t+h} give the h-step forecast uncertainty used in
  08_probability_outputs.py to compute P(ΔF > k%).

EXPANDING WINDOW:
  Training: 2006-01-01 → 2022-12-31 (initial), grows one week per step.
  Test:     2023-01-01 → 2026-05-29.
  At each test date t, all models are re-estimated on all data up to t.

MODEL HIERARCHY:
  0: Random walk           - β̂_{i,t+h} = β_{i,t}  (no fitting)
  1: AR-Direct             - LP with p lags of all 3 NS factors, no macro
  2: LP + Macro            - LP with p lags of NS factors + macro_t [, macro_{t-1}]
  3: LP + Macro + EW       - same as Model 2, exponentially weighted observations

INPUTS:
  data/processed/ns_factors.parquet
  data/processed/macro_changes.parquet  (DXY_z52W, inventory_d4W, US3M_d26W)
  data/processed/macro.parquet          (VIX level)
  data/processed/lag_selection.parquet  (selected AR lag order p)
  data/processed/curves.parquet         (actual Bloomberg prices for error computation)
  data/processed/lambda_ew_optimal.parquet  (optimal λ_EW from 07b)

OUTPUTS:
  data/processed/forecast_errors.parquet   - raw errors per model/horizon/maturity/date
  data/processed/forecast_metrics.parquet  - RMSE, MAE, hit rate aggregated
  report/figures/fig_forecast_rmse.png     - RMSE bar chart (1M and 3M horizons)

Run: .venv/bin/python code/07_forecast_evaluation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import lstsq  # retained for _build_forecast_row usage if any

from config import (
    MATURITIES_MONTHS, LAMBDA_NS, LAMBDA_EW, VAR_MAX_LAGS,
    TRAIN_END, TEST_START,
    FORECAST_HORIZONS_WEEKS,
    PROCESSED_DATA_DIR, FIGURES_DIR,
    MACRO_CHANGE_COLS, MACRO_LEVEL_COLS, MACRO_VAR_SPEC,
)
from utils import ns_curve, build_lp_matrices, fit_lp


# ── Single forecast step ──────────────────────────────────────────────────────

def lp_forecasts_at_t(factors_train: pd.DataFrame, macro_train: pd.DataFrame,
                      macro_cols: list, p: int, q: int, lambda_ew: float,
                      horizons: list) -> dict:
    """
    For each horizon h in `horizons`, fit four LP models on the training window
    ending at the last date of factors_train, then produce one-observation forecasts.

    Returns:
        dict  {h: {model_id: np.ndarray shape (3,)}}
              model_id 0=RW, 1=AR-direct, 2=LP+macro, 3=LP+macro+EW
    """
    factor_cols = ["beta0", "beta1", "beta2"]
    last_obs    = factors_train[factor_cols].iloc[-1].values   # β at t

    result = {}
    for h in horizons:
        # ── Model 0: Random walk ──────────────────────────────────────────────
        m0 = last_obs.copy()

        # ── Build LP matrices for AR-only (Model 1) ───────────────────────────
        X_ar, Y_ar, idx_ar = build_lp_matrices(
            factors_train, macro_train, p=p, q=0, h=h, macro_cols=None)

        m1 = last_obs.copy()      # fallback = RW
        m2 = last_obs.copy()
        m3 = last_obs.copy()

        if X_ar is not None and len(X_ar) >= p + 2:
            # Model 1 - AR-direct, equal weights
            coef1 = fit_lp(X_ar, Y_ar)
            # Forecast row: [1, β₀_t, β₁_t, β₂_t, β₀_{t-1}, β₁_{t-1}, β₂_{t-1}]
            x_now = _build_forecast_row(factors_train, macro_train, p, q=0,
                                        macro_cols=None)
            if x_now is not None:
                m1 = (x_now @ coef1).ravel()

        # ── Build LP matrices for macro-augmented (Model 2, 3) ────────────────
        X_lp, Y_lp, idx_lp = build_lp_matrices(
            factors_train, macro_train, p=p, q=q, h=h, macro_cols=macro_cols)

        if X_lp is not None and len(X_lp) >= p + q + 2:
            x_now_m = _build_forecast_row(factors_train, macro_train, p, q,
                                          macro_cols=macro_cols)
            if x_now_m is not None:
                # Model 2 - LP + macro, equal weights
                coef2 = fit_lp(X_lp, Y_lp)
                m2 = (x_now_m @ coef2).ravel()

                # Model 3 - LP + macro + EW
                T = len(X_lp)
                w  = lambda_ew ** np.arange(T - 1, -1, -1)   # most recent = 1
                coef3 = fit_lp(X_lp, Y_lp, weights=w)
                m3 = (x_now_m @ coef3).ravel()

        result[h] = {0: m0, 1: m1, 2: m2, 3: m3}

    return result


def _build_forecast_row(factors_train: pd.DataFrame, macro_train: pd.DataFrame,
                        p: int, q: int, macro_cols: list | None) -> np.ndarray | None:
    """
    Build the single-row design vector for predicting from the last training date.
    Returns shape (1, n_features) or None if insufficient history.
    """
    factor_cols = ["beta0", "beta1", "beta2"]
    combined    = pd.concat(
        [factors_train[factor_cols]] +
        ([macro_train[macro_cols]] if macro_cols else []),
        axis=1
    ).dropna()

    if len(combined) < max(p, q):
        return None

    ar_block = []
    for lag in range(p):
        ar_block.extend(combined.iloc[-(lag + 1)][factor_cols].values)

    macro_block = []
    if macro_cols:
        for lag in range(q):
            macro_block.extend(combined.iloc[-(lag + 1)][macro_cols].values)

    return np.array([[1.0] + ar_block + macro_block])


# ── Rolling backtest ──────────────────────────────────────────────────────────

def rolling_backtest(factors: pd.DataFrame, macro: pd.DataFrame,
                     copper: pd.DataFrame, macro_cols: list,
                     p: int, q: int, lambda_ew: float) -> pd.DataFrame:
    """
    Expanding-window backtest over the test period 2023-2026.

    At each date t in the test period:
      1. Fit all four LP models on factors + macro from 2006 to t.
      2. For each horizon h, generate forecasted NS factors.
      3. Reconstruct the forecasted copper curve (exp(NS)).
      4. Compare against actual Bloomberg prices at t+h.
      5. Record errors for RMSE/MAE/hit-rate computation.
    """
    tau           = np.array(MATURITIES_MONTHS, dtype=float)
    test_dates    = factors.loc[TEST_START:].index
    records       = []
    factor_records = []

    print(f"\nRolling LP backtest: {len(test_dates)} test dates | "
          f"horizons {FORECAST_HORIZONS_WEEKS}W | models 0–3")
    print("Progress: ", end="", flush=True)

    for i, t in enumerate(test_dates):
        if i % 20 == 0:
            print(f"{i}/{len(test_dates)}...", end="", flush=True)

        factors_train = factors.loc[:t]
        macro_train   = macro.loc[:t]

        # Get forecasts for ALL horizons at this training cutoff
        try:
            forecasts = lp_forecasts_at_t(
                factors_train, macro_train, macro_cols,
                p=p, q=q, lambda_ew=lambda_ew,
                horizons=FORECAST_HORIZONS_WEEKS,
            )
        except Exception as e:
            print(f"\n  [t={t.date()}] forecast error: {e}")
            continue

        # Current price for hit-rate computation
        current_prices = copper.loc[t].values if t in copper.index else None

        for h in FORECAST_HORIZONS_WEEKS:
            # Find actual date at t+h (weekly steps from t onward)
            future_idx = factors.index[factors.index > t]
            if len(future_idx) < h:
                continue
            t_ph = future_idx[h - 1]
            if t_ph not in copper.index:
                continue

            actual_prices = copper.loc[t_ph].values  # observed Bloomberg prices

            for model_id in [0, 1, 2, 3]:
                betas_h = forecasts[h][model_id]   # predicted (β₀, β₁, β₂)

                # Save factor-level forecasts (needed by 08 for fan chart / Σ_h)
                factor_records.append({
                    "date":      t,
                    "horizon":   h,
                    "model":     model_id,
                    "beta0_hat": round(float(betas_h[0]), 6),
                    "beta1_hat": round(float(betas_h[1]), 6),
                    "beta2_hat": round(float(betas_h[2]), 6),
                    "t_plus_h":  t_ph,
                })

                # Reconstruct forecasted curve (factors are log-scale)
                fitted_prices = np.exp(ns_curve(betas_h, tau, LAMBDA_NS))

                for j, mat in enumerate(MATURITIES_MONTHS):
                    if j >= len(actual_prices) or np.isnan(actual_prices[j]):
                        continue
                    error = fitted_prices[j] - actual_prices[j]
                    cp_j  = (current_prices[j]
                             if current_prices is not None and j < len(current_prices)
                             else np.nan)
                    records.append({
                        "date":          t,
                        "horizon":       h,
                        "maturity":      mat,
                        "model":         model_id,
                        "forecast":      round(fitted_prices[j], 4),
                        "actual":        round(actual_prices[j], 4),
                        "current_price": round(cp_j, 4) if not np.isnan(cp_j) else np.nan,
                        "error":         round(error, 4),
                    })

    print("done.")
    return pd.DataFrame(records), pd.DataFrame(factor_records)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(errors_df: pd.DataFrame) -> pd.DataFrame:
    """
    RMSE, MAE, and directional hit rate per model/horizon/maturity.
    Hit rate: fraction of forecasts where sign(forecast change) == sign(actual change).
    RW (model 0) always predicts zero change → hit rate undefined (NaN).
    """
    metrics = []
    for (model, h, mat), grp in errors_df.groupby(["model", "horizon", "maturity"]):
        grp = grp.sort_values("date")
        e   = grp["error"].values
        fc  = grp["forecast"].values
        ac  = grp["actual"].values

        rmse = np.sqrt(np.mean(e ** 2))
        mae  = np.mean(np.abs(e))

        hit_rate = np.nan
        if "current_price" in grp.columns and model != 0:
            cp    = grp["current_price"].values
            valid = ~np.isnan(cp)
            if valid.sum() > 0:
                fc_chg = fc[valid] - cp[valid]
                ac_chg = ac[valid] - cp[valid]
                moved  = ac_chg != 0
                if moved.sum() > 0:
                    hit_rate = float(np.mean(
                        np.sign(fc_chg[moved]) == np.sign(ac_chg[moved])
                    ))

        metrics.append({
            "model":    model,
            "horizon":  h,
            "maturity": mat,
            "RMSE":     round(rmse, 4),
            "MAE":      round(mae,  4),
            "hit_rate": round(hit_rate, 4) if not np.isnan(hit_rate) else np.nan,
            "N":        len(grp),
        })
    return pd.DataFrame(metrics)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_rmse_comparison(metrics: pd.DataFrame, out_dir: str) -> None:
    """Bar chart of RMSE by model for 1M and 3M horizons at 3M maturity."""
    model_labels = {0: "RW", 1: "AR-Direct", 2: "LP+Macro", 3: "LP+Macro+EW"}
    colors       = ["#cccccc", "#aec6cf", "#779ecb", "#003366"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, h_w, h_lbl in zip(axes, [4, 13], ["1-Month (4W)", "3-Month (13W)"]):
        sub = metrics[
            (metrics["horizon"]  == h_w) &
            (metrics["maturity"] == 3)
        ].sort_values("model")
        if sub.empty:
            ax.set_title(f"RMSE - {h_lbl}\n(no data)")
            continue
        bars = ax.bar(
            [model_labels.get(m, str(m)) for m in sub["model"]],
            sub["RMSE"], color=colors[:len(sub)]
        )
        ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set_title(f"RMSE - {h_lbl} Horizon\n(3M maturity, test 2023–2026)")
        ax.set_ylabel("RMSE (USD/tonne)")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_forecast_rmse.png"), dpi=150,
                facecolor="white")
    plt.close(fig)
    print("  Saved: fig_forecast_rmse.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("Phase 4 - Direct LP Forecast Evaluation (2023–2026)")
    print("=" * 60)

    # Load NS factors
    factors = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet")
    )[["beta0", "beta1", "beta2"]]

    # Macro specification - same 5 variables used in the VAR (05_models.py):
    #   DXY_z52W, inventory_d4W, US3M_d26W, GPR_d26W  (from macro_changes.parquet)
    #   VIX                                            (level, from macro.parquet)
    macro_changes = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "macro_changes.parquet"))
    macro_levels  = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "macro.parquet"))
    macro = pd.concat([
        macro_changes[MACRO_CHANGE_COLS],
        macro_levels[MACRO_LEVEL_COLS],
    ], axis=1)
    macro_cols = MACRO_VAR_SPEC

    # Actual copper prices (Bloomberg) for error computation
    copper = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "curves.parquet"))

    # Lag order (AR lags in the LP design matrix)
    lag_sel = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "lag_selection.parquet"))
    p = int(lag_sel["recommended"].iloc[0])   # AR lags of NS factors

    # Macro lags in LP: q=1 means only current macro (t), q=2 adds one lag (t-1)
    # q=1 is the primary specification - today's macro directly conditions β_{t+h}
    Q = 1

    # λ_EW: use data-optimal value from 07b if available
    opt_path = os.path.join(PROCESSED_DATA_DIR, "lambda_ew_optimal.parquet")
    if os.path.exists(opt_path):
        lambda_ew = float(pd.read_parquet(opt_path)["lambda_ew_optimal"].iloc[0])
        print(f"λ_EW = {lambda_ew:.4f} (data-optimal from 07b grid search)")
    else:
        lambda_ew = LAMBDA_EW
        print(f"λ_EW = {lambda_ew:.4f} (config default)")

    print(f"AR lags p = {p} | Macro lags q = {Q}")
    print(f"Macro columns: {macro_cols}")
    print(f"Test period: {TEST_START} → {copper.index.max().date()}")
    print(f"LP features per equation: 1 (intercept) + {3*p} (AR) + {len(macro_cols)*Q} (macro)"
          f" = {1 + 3*p + len(macro_cols)*Q} total")

    errors_df, factor_df = rolling_backtest(factors, macro, copper, macro_cols,
                                             p=p, q=Q, lambda_ew=lambda_ew)

    if errors_df.empty:
        print("WARNING: No forecast errors recorded. Check data alignment.")
        return

    # Save raw errors
    err_path = os.path.join(PROCESSED_DATA_DIR, "forecast_errors.parquet")
    errors_df.to_parquet(err_path)
    print(f"\nSaved forecast errors: {err_path}")

    # Save factor-level forecasts (β̂₀, β̂₁, β̂₂ per date/horizon/model)
    fct_path  = os.path.join(PROCESSED_DATA_DIR, "factor_forecasts.parquet")
    factor_df.to_parquet(fct_path)
    print(f"Saved factor forecasts: {fct_path}")

    # Compute and save metrics
    metrics = compute_metrics(errors_df)
    met_path = os.path.join(PROCESSED_DATA_DIR, "forecast_metrics.parquet")
    metrics.to_parquet(met_path)
    print(f"Saved forecast metrics: {met_path}")

    # Print summary table
    model_labels = {0: "RW", 1: "AR-Direct", 2: "LP+Macro", 3: "LP+Macro+EW"}
    print("\n" + "─" * 70)
    print("SUMMARY - RMSE and Hit Rate (3M maturity)")
    print("─" * 70)
    for h_w, h_lbl in [(4, "1M (4W)"), (9, "2M (9W)"), (13, "3M (13W)"), (26, "6M (26W)")]:
        sub = metrics[(metrics["horizon"] == h_w) & (metrics["maturity"] == 3)
                      ].sort_values("model")
        if sub.empty:
            continue
        print(f"\n{h_lbl} horizon:")
        print(f"  {'Model':18s} {'RMSE':>8s} {'MAE':>8s} {'HitRate':>9s} {'N':>5s}")
        for _, row in sub.iterrows():
            lbl = model_labels.get(row["model"], str(row["model"]))
            hr  = f"{row['hit_rate']*100:.1f}%" if not np.isnan(row["hit_rate"]) else "  n/a"
            print(f"  {lbl:18s} {row['RMSE']:>8.1f} {row['MAE']:>8.1f} {hr:>9s} {int(row['N']):>5d}")

    # RW comparison
    print("\n" + "─" * 70)
    print("Best model vs Random Walk (RMSE ratio, 3M maturity):")
    for h_w, h_lbl in [(4, "1M"), (9, "2M"), (13, "3M"), (26, "6M")]:
        sub = metrics[(metrics["horizon"] == h_w) & (metrics["maturity"] == 3)]
        if sub.empty:
            continue
        rw_rmse   = sub[sub["model"] == 0]["RMSE"].values[0]
        rest      = sub[sub["model"] != 0]
        best_row  = rest.loc[rest["RMSE"].idxmin()]
        ratio     = best_row["RMSE"] / rw_rmse
        verdict   = "BEATS RW ✓" if ratio < 1.0 else f"+{(ratio-1)*100:.1f}% vs RW"
        print(f"  {h_lbl:4s}: RW={rw_rmse:.1f}  best={best_row['RMSE']:.1f} "
              f"({model_labels[best_row['model']]})  ratio={ratio:.3f}  [{verdict}]")

    plot_rmse_comparison(metrics, FIGURES_DIR)

    print("\nPhase 4 complete.")
    print("Next: run .venv/bin/python code/08_probability_outputs.py")


if __name__ == "__main__":
    main()
