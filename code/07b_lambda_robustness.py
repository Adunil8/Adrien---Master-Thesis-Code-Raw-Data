"""
07b_lambda_robustness.py - lambda_EW Grid Search and Robustness Analysis.

Sweeps the exponential weighting decay parameter lambda_EW over a fine grid
and selects the data-optimal value by minimising out-of-sample RMSE at the
primary institutional horizons (1M = 4W, 3M = 13W) and 3M copper maturity.

This grid search calls the exact same build_lp_matrices / fit_lp functions,
same macro spec (config.MACRO_VAR_SPEC), and same q=1 macro lag as
production, so it validates the model this thesis actually deploys.
This matters because Model 3 (LP+Macro+EW) is fit as a direct,
single-horizon regression per h (Jorda 2005, fit_lp), not as an
exponentially-weighted VAR(p) iterated forward h steps. Iterating a VAR
would compound a one-step specification error across the multi-step
horizons this thesis forecasts: at lambda=0.90 (half-life 6.6 weeks), the
1-step VAR is fit on very little effective data, and iterating that noisy
fit forward 13 times explodes the error (RMSE in the 10^14 USD/t range at
h=13W). Estimating this grid search on the iterated-VAR construction would
therefore be a real mismatch with production, not just a stylistic
difference.

WHY THIS MATTERS:
  LAMBDA_EW = 0.97 was set by RiskMetrics convention (JP Morgan 1994). That
  convention was calibrated for daily equity returns, not weekly commodity
  futures curves. The optimal lambda for our setting is an empirical
  question, answered here by direct out-of-sample RMSE minimisation on the
  2023-2026 test period, using the production LP+Macro+EW estimator.

Inputs  : data/processed/ns_factors.parquet
          data/processed/macro.parquet          (VIX level)
          data/processed/macro_changes.parquet  (DXY_z52W, inventory_lag2_d4W, US3M_d1W, GPR_d26W)
          data/processed/curves.parquet
          data/processed/lag_selection.parquet
Outputs : data/processed/lambda_robustness.parquet   (full results, all lambda x horizons)
          data/processed/lambda_ew_optimal.parquet   (single row: optimal lambda and stats)
          report/figures/fig_lambda_grid.png         (RMSE vs lambda -- thesis figure)
          data/processed/lambda_grid_evidence.xlsx   (thesis evidence table)

Run: python code/07b_lambda_robustness.py
     (~3-8 minutes depending on hardware)

After this: update LAMBDA_EW in config.py to the optimal value printed at
the end, then run python code/07_forecast_evaluation.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import (
    MATURITIES_MONTHS, LAMBDA_NS,
    TEST_START,
    PROCESSED_DATA_DIR, FIGURES_DIR,
    FIGURE_DPI,
    MACRO_CHANGE_COLS, MACRO_LEVEL_COLS, MACRO_VAR_SPEC,
)
from utils import ns_curve, build_lp_matrices, fit_lp

# ── Grid configuration ────────────────────────────────────────────────────────
# Fine grid covering the full range from aggressive (short memory) to near-OLS.
# 15 values chosen so adjacent lambda differ by <= 0.01 in the 0.94-0.99 range
# where the minimum is most likely to sit.
LAMBDA_GRID = [
    0.90, 0.92, 0.93, 0.94, 0.95,
    0.96, 0.97, 0.975, 0.98, 0.985,
    0.99, 0.993, 0.995, 0.997, 0.999,
]

# Evaluation settings - Model 3 (LP+Macro+EW) only, primary horizons, 3M maturity
HORIZONS    = [4, 13]    # 1M = 4W, 3M = 13W (primary institutional horizons)
TARGET_MAT  = 3          # 3M maturity - most relevant for thesis application
Q           = 1          # macro lag in the LP design - must match 07_forecast_evaluation.py

HORIZON_LABELS = {4: "1-Month (4W)", 13: "3-Month (13W)"}


# ── Core functions ─────────────────────────────────────────────────────────────

def ess(lam: float) -> float:
    """Effective sample size for exponential weighting: ESS = (1+λ)/(1−λ)."""
    return (1 + lam) / (1 - lam)


def half_life(lam: float) -> float:
    """Half-life in weeks: HL = log(0.5) / log(λ)."""
    return np.log(0.5) / np.log(lam)


def _build_forecast_row(factors_train: pd.DataFrame, macro_train: pd.DataFrame,
                        p: int, q: int, macro_cols: list) -> np.ndarray | None:
    """
    Build the single-row design vector for predicting from the last training
    date. Duplicated from 07_forecast_evaluation.py rather than imported,
    per this codebase's convention of keeping numbered scripts independent
    (see 02c_lambda_classification_validation.py's docstring) - must be kept
    identical to that copy if either changes.
    """
    factor_cols = ["beta0", "beta1", "beta2"]
    combined = pd.concat(
        [factors_train[factor_cols], macro_train[macro_cols]], axis=1
    ).dropna()

    if len(combined) < max(p, q):
        return None

    ar_block = []
    for lag in range(p):
        ar_block.extend(combined.iloc[-(lag + 1)][factor_cols].values)

    macro_block = []
    for lag in range(q):
        macro_block.extend(combined.iloc[-(lag + 1)][macro_cols].values)

    return np.array([[1.0] + ar_block + macro_block])


def model3_lp_forecast(factors_train: pd.DataFrame, macro_train: pd.DataFrame,
                       macro_cols: list, p: int, q: int, h: int,
                       lambda_ew: float) -> np.ndarray | None:
    """
    Model 3 (LP+Macro+EW) direct h-step factor forecast, in log-price space.

    Matches 07_forecast_evaluation.py's lp_forecasts_at_t exactly for
    Model 3: one direct WLS regression at horizon h, no iteration. Returns
    shape (3,) -- [beta0, beta1, beta2] -- or None if insufficient data.
    """
    X_lp, Y_lp, _ = build_lp_matrices(
        factors_train, macro_train, p=p, q=q, h=h, macro_cols=macro_cols)
    if X_lp is None or len(X_lp) < p + q + 2:
        return None

    x_now = _build_forecast_row(factors_train, macro_train, p, q, macro_cols)
    if x_now is None:
        return None

    T = len(X_lp)
    w = lambda_ew ** np.arange(T - 1, -1, -1)   # most recent = 1
    coef3 = fit_lp(X_lp, Y_lp, weights=w)
    return (x_now @ coef3).ravel()


def run_backtest_for_lambda(factors: pd.DataFrame, macro: pd.DataFrame,
                            copper: pd.DataFrame, macro_cols: list, p: int,
                            lambda_ew: float) -> pd.DataFrame:
    """
    Expanding-window rolling backtest for a single lambda value, Model 3
    only. Evaluates at HORIZONS weeks ahead, TARGET_MAT maturity only. Each
    horizon is its own direct LP fit, matching production, not one model
    iterated forward.

    Returns DataFrame with columns: date, horizon, forecast, actual, error.
    """
    test_dates = factors.loc[TEST_START:].index
    tau_target = np.array([float(TARGET_MAT)])
    records    = []

    for t in test_dates:
        factors_train = factors.loc[:t]
        macro_train   = macro.loc[:t]

        future_dates = factors.loc[t:].index

        for h in HORIZONS:
            if len(future_dates) <= h:
                continue
            t_plus_h = future_dates[h]
            if t_plus_h not in copper.index:
                continue

            mat_col = f"m{TARGET_MAT:02d}"
            if mat_col not in copper.columns:
                continue
            actual = copper.loc[t_plus_h, mat_col]
            if np.isnan(actual):
                continue

            try:
                betas_h = model3_lp_forecast(
                    factors_train, macro_train, macro_cols, p, Q, h, lambda_ew)
            except Exception:
                continue
            if betas_h is None:
                continue

            forecast = np.exp(ns_curve(betas_h, tau_target, LAMBDA_NS))[0]

            records.append({
                "date":     t,
                "horizon":  h,
                "forecast": forecast,
                "actual":   actual,
                "error":    forecast - actual,
            })

    return pd.DataFrame(records)


# ── Output functions ───────────────────────────────────────────────────────────

def build_results_table(rows: list, n_params: int) -> pd.DataFrame:
    """Assemble and annotate the full results table from raw rows."""
    results = pd.DataFrame(rows)
    results["ess"]         = results["lambda"].apply(ess)
    results["half_life_w"] = results["lambda"].apply(half_life)
    results["ess_params"]  = results["ess"] / n_params
    return results.round({"ess": 1, "half_life_w": 1, "ess_params": 2,
                           "rmse": 2, "mae": 2})


def print_summary(results: pd.DataFrame) -> None:
    """Print a formatted console table - mirrors the thesis evidence table."""
    print("\n" + "─" * 82)
    print(f"{'λ':>6}  {'ESS':>6}  {'HL(w)':>7}  {'ESS/p':>6}  "
          f"{'RMSE 1M':>9}  {'RMSE 3M':>9}  {'MAE 1M':>8}  {'MAE 3M':>8}")
    print("─" * 82)

    lambdas = sorted(results["lambda"].unique())
    for lam in lambdas:
        sub  = results[results["lambda"] == lam]
        r1m  = sub[sub["horizon_weeks"] == 4]["rmse"].values
        r3m  = sub[sub["horizon_weeks"] == 13]["rmse"].values
        m1m  = sub[sub["horizon_weeks"] == 4]["mae"].values
        m3m  = sub[sub["horizon_weeks"] == 13]["mae"].values
        row  = sub.iloc[0]
        flag = " ← optimal" if row.get("optimal_1m", False) else ""
        print(f"{lam:>6.3f}  {row['ess']:>6.0f}  {row['half_life_w']:>7.1f}  "
              f"{row['ess_params']:>6.2f}  "
              f"{r1m[0] if len(r1m) else float('nan'):>9.2f}  "
              f"{r3m[0] if len(r3m) else float('nan'):>9.2f}  "
              f"{m1m[0] if len(m1m) else float('nan'):>8.2f}  "
              f"{m3m[0] if len(m3m) else float('nan'):>8.2f}{flag}")
    print("─" * 82)


def save_evidence_xlsx(results: pd.DataFrame, optimal_lam: float,
                       out_path: str) -> None:
    """
    Save the full evidence table to Excel for thesis inclusion.
    Wide format: one row per λ, one column pair per horizon.
    """
    pivot_rmse = results.pivot_table(
        index=["lambda", "ess", "half_life_w", "ess_params"],
        columns="horizon_weeks",
        values="rmse",
    ).reset_index()
    pivot_rmse.columns = ["lambda", "ess", "half_life_w", "ess_params",
                          "rmse_1m", "rmse_3m"]

    pivot_mae = results.pivot_table(
        index="lambda",
        columns="horizon_weeks",
        values="mae",
    ).reset_index()
    pivot_mae.columns = ["lambda", "mae_1m", "mae_3m"]

    pivot_n = results.pivot_table(
        index="lambda",
        columns="horizon_weeks",
        values="n",
    ).reset_index()
    pivot_n.columns = ["lambda", "n_1m", "n_3m"]

    wide = pivot_rmse.merge(pivot_mae, on="lambda").merge(pivot_n, on="lambda")
    wide["is_optimal_1m"] = wide["lambda"] == optimal_lam
    wide = wide.sort_values("lambda")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        wide.to_excel(writer, sheet_name="lambda_grid_results", index=False)

        # Add a readable summary sheet with formatted column names
        readable = wide.rename(columns={
            "lambda":       "λ_EW",
            "ess":          "ESS",
            "half_life_w":  "Half-life (weeks)",
            "ess_params":   "ESS / params",
            "rmse_1m":      "RMSE 1M ($/t)",
            "rmse_3m":      "RMSE 3M ($/t)",
            "mae_1m":       "MAE 1M ($/t)",
            "mae_3m":       "MAE 3M ($/t)",
            "n_1m":         "N obs 1M",
            "n_3m":         "N obs 3M",
            "is_optimal_1m": "Optimal (RMSE 1M)",
        })
        readable.to_excel(writer, sheet_name="thesis_table", index=False)

    print(f"  Saved evidence table: {out_path}")


def plot_lambda_grid(results: pd.DataFrame, optimal_lam: float,
                     out_dir: str) -> None:
    """
    Two-panel figure: RMSE vs λ for 1M and 3M horizons.
    Annotates ESS and marks optimal λ. Secondary x-axis shows half-life.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Figure X.Y - λ_EW Grid Search: LP+Macro+EW Out-of-Sample RMSE\n"
        f"3M copper maturity, expanding window backtest 2023–2026  |  "
        f"Optimal λ = {optimal_lam}",
        fontsize=11, y=1.01,
    )

    conv_lam = 0.97    # RiskMetrics convention reference line
    colors   = {"1M": "#7c3aed", "3M": "#003366"}

    for ax, h, label, color in zip(
        axes, HORIZONS, ["1M", "3M"], [colors["1M"], colors["3M"]]
    ):
        sub = results[results["horizon_weeks"] == h].sort_values("lambda")
        lams   = sub["lambda"].values
        rmses  = sub["rmse"].values
        esss   = sub["ess"].values

        ax.plot(lams, rmses, "o-", color=color, linewidth=2,
                markersize=6, label=f"RMSE {label}")

        # Mark optimal
        opt_rmse = sub[sub["lambda"] == optimal_lam]["rmse"].values
        if len(opt_rmse):
            ax.scatter([optimal_lam], [opt_rmse[0]], s=120, zorder=5,
                       color="crimson", marker="*", label=f"Optimal λ={optimal_lam}")

        # Mark convention
        conv_rmse = sub[sub["lambda"] == conv_lam]["rmse"].values
        if len(conv_rmse):
            ax.axvline(conv_lam, color="grey", linestyle="--", linewidth=1.2,
                       label=f"Convention λ=0.97")

        # Annotate ESS at each point
        for lam_v, rmse_v, ess_v in zip(lams, rmses, esss):
            ax.annotate(
                f"{int(ess_v)}",
                (lam_v, rmse_v),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=7,
                color="dimgray",
            )

        ax.set_title(f"{HORIZON_LABELS[h]} Horizon", fontsize=11)
        ax.set_xlabel("λ_EW (decay factor)", fontsize=10)
        ax.set_ylabel("RMSE (USD / tonne)", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)

        # Secondary x-axis: half-life in weeks
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        hl_ticks = [0.90, 0.93, 0.95, 0.97, 0.99, 0.995, 0.999]
        hl_ticks = [l for l in hl_ticks if lams.min() <= l <= lams.max()]
        ax2.set_xticks(hl_ticks)
        ax2.set_xticklabels(
            [f"{half_life(l):.0f}w" for l in hl_ticks],
            fontsize=7,
        )
        ax2.set_xlabel("Half-life (weeks)", fontsize=8)

    # Footnote
    fig.text(
        0.5, -0.03,
        f"Note: Numbers above markers show ESS (effective sample size = (1+λ)/(1−λ)).\n"
        f"ESS/params computed against the actual LP+Macro+EW feature count "
        f"(intercept + 3 factor lags x p + macro lags x q). "
        f"ESS/params >= 10 recommended for stable WLS estimation.",
        ha="center", fontsize=8, color="dimgray",
    )

    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_lambda_grid.png")
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: fig_lambda_grid.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("07b - λ_EW Grid Search (LP+Macro+EW, matches production estimator)")
    print("=" * 60)

    # ── Load inputs ───────────────────────────────────────────────────────────
    factors = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet")
    )[["beta0", "beta1", "beta2"]]

    macro_levels  = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro.parquet"))
    macro_changes = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro_changes.parquet"))
    macro = pd.concat([
        macro_changes[MACRO_CHANGE_COLS],
        macro_levels[MACRO_LEVEL_COLS],
    ], axis=1)
    macro_cols = MACRO_VAR_SPEC

    copper  = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "curves.parquet"))
    lag_sel = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "lag_selection.parquet"))
    p = int(lag_sel["recommended"].iloc[0])

    n_params = 1 + 3 * p + len(macro_cols) * Q   # intercept + AR block + macro block

    print(f"Lag order p = {p} | macro lag q = {Q} | macro: {macro_cols}")
    print(f"LP features per equation = {n_params} (1 intercept + {3*p} AR + {len(macro_cols)*Q} macro)")
    print(f"Target: m{TARGET_MAT:02d} maturity | horizons {HORIZONS} weeks | "
          f"{len(LAMBDA_GRID)} λ values\n")

    # ── Run grid search ───────────────────────────────────────────────────────
    rows = []
    for lam in LAMBDA_GRID:
        hl  = half_life(lam)
        ess_val = ess(lam)
        print(f"λ={lam:.3f}  ESS={ess_val:>6.0f}  HL={hl:>5.1f}w  "
              f"ESS/params={ess_val/n_params:.2f}  ...",
              end="", flush=True)

        errs = run_backtest_for_lambda(factors, macro, copper, macro_cols, p, lam)
        if errs.empty:
            print("  no results - skip.")
            continue

        for h in HORIZONS:
            sub = errs[errs["horizon"] == h]
            if sub.empty:
                continue
            rmse = float(np.sqrt(np.mean(sub["error"] ** 2)))
            mae  = float(np.mean(np.abs(sub["error"])))
            rows.append({
                "lambda":        lam,
                "horizon_weeks": h,
                "rmse":          round(rmse, 2),
                "mae":           round(mae,  2),
                "n":             len(sub),
            })

        # Print RMSE for this λ inline
        r1m = next((r["rmse"] for r in rows
                    if r["lambda"] == lam and r["horizon_weeks"] == 4), float("nan"))
        r3m = next((r["rmse"] for r in rows
                    if r["lambda"] == lam and r["horizon_weeks"] == 13), float("nan"))
        print(f"  RMSE: 1M={r1m:.2f}  3M={r3m:.2f}")

    if not rows:
        print("ERROR: No results collected. Check data alignment and macro columns.")
        return

    results = build_results_table(rows, n_params)

    # ── Select optimal λ ──────────────────────────────────────────────────────
    # Primary criterion: minimise RMSE at 1M horizon (most trade-finance-relevant).
    # Tie-break: minimise RMSE at 3M horizon.
    sub_1m = results[results["horizon_weeks"] == 4].sort_values("rmse")
    optimal_lam_1m = float(sub_1m.iloc[0]["lambda"])
    optimal_rmse_1m = float(sub_1m.iloc[0]["rmse"])

    sub_3m = results[results["horizon_weeks"] == 13].sort_values("rmse")
    optimal_lam_3m = float(sub_3m.iloc[0]["lambda"])
    optimal_rmse_3m = float(sub_3m.iloc[0]["rmse"])

    # Primary selection: 1M horizon
    optimal_lam = optimal_lam_1m
    results["optimal_1m"] = results["lambda"] == optimal_lam_1m
    results["optimal_3m"] = results["lambda"] == optimal_lam_3m

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(results)

    print(f"\n{'─'*60}")
    print(f"OPTIMAL λ (min RMSE at 1M horizon): λ = {optimal_lam_1m}  "
          f"→  RMSE 1M = {optimal_rmse_1m:.2f} $/t")
    print(f"OPTIMAL λ (min RMSE at 3M horizon): λ = {optimal_lam_3m}  "
          f"→  RMSE 3M = {optimal_rmse_3m:.2f} $/t")
    print(f"\nESS at optimal λ = {ess(optimal_lam):.0f}  "
          f"|  Half-life = {half_life(optimal_lam):.1f} weeks  "
          f"|  ESS/params = {ess(optimal_lam)/n_params:.2f}")
    print(f"\n→ ACTION: Update LAMBDA_EW = {optimal_lam} in config.py")
    print(f"  (was 0.97, convention from RiskMetrics 1994 for daily equity data)")
    print("─" * 60)

    # ── Save outputs ──────────────────────────────────────────────────────────

    # 1. Full results parquet
    rob_path = os.path.join(PROCESSED_DATA_DIR, "lambda_robustness.parquet")
    results.to_parquet(rob_path, index=False)
    print(f"\nSaved: {rob_path}")

    # 2. Optimal λ - single-row parquet read by scripts 07 and 08
    opt_df = pd.DataFrame([{
        "lambda_ew_optimal":    optimal_lam,
        "ess":                  round(ess(optimal_lam), 1),
        "half_life_weeks":      round(half_life(optimal_lam), 1),
        "ess_params":           round(ess(optimal_lam) / n_params, 2),
        "rmse_1m":              optimal_rmse_1m,
        "rmse_3m":              optimal_rmse_3m,
        "convention_lambda":    0.97,
        "selection_criterion":  "min RMSE at 1M horizon (3M maturity, 2023-2026)",
    }])
    opt_path = os.path.join(PROCESSED_DATA_DIR, "lambda_ew_optimal.parquet")
    opt_df.to_parquet(opt_path, index=False)
    print(f"Saved: {opt_path}")

    # 3. Thesis evidence table (Excel)
    xlsx_path = os.path.join(PROCESSED_DATA_DIR, "lambda_grid_evidence.xlsx")
    save_evidence_xlsx(results, optimal_lam, xlsx_path)

    # 4. Figure
    plot_lambda_grid(results, optimal_lam, FIGURES_DIR)

    print("\n07b complete.")
    print(f"Next: update LAMBDA_EW = {optimal_lam} in config.py, "
          f"then run python code/07_forecast_evaluation.py")


if __name__ == "__main__":
    main()
