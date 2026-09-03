"""
02c_lambda_classification_validation.py - Selects the Nelson-Siegel decay
parameter lambda, since fit-quality (RMSE) cannot: with only 6 maturities
and 3 NS parameters, the RMSE landscape is flat (thesis Section 3.2).

Lambda is chosen instead by a market-relevant, model-free criterion: which
value makes sign(beta1) agree most often with the curve's own, directly
observed backwardation/contango state (front-month vs the 6M contract)?
This script grid-searches that criterion directly, and also checks it
against the classic "curvature loading peaks at the sample's median
maturity" rule as a robustness comparison (that rule performs far worse,
see console output). 02_nelson_siegel.py reads this script's output as its
fallback lambda, so the value used throughout the thesis is never hardcoded.

Inputs : data/processed/curves.parquet
Outputs: data/processed/lambda_classification_validation.parquet
         (columns: lambda, agreement_pct, n_valid_dates)

Run: python code/02c_lambda_classification_validation.py
     (from thesis/ root, .venv active; run after 01_data_cleaning.py,
     before 02_nelson_siegel.py - see run_all.py for pipeline order)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import (
    PROCESSED_DATA_DIR,
    MATURITIES_MONTHS,
    LAMBDA_NS_GRID_MIN,
    LAMBDA_NS_GRID_MAX,
    LAMBDA_NS_GRID_N,
)
from utils import ns_loadings


def tau_from_columns(curves: pd.DataFrame) -> np.ndarray:
    taus = []
    for col in curves.columns:
        if col.startswith("m") and col[1:].isdigit():
            taus.append(float(col[1:]))
    return np.array(taus, dtype=float)


def fit_beta1_series(curves_log: pd.DataFrame, tau: np.ndarray, lam: float) -> pd.Series:
    """
    Fit NS at every date for one fixed lambda, return the beta1 (slope) series
    only, since that is all the classification test needs. Same OLS logic as
    02_nelson_siegel.fit_ns_date, kept independent here so this script does
    not silently inherit an error from that module.
    """
    beta1_vals = []
    for _date, row in curves_log.iterrows():
        valid = row.dropna()
        if len(valid) < 3:
            beta1_vals.append(np.nan)
            continue
        valid_tau = np.array(
            [tau[i] for i, c in enumerate(curves_log.columns) if c in valid.index]
        )
        loadings = ns_loadings(valid_tau, lam)
        betas, _, _, _ = np.linalg.lstsq(loadings, valid.values, rcond=None)
        beta1_vals.append(betas[1])
    return pd.Series(beta1_vals, index=curves_log.index, name="beta1")


def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print("=" * 70)
    print("Lambda selection: classification-accuracy validation (Section 3.2)")
    print("=" * 70)

    curves_path = os.path.join(PROCESSED_DATA_DIR, "curves.parquet")
    if not os.path.exists(curves_path):
        raise FileNotFoundError(f"{curves_path} not found. Run 01_data_cleaning.py first.")
    curves = pd.read_parquet(curves_path)

    col_names = [f"m{m:02d}" for m in MATURITIES_MONTHS]
    curves = curves[col_names]
    tau = tau_from_columns(curves)
    front_col, far_col = col_names[0], col_names[-1]
    print(f"\nMaturities: {list(tau.astype(int))} months ({len(tau)} contracts)")
    print(f"Ground-truth pair: front = {front_col} (τ={tau[0]:.0f}M), "
          f"far = {far_col} (τ={tau[-1]:.0f}M)")

    # ── Ground truth: directly observed regime, no model involved ───────────
    # Backwardation: front price above far price. Contango: front below far.
    # This uses raw prices; the sign of (front - far) is unaffected by log.
    ground_truth = np.sign(curves[front_col] - curves[far_col])
    ground_truth = ground_truth.replace(0, np.nan)  # exact ties: undefined regime, dropped
    n_backwardation = int((ground_truth > 0).sum())
    n_contango = int((ground_truth < 0).sum())
    print(f"\nObserved regime (full sample, no model): "
          f"{n_backwardation} backwardation weeks, {n_contango} contango weeks")

    # ── NS fit input: log prices, matching NS_PRICE_SCALE='log' in config ───
    curves_log = np.log(curves)

    # ── Candidate 1: the theoretical "curvature loading peaks at the sample's
    #    median maturity" rule. The NS curvature loading L(tau,lambda) is
    #    maximised at lambda*tau ≈ 1.793 (a standard closed-form NS result).
    #    Solving for lambda at the target maturity tau* = median(tau):
    tau_median = float(np.median(tau))
    lambda_theory = 1.793 / tau_median
    print(f"\nCandidate 1 (theoretical): lambda = 1.793 / median(tau) "
          f"= 1.793 / {tau_median:.2f} = {lambda_theory:.4f}")

    # ── Candidate 2: grid search directly over classification accuracy ──────
    print(f"\nGrid search over classification accuracy: "
          f"lambda in [{LAMBDA_NS_GRID_MIN}, {LAMBDA_NS_GRID_MAX}] "
          f"({LAMBDA_NS_GRID_N} points)...")

    lambda_grid = np.linspace(LAMBDA_NS_GRID_MIN, LAMBDA_NS_GRID_MAX, LAMBDA_NS_GRID_N)
    results = []
    for i, lam in enumerate(lambda_grid):
        beta1 = fit_beta1_series(curves_log, tau, lam)
        model_regime = np.sign(beta1)
        compare = pd.concat([model_regime.rename("model"), ground_truth.rename("truth")], axis=1).dropna()
        agree_pct = float((compare["model"] == compare["truth"]).mean() * 100)
        results.append({"lambda": lam, "agreement_pct": agree_pct, "n_valid_dates": len(compare)})
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(lambda_grid)} lambda values tested")

    results_df = pd.DataFrame(results)
    out_path = os.path.join(PROCESSED_DATA_DIR, "lambda_classification_validation.parquet")
    results_df.to_parquet(out_path)
    print(f"\nSaved: {out_path}")

    best_row = results_df.loc[results_df["agreement_pct"].idxmax()]
    lambda_best = float(best_row["lambda"])
    agree_best = float(best_row["agreement_pct"])

    # ── Re-check the theoretical candidate's own accuracy on the same grid ──
    beta1_theory = fit_beta1_series(curves_log, tau, lambda_theory)
    model_regime_theory = np.sign(beta1_theory)
    compare_theory = pd.concat(
        [model_regime_theory.rename("model"), ground_truth.rename("truth")], axis=1
    ).dropna()
    agree_theory = float((compare_theory["model"] == compare_theory["truth"]).mean() * 100)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Theoretical lambda (curvature peak at median maturity) = {lambda_theory:.4f}")
    print(f"  -> backwardation/contango agreement = {agree_theory:.1f}%  (rejected)")
    print(f"Selected lambda* (classification accuracy, grid optimum) = {lambda_best:.4f}")
    print(f"  -> backwardation/contango agreement = {agree_best:.1f}%  (adopted)")


if __name__ == "__main__":
    main()
