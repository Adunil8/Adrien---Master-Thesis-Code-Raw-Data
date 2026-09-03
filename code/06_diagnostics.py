"""
06_diagnostics.py - Phase 3d: VAR residual diagnostics (Section 4.3).

Mandatory tests:
  E1: Portmanteau / Ljung-Box - serial correlation in residuals
  E2: Engle ARCH test          - conditional heteroscedasticity
Optional tests:
  E3: Jarque-Bera              - normality of residuals
  E4: VIF                      - multicollinearity among macro inputs

Inputs  : data/processed/model_fits.pkl
Outputs : data/processed/diagnostics_results.parquet
          Console output for Section 4.3

Run: python code/06_diagnostics.py

Interpretation:
  Ljung-Box significant → increase lag order
  ARCH detected → use HAC robust SE (already default) ± GARCH robustness check
  Jarque-Bera rejected → normality violated; note as limitation on probability outputs
  VIF > 10 → consider dropping the collinear macro variable
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import PROCESSED_DATA_DIR


def test_ljung_box(residuals: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    """Ljung-Box test for serial correlation in each residual series."""
    rows = []
    for col in residuals.columns:
        res = acorr_ljungbox(residuals[col].dropna(), lags=lags, return_df=True)
        min_pval = res["lb_pvalue"].min()
        rows.append({
            "series":      col,
            "min_pval":    round(min_pval, 4),
            "significant": min_pval < 0.05,
            "verdict":     "SERIAL CORRELATION - increase lag order" if min_pval < 0.05
                           else "No serial correlation (OK)",
        })
    return pd.DataFrame(rows)


def test_arch(residuals: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Engle ARCH test for conditional heteroscedasticity."""
    rows = []
    for col in residuals.columns:
        res_sq = residuals[col].dropna()
        try:
            lm_stat, lm_pval, f_stat, f_pval = het_arch(res_sq, nlags=lags)
            rows.append({
                "series":      col,
                "LM_stat":     round(lm_stat, 4),
                "LM_pval":     round(lm_pval, 4),
                "significant": lm_pval < 0.05,
                "verdict":     "ARCH DETECTED - use HAC SE, consider GARCH robustness"
                               if lm_pval < 0.05 else "No ARCH (OK)",
            })
        except Exception as e:
            rows.append({"series": col, "verdict": f"Test failed: {e}"})
    return pd.DataFrame(rows)


def test_jarque_bera(residuals: pd.DataFrame) -> pd.DataFrame:
    """Jarque-Bera normality test on each residual series."""
    rows = []
    for col in residuals.columns:
        s = residuals[col].dropna()
        jb_stat, jb_pval = jarque_bera(s)
        rows.append({
            "series":    col,
            "JB_stat":   round(jb_stat, 4),
            "JB_pval":   round(jb_pval, 4),
            "normal":    jb_pval > 0.05,
            "verdict":   "NORMAL (OK)" if jb_pval > 0.05
                         else "NON-NORMAL - note as limitation on Gaussian probability outputs",
        })
    return pd.DataFrame(rows)


def test_vif(endog: pd.DataFrame) -> pd.DataFrame:
    """VIF for multicollinearity among all endogenous variables."""
    rows = []
    X = endog.dropna().values
    for i, col in enumerate(endog.columns):
        try:
            vif = variance_inflation_factor(X, i)
            rows.append({
                "variable": col,
                "VIF":      round(vif, 2),
                "verdict":  "HIGH MULTICOLLINEARITY (consider dropping)" if vif > 10
                            else "OK",
            })
        except Exception as e:
            rows.append({"variable": col, "verdict": f"Failed: {e}"})
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Phase 3d - VAR Residual Diagnostics")
    print("=" * 60)

    model_path = os.path.join(PROCESSED_DATA_DIR, "model_fits.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Run 05_models.py first.")

    with open(model_path, "rb") as f:
        store = pickle.load(f)

    var_ew   = store["var_ew"]
    endog    = store["endog_train"]
    p        = store["lag_order"]

    # Extract residuals from Model 3 (VAR + macro + EW) - dict from utils.fit_var_ew
    residuals = pd.DataFrame(
        var_ew["resid"],
        columns=var_ew["cols"]
    )

    print(f"Residual matrix: {residuals.shape[0]} observations × {residuals.shape[1]} series")

    # ── E1: Ljung-Box ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("E1 - Ljung-Box: Serial Correlation in Residuals")
    print("─" * 60)
    lb_results = test_ljung_box(residuals, lags=2 * p)
    print(lb_results.to_string(index=False))

    # ── E2: ARCH test ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("E2 - Engle ARCH Test: Conditional Heteroscedasticity")
    print("─" * 60)
    arch_results = test_arch(residuals)
    print(arch_results.to_string(index=False))
    print("\n  NOTE: HAC robust standard errors are ALWAYS used (regardless of ARCH result)")
    print("  as commodity markets exhibit volatility clustering.")

    # ── E3: Jarque-Bera (optional) ────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("E3 - Jarque-Bera: Normality of Residuals (Optional)")
    print("─" * 60)
    jb_results = test_jarque_bera(residuals)
    print(jb_results.to_string(index=False))
    print("\n  NOTE: If normality rejected, Gaussian probability outputs (P(ΔF > k%))")
    print("  are approximations. Document as limitation in Section 4.3 / Section 7.")

    # ── E4: VIF (optional) ────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("E4 - VIF: Multicollinearity Among Macro Variables (Optional)")
    print("─" * 60)
    vif_results = test_vif(endog)
    print(vif_results.to_string(index=False))

    # ── Save all results ──────────────────────────────────────────────────────
    all_results = {
        "ljung_box": lb_results,
        "arch":      arch_results,
        "jb":        jb_results,
        "vif":       vif_results,
    }
    out_path = os.path.join(PROCESSED_DATA_DIR, "diagnostics_results.parquet")
    lb_results.to_parquet(out_path.replace(".parquet", "_ljb.parquet"))
    arch_results.to_parquet(out_path.replace(".parquet", "_arch.parquet"))
    jb_results.to_parquet(out_path.replace(".parquet", "_jb.parquet"))
    vif_results.to_parquet(out_path.replace(".parquet", "_vif.parquet"))
    print(f"\nSaved diagnostic results to {PROCESSED_DATA_DIR}/")

    print("\nPhase 3d complete.")
    print("Next: run python code/07_forecast_evaluation.py")


if __name__ == "__main__":
    main()
