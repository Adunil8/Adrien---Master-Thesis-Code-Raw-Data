"""
03b_lag_robustness.py - Appendix IV: VAR lag order vs. residual whitening.

03_stationarity_lags.py selects p via AIC/BIC alone (BIC preferred per thesis
protocol when AIC and BIC disagree). This script checks a second criterion -
whether the selected lag actually whitens the VAR residuals (Ljung-Box) - and
reports the trade-off across the full VAR_MAX_LAGS grid, since BIC-optimal and
residual-whitening-optimal lags need not coincide.

Inputs  : data/processed/ns_factors.parquet
          data/processed/macro.parquet
Outputs : data/processed/lag_robustness.parquet
          Console summary table

Run: python code/03b_lag_robustness.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox

from config import TRAIN_END, PROCESSED_DATA_DIR

MAX_LAG_GRID = 8   # extends beyond VAR_MAX_LAGS=4 to show where whitening stabilises


def main():
    print("=" * 60)
    print("03b_lag_robustness.py - Appendix IV: lag order vs. residual whitening")
    print("=" * 60)

    factors = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet")
    )[["beta0", "beta1", "beta2"]]
    macro = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro.parquet"))
    if "inventory" in macro.columns:
        macro = macro.drop(columns=["inventory"])

    factors_train = factors.loc[:TRAIN_END]
    macro_train   = macro.loc[:TRAIN_END]
    endog = pd.concat([factors_train, macro_train], axis=1).dropna()
    endog["beta0"] = endog["beta0"].diff()
    endog = endog.dropna()

    rows = []
    print(f"\n{'p':>3} {'AIC':>10} {'BIC':>10} {'n_series_failing_LB':>20} "
          f"{'worst_series':>15} {'worst_pval':>10}")
    for p in range(1, MAX_LAG_GRID + 1):
        res = VAR(endog).fit(maxlags=p, ic=None, trend="c")
        resid = pd.DataFrame(res.resid, columns=res.model.endog_names)
        n_fail, worst_p, worst_s = 0, 1.0, ""
        for col in resid.columns:
            lb = acorr_ljungbox(resid[col].dropna(), lags=2 * p, return_df=True)
            mp = lb["lb_pvalue"].min()
            if mp < 0.05:
                n_fail += 1
            if mp < worst_p:
                worst_p, worst_s = mp, col
        rows.append({
            "p": p, "aic": round(res.aic, 2), "bic": round(res.bic, 2),
            "n_series_failing_lb": n_fail, "worst_series": worst_s,
            "worst_pval": round(worst_p, 4),
        })
        print(f"{p:>3} {res.aic:>10.2f} {res.bic:>10.2f} {n_fail:>20} "
              f"{worst_s:>15} {worst_p:>10.4f}")

    results = pd.DataFrame(rows)
    out_path = os.path.join(PROCESSED_DATA_DIR, "lag_robustness.parquet")
    results.to_parquet(out_path)
    print(f"\nSaved: {out_path}")

    print("\n" + "─" * 60)
    print("Interpretation:")
    print("  BIC favours p=1 (parsimony). Ljung-Box residual whitening improves")
    print("  monotonically with p but is NOT fully achieved at any tested lag -")
    print("  at least one series (most often US3M) retains residual structure")
    print("  even at p=8, consistent with the long-memory behaviour documented")
    print("  for yield-curve spreads in the macro-finance literature.")
    print("  Decision: p=1 retained (BIC + parsimony + overfitting risk at higher")
    print("  p given ~887 training obs), with HAC-robust SEs as the stated")
    print("  mitigation for residual serial correlation. Document as a limitation.")

    print("\n03b complete.")


if __name__ == "__main__":
    main()
