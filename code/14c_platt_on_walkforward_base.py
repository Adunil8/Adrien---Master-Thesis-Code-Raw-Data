"""
14c_platt_on_walkforward_base.py - Re-runs 14_platt_calibration.py's EXACT
methodology (same fit_platt, same a>=0 constraint, same stable-period-only
fitting domain, same everything), unchanged, pointed at the walk-forward-
pool probability outputs (13f) instead of the original, idealised-pool ones.

WHY THIS SCRIPT: Section 5.2's AUC numbers already got re-tested against the
walk-forward pool (13f) and changed (AR-Direct ~0.70-0.73 -> ~0.62-0.68,
conclusion unchanged). Section 4.5's BSS-chain table, however, would
otherwise still be built on the OLD (idealised) raw probabilities -- an
internal inconsistency if left as is. This closes that gap: does the "4 of
6 primary cells go positive after Platt" story change once Platt is fit on
the corrected raw probabilities?

NOTE ON SCOPE: this reruns ONLY the original in-sample, pooled-fit Platt
methodology (the one already in the thesis text) on the new base -- it does
NOT redo the separate, larger point-in-time-honest Platt investigation
(14b: maturity-pooling, extended history, window-size grid search) on this
new base. That would be a further, larger undertaking; this script answers
the narrower, cheaper question of internal consistency first.

STANDALONE / NON-DESTRUCTIVE: does not touch 13/14/15 or their outputs.

Inputs : data/processed/probability_outputs_bootstrap_walkforward.parquet (13f)
         data/processed/platt_coefficients.parquet (original, for comparison)
         data/processed/brier_scores.parquet (Gaussian baseline, for the full chain)
Outputs: data/processed/platt_coefficients_walkforward.parquet
         data/processed/calibrated_probabilities_walkforward.parquet

Run: python code/14c_platt_on_walkforward_base.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import PROCESSED_DATA_DIR

EPS = 1e-4
MODEL_LABELS = {0: "Random Walk", 1: "AR-Direct", 2: "LP + Macro", 3: "LP + Macro + EW"}
PRIMARY_MATURITY, PRIMARY_THRESHOLD = 3, 0.08
PRIMARY_HORIZONS = [4, 13, 26]
TARIFF_SHOCK_START = pd.Timestamp("2025-04-11")
PRIMARY_MODEL = 1


# ── Identical to 14_platt_calibration.py -- see that file's fit_platt
# docstring for the current, verified rationale for the a>=0 constraint. ──
def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(p_raw, y):
    z = logit(p_raw)

    def neg_log_lik(params):
        log_a, b = params
        a = np.exp(log_a)
        p_cal = np.clip(sigmoid(a * z + b), EPS, 1 - EPS)
        return -np.mean(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal))

    res = minimize(neg_log_lik, x0=[0.0, 0.0], method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
    log_a, b = res.x
    return float(np.exp(log_a)), float(b)


def brier(p, y): return float(np.mean((p - y) ** 2))
def brier_skill(bs, base_rate):
    bs_clim = base_rate * (1 - base_rate)
    return 1 - bs / bs_clim if bs_clim > 0 else np.nan


def calibrate_all(prob_df):
    """Identical logic to 14_platt_calibration.py's calibrate_all."""
    calib_df = prob_df.copy()
    calib_df["p_up_cal"] = np.nan
    calib_df["p_down_cal"] = np.nan
    calib_df["in_calibration_domain"] = pd.to_datetime(calib_df["date"]) < TARIFF_SHOCK_START
    rows = []

    for (model, h, mat, k), grp in calib_df.groupby(["model", "horizon", "maturity", "threshold"]):
        fit_grp = grp[grp["in_calibration_domain"]]
        for direction in ["up", "down"]:
            p_col, act_col, cal_col = f"p_{direction}", f"actual_{direction}", f"p_{direction}_cal"
            valid_fit = fit_grp[[p_col, act_col]].dropna()
            if len(valid_fit) < 20:
                continue
            p_raw_fit, y_fit = valid_fit[p_col].values, valid_fit[act_col].values.astype(float)
            a, b = fit_platt(p_raw_fit, y_fit)

            all_valid = grp[[p_col, act_col]].dropna()
            p_cal_all = sigmoid(a * logit(all_valid[p_col].values) + b)
            calib_df.loc[all_valid.index, cal_col] = p_cal_all

            base_rate = float(y_fit.mean())
            bs_raw, bs_cal = brier(p_raw_fit, y_fit), brier(sigmoid(a * logit(p_raw_fit) + b), y_fit)
            rows.append({
                "model": model, "horizon": h, "maturity": mat, "threshold": k, "direction": direction,
                "N_fit": len(valid_fit), "a": round(a, 4), "b": round(b, 4), "base_rate": round(base_rate, 4),
                "brier_raw": round(bs_raw, 5), "brier_cal": round(bs_cal, 5),
                "bss_raw": round(brier_skill(bs_raw, base_rate), 4),
                "bss_cal": round(brier_skill(bs_cal, base_rate), 4),
                "mean_p_raw": round(float(p_raw_fit.mean()), 4),
            })
    return pd.DataFrame(rows), calib_df


def main():
    print("=" * 70)
    print("Platt scaling (ORIGINAL in-sample pooled methodology) on the walk-forward-pool base")
    print("=" * 70)

    prob_path = os.path.join(PROCESSED_DATA_DIR, "probability_outputs_bootstrap_walkforward.parquet")
    if not os.path.exists(prob_path):
        print(f"[ERROR] {prob_path} not found -- run 13f_walkforward_probability_and_auc.py first.")
        sys.exit(1)
    prob_df = pd.read_parquet(prob_path)
    print(f"Loaded {len(prob_df)} walk-forward-pool bootstrap probability records")

    coef_df, calib_df = calibrate_all(prob_df)
    coef_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "platt_coefficients_walkforward.parquet"))
    calib_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "calibrated_probabilities_walkforward.parquet"))
    print(f"Saved: platt_coefficients_walkforward.parquet ({len(coef_df)} rows), "
          f"calibrated_probabilities_walkforward.parquet")

    # ── Original vs walk-forward-base comparison, primary cells ────────────
    orig_path = os.path.join(PROCESSED_DATA_DIR, "platt_coefficients.parquet")
    gauss_brier_path = os.path.join(PROCESSED_DATA_DIR, "brier_scores.parquet")
    orig = pd.read_parquet(orig_path) if os.path.exists(orig_path) else None
    gauss_brier = pd.read_parquet(gauss_brier_path) if os.path.exists(gauss_brier_path) else None

    print("\n" + "-" * 100)
    print(f"BSS-CHAIN COMPARISON - {MODEL_LABELS[PRIMARY_MODEL]}, {PRIMARY_MATURITY}M, "
          f"{int(PRIMARY_THRESHOLD*100)}%: original idealised-pool base vs walk-forward-pool base")
    print("-" * 100)
    primary = coef_df[(coef_df.model == PRIMARY_MODEL) & (coef_df.maturity == PRIMARY_MATURITY) &
                       (coef_df.threshold == PRIMARY_THRESHOLD) & (coef_df.horizon.isin(PRIMARY_HORIZONS))
                       ].sort_values(["horizon", "direction"])
    n_positive_new = 0
    for _, row in primary.iterrows():
        bss_g = np.nan
        if gauss_brier is not None:
            g = gauss_brier[(gauss_brier.model == row.model) & (gauss_brier.horizon == row.horizon) &
                             (gauss_brier.maturity == row.maturity) & (gauss_brier.threshold == row.threshold) &
                             (gauss_brier.direction == row.direction)]
            if not g.empty:
                bss_g = g.iloc[0]["brier_skill"]
        bss_cal_orig = np.nan
        if orig is not None:
            o = orig[(orig.model == row.model) & (orig.horizon == row.horizon) &
                     (orig.maturity == row.maturity) & (orig.threshold == row.threshold) &
                     (orig.direction == row.direction)]
            if not o.empty:
                bss_cal_orig = o.iloc[0]["bss_cal"]
        if row.bss_cal > 0:
            n_positive_new += 1
        print(f"h={row.horizon:>2}W {row.direction:>4}: Gaussian={bss_g:+.4f}  "
              f"Bootstrap(raw, walk-fwd)={row.bss_raw:+.4f}  "
              f"+Platt(ORIGINAL base)={bss_cal_orig:+.4f}  ->  +Platt(walk-fwd base)={row.bss_cal:+.4f}   "
              f"(a={row.a:.2f}, b={row.b:.2f}, N_fit={row.N_fit})")

    print(f"\n{n_positive_new}/6 primary cells positive after Platt on the walk-forward base "
          f"(vs 4/6 on the original idealised-pool base).")
    print("\nDone. 13/14/15 and their outputs were not modified.")


if __name__ == "__main__":
    main()
