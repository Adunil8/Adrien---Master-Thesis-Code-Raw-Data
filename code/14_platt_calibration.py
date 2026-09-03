"""
14_platt_calibration.py - Post-hoc probability calibration (Platt scaling),
applied on top of the bootstrap Monte Carlo probability outputs.

CONTEXT (see 12_bootstrap_vs_gaussian_test.py and 13_bootstrap_probability_
outputs.py for the prior steps in this chain):
  Step 1 diagnosed miscalibration in the raw Gaussian probability outputs
    (negative Brier Skill Score at every horizon/threshold - Section 4.5).
  Step 2 tested whether the Gaussian DISTRIBUTIONAL SHAPE assumption was the
    cause (motivated by the Jarque-Bera non-normality result, Section 4.3):
    switching to a nonparametric bootstrap of the actual in-sample LP
    residuals improved BSS in 4 of 6 primary (horizon, direction)
    configurations, but not all - confirming shape was PART of the problem,
    not all of it.
  Step 3 (this script): the remaining gap is consistent with a SCALE problem
    - in-sample residuals mechanically understate genuine out-of-sample
    forecast uncertainty (OLS/LP residuals are, by construction, smaller than
    true prediction error). Platt scaling (Platt, 1999) corrects this
    empirically: fit p_cal = sigmoid(a*logit(p_raw) + b) by maximum
    likelihood on the realised (p_raw, outcome) pairs. This does not require
    diagnosing the exact mechanical source of the remaining gap - it
    directly matches predicted probabilities to observed frequencies.

DISCLOSURE: this is an IN-SAMPLE, POST-HOC correction - the same 2023-2026
window is used to both evaluate the model (Ch.4) and fit the calibration map
(Ch.6). It demonstrates that a standard, transparent correction removes the
MEASURED miscalibration; it is not out-of-sample evidence the correction
generalises. A production deployment would refit (a,b) periodically on
trailing realised outcomes (see Section 6.5, Limitations).

Inputs  : data/processed/probability_outputs_bootstrap.parquet
Outputs : data/processed/platt_coefficients.parquet
          data/processed/calibrated_probabilities.parquet
          report/figures/fig_reliability_calibrated_*.png

Run: python code/14_platt_calibration.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from config import PROCESSED_DATA_DIR, FIGURES_DIR

EPS = 1e-4
MODEL_LABELS = {0: "Random Walk", 1: "AR-Direct", 2: "LP + Macro", 3: "LP + Macro + EW"}

PRIMARY_MATURITY, PRIMARY_THRESHOLD = 3, 0.08
PRIMARY_HORIZONS = [4, 13, 26]
TARIFF_SHOCK_START = pd.Timestamp("2025-04-11")  # aligned with 08_probability_outputs.py / 10_subperiod_analysis.py official split date

# FINAL MODEL CHOICE: 15_calibration_diagnostics.py's AUC test shows
# LP+Macro+EW (model 3) has essentially zero rank-discrimination skill at
# any horizon (AUC ~0.5, occasional "hits" are artifacts of tiny event
# counts). AR-Direct (model 1) has real, substantial
# skill (AUC 0.60-0.85) but ONLY in the stable period (pre-April 2025) -- it
# inverts during the tariff shock, consistent with its Ch.5 hit-rate collapse
# to 6.4% in that sub-period. Calibration is therefore fit on AR-Direct,
# STABLE PERIOD ONLY: the tariff-shock period is where the underlying model
# has no validated skill to calibrate, not just miscalibrated scale -- see
# Section 4.5 / 6.5 for the regime-flag deployment implication.
PRIMARY_MODEL = 1


def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(p_raw, y):
    """
    Fit Platt scaling with the constraint a >= 0.

    NOTE: for AR-Direct's own primary cells, an unconstrained 2-parameter
    MLE already recovers a > 0 throughout, matching this constrained fit
    almost exactly. The constraint is not fixing an observed problem for
    AR-Direct specifically, it is kept as a structural guarantee: this
    function is called mechanically across
    the full (model, horizon, maturity, threshold, direction) grid, most of
    which belongs to models with no genuine ranking skill (only AR-Direct's
    output is ever reported or used downstream). Across that full grid, an
    unconstrained fit DOES produce a < 0 in roughly half of all well-posed
    cells, concentrated in exactly the models/cells with no real signal to
    rescale. The constraint ensures this function can never invert a
    ranking regardless of which cell it is applied to, without depending on
    that being empirically true for AR-Direct specifically. In the limit
    log_a -> -inf (a -> 0), the fit degenerates gracefully to "predict the
    base rate for everyone" -- the honest outcome when there is no genuine
    rank information to rescale.
    """
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
    """
    Fit Platt scaling on STABLE-PERIOD data only (see PRIMARY_MODEL note above:
    the tariff-shock period is outside the underlying model's validated skill
    domain, not just miscalibrated in scale -- fitting a calibration map to it
    would paper over a real breakdown, not correct a measurement bias).

    The fitted (a, b) map is then APPLIED to every date, including the shock
    period, but calib_df carries an `in_calibration_domain` flag so downstream
    consumers (09_stylised_deal.py) can fall back to the naive buffer outside
    the validated domain rather than trust an extrapolated calibration.
    """
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

            # Apply the fitted map to ALL dates for this (model,h,mat,k) -
            # the domain flag tells downstream code whether to trust it.
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


def plot_reliability_calibrated(calib_df, coef_row, horizon, threshold, maturity, direction,
                                 n_bins=10, save_path=None, domain_only=True):
    """domain_only=True restricts the plot to the dates the calibration was
    actually fit on (stable period) - mixing in shock-period points the fit
    never saw would misleadingly show "miscalibration" that is really just
    the model being applied outside its validated domain (see 6.5)."""
    p_col, act_col, cal_col = f"p_{direction}", f"actual_{direction}", f"p_{direction}_cal"
    sub = calib_df[(calib_df.model == coef_row["model"]) & (calib_df.horizon == horizon) &
                   (calib_df.threshold == threshold) & (calib_df.maturity == maturity)
                   ].dropna(subset=[p_col, act_col, cal_col])
    if domain_only:
        sub = sub[sub["in_calibration_domain"]]
    if sub.empty:
        return None
    bin_edges = np.linspace(0, 1, n_bins + 1)

    def binned(p_vals, act_vals):
        bin_idx = np.digitize(p_vals, bin_edges[1:-1])
        mp, fo = [], []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() < 3:
                continue
            mp.append(p_vals[mask].mean()); fo.append(act_vals[mask].mean())
        return np.array(mp), np.array(fo)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    mp_r, fo_r = binned(sub[p_col].values, sub[act_col].values.astype(float))
    mp_c, fo_c = binned(sub[cal_col].values, sub[act_col].values.astype(float))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#aaaaaa", label="Perfect calibration")
    ax.plot(mp_r, fo_r, "o-", color="#999999", lw=1.4, ms=4.5, label="Bootstrap (pre-calibration)")
    ax.plot(mp_c, fo_c, "o-", color="#1a4a8a", lw=1.6, ms=4.5, label="Platt-calibrated")
    dir_lbl = f"+{int(threshold*100)}%" if direction == "up" else f"-{int(threshold*100)}%"
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability", fontsize=9)
    ax.set_ylabel("Observed frequency", fontsize=9)
    ax.set_title(f"P(ΔF {dir_lbl}), {maturity}M maturity, {horizon}W horizon\n"
                 f"{MODEL_LABELS[coef_row['model']]} - a={coef_row['a']:.2f}, b={coef_row['b']:.2f}", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(lw=0.3, color="#dddddd"); ax.set_axisbelow(True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("=" * 70)
    print("Platt scaling calibration (applied on top of bootstrap probability outputs)")
    print("=" * 70)

    prob_path = os.path.join(PROCESSED_DATA_DIR, "probability_outputs_bootstrap.parquet")
    if not os.path.exists(prob_path):
        print(f"[ERROR] {prob_path} not found - run 13_bootstrap_probability_outputs.py first.")
        sys.exit(1)
    prob_df = pd.read_parquet(prob_path)
    print(f"Loaded {len(prob_df)} bootstrap probability records")

    coef_df, calib_df = calibrate_all(prob_df)
    print(f"Fitted Platt scaling for {len(coef_df)} (model,h,mat,k,direction) combinations")

    coef_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "platt_coefficients.parquet"))
    calib_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "calibrated_probabilities.parquet"))
    print("Saved: platt_coefficients.parquet, calibrated_probabilities.parquet")

    print("\n" + "-" * 90)
    print(f"FULL CHAIN SUMMARY (Gaussian -> Bootstrap -> +Platt) - "
          f"{MODEL_LABELS[PRIMARY_MODEL]}, {PRIMARY_MATURITY}M, {int(PRIMARY_THRESHOLD*100)}%")
    print("-" * 90)
    gauss_brier_path = os.path.join(PROCESSED_DATA_DIR, "brier_scores.parquet")
    gauss_brier = pd.read_parquet(gauss_brier_path) if os.path.exists(gauss_brier_path) else None

    primary = coef_df[(coef_df.model == PRIMARY_MODEL) & (coef_df.maturity == PRIMARY_MATURITY) &
                       (coef_df.threshold == PRIMARY_THRESHOLD) & (coef_df.horizon.isin(PRIMARY_HORIZONS))
                       ].sort_values(["horizon", "direction"])
    for _, row in primary.iterrows():
        bss_g = np.nan
        if gauss_brier is not None:
            g = gauss_brier[(gauss_brier.model == row.model) & (gauss_brier.horizon == row.horizon) &
                             (gauss_brier.maturity == row.maturity) & (gauss_brier.threshold == row.threshold) &
                             (gauss_brier.direction == row.direction)]
            if not g.empty:
                bss_g = g.iloc[0]["brier_skill"]
        print(f"h={row.horizon:>2}W {row.direction:>4}: "
              f"Gaussian BSS={bss_g:+.4f}  ->  Bootstrap BSS={row.bss_raw:+.4f}  "
              f"->  +Platt BSS={row.bss_cal:+.4f}   (a={row.a:.2f}, b={row.b:.2f}, N_fit={row.N_fit})")

    print("\nGenerating calibrated reliability diagrams...")
    fig_specs = [(4, "up", "fig_reliability_calibrated_1m_up.png"),
                 (13, "up", "fig_reliability_calibrated_3m_up.png"),
                 (13, "down", "fig_reliability_calibrated_3m_down.png"),
                 (26, "up", "fig_reliability_calibrated_6m_up.png")]
    for h, direction, fname in fig_specs:
        row = primary[(primary.horizon == h) & (primary.direction == direction)]
        if row.empty:
            continue
        fig = plot_reliability_calibrated(calib_df, row.iloc[0], h, PRIMARY_THRESHOLD,
                                           PRIMARY_MATURITY, direction,
                                           save_path=os.path.join(FIGURES_DIR, fname))
        if fig is not None:
            plt.close(fig)
            print(f"  Saved: {fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()
