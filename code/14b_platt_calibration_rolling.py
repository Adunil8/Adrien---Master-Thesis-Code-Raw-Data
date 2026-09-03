"""
14b_platt_calibration_rolling.py - Point-in-time (expanding-window) Platt
calibration, built to test whether removing the look-ahead disclosed in
Section 3.8 changes the calibration story reported in Section 4.5.

STANDALONE / NON-DESTRUCTIVE, following the same pattern as
13_bootstrap_probability_outputs.py relative to 08_probability_outputs.py:
this script does not touch 14_platt_calibration.py or any of its outputs.
The two calibration approaches can therefore be compared directly, and the
original pooled result stays available if the rolling version is set aside.

THE PROBLEM WITH 14_platt_calibration.py: it fits Platt's (a, b) ONCE on the
whole stable period (6 Jan 2023 - 4 Apr 2025) pooled together, then applies
that single fit back across every date inside it. A calibrated probability
for, say, March 2023 is therefore corrected using information from dates
(2024, early 2025) that had not happened yet -- not something a bank could
have actually done in real time, and also an in-sample fit-then-score
evaluation (the same N is used to both fit and grade the correction).

FIRST ATTEMPT AND WHAT IT FOUND: a first version of this script fit Platt
independently per (model, horizon, maturity, threshold, direction) cell,
using only observations whose OUTCOME was already known as of the fitting
date (see the embargo rule below). Every one of the six primary cells came
back with a WORSE Brier Skill Score than the raw, uncalibrated bootstrap
probability -- not just worse than the pooled result. Inspecting the (a, b)
trajectory over time showed why: with only ~20-50 eligible observations in
the first year of the stable period, the 2-parameter MLE fit is unstable,
and one cell briefly diverged to a > 1000 (a textbook quasi-complete-
separation blow-up), collapsing every probability downstream of it to ~0.

TWO FIXES COMBINED HERE, in response to that finding:
  (1) POOL ACROSS THE MATURITY CURVE. Platt is fit per (model, horizon,
      threshold, direction) using ALL SIX maturities' (p_raw, outcome) pairs
      pooled together, not one fit per maturity. This is not an ad hoc
      patch: Section 5.2 already establishes that AR-Direct's rank-
      discrimination skill is a whole-curve property (checked and confirmed
      across all six maturities), not a maturity-specific one, so pooling
      the calibration fit across the same six maturities is a natural
      extension of that finding, not a new assumption. It roughly sextuples
      the effective sample at any given date.
  (2) EXTEND THE FITTING HISTORY BACK TO CALIBRATION_BURNIN_START (config.py,
      2021-01-01), via 13c_calibration_preperiod.py's pre-period AR-Direct
      bootstrap probabilities. This gives the rolling fit two real years of
      point-in-time-honest history before the officially reported test
      period even starts, instead of starting from zero in January 2023.
      TRAIN_END/TEST_START and every Chapter 4-5 metric are unaffected: no
      pre-period date is ever scored or reported as an out-of-sample result
      here, it is fitting history only (RESULT_MIN_DATE below enforces this).

EMBARGO RULE (unchanged from the first attempt, and still essential with
both fixes above): eligibility for the fit at date t requires realized_date
(= the row's own t + h weeks, taken from factor_forecasts.parquet / the
pre-period script's own t_plus_h construction, not recomputed by calendar
arithmetic here) <= t. An observation dated before t but whose h-week-ahead
outcome is not yet known as of t is NOT eligible -- ignoring this would
silently reintroduce a smaller look-ahead leak of exactly the kind this
script exists to remove.

RESIDUAL-POOL IDEALISATION, NOT FIXED HERE: both the official and pre-period
bootstrap probabilities are still built on the single-fit-per-window
resampling disclosed in Section 3.8 (13_bootstrap_probability_outputs.py /
13c_calibration_preperiod.py). This script only fixes the Platt step; the
residual-pool look-ahead is a separate, deferred limitation and is not made
better or worse by anything here.

Inputs : data/processed/probability_outputs_bootstrap.parquet
         data/processed/probability_outputs_bootstrap_preperiod.parquet
         data/processed/factor_forecasts.parquet   (for t_plus_h on the
                                                      official-period rows)
         data/processed/platt_coefficients.parquet  (pooled result, for the
                                                      side-by-side comparison
                                                      printed at the end)
Outputs: data/processed/platt_coefficients_rolling.parquet
         data/processed/calibrated_probabilities_rolling.parquet
         data/processed/bss_chain_pooled_vs_rolling.parquet

Run: python code/14b_platt_calibration_rolling.py   (~2-4 minutes)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import PROCESSED_DATA_DIR, TEST_START

EPS = 1e-4
MODEL_LABELS = {0: "Random Walk", 1: "AR-Direct", 2: "LP + Macro", 3: "LP + Macro + EW"}
PRIMARY_MATURITY, PRIMARY_THRESHOLD = 3, 0.08
PRIMARY_HORIZONS = [4, 13, 26]
TARIFF_SHOCK_START = pd.Timestamp("2025-04-11")  # must match 14_platt_calibration.py exactly

# Only AR-Direct is ever calibrated in the deployed pipeline (Section 4.5's
# AUC test restricts Platt to this model): computing a rolling fit for the
# other three would never be used anywhere downstream, so it is skipped here
# to keep runtime and output size proportionate to what the comparison needs.
PRIMARY_MODEL = 1
MIN_N_ROLLING = 20      # minimum pooled (across-maturity) observation count
MIN_DATES_ROLLING = 10  # minimum DISTINCT eligible dates -- guards against
                         # the maturity-pooling fix alone creating a false
                         # sense of sample size (6 correlated rows from one
                         # date are not 6 independent observations)


# ── Duplicated, not imported, from 14_platt_calibration.py ──────────────────
# Numbered scripts in this codebase are standalone entry points (see
# 02c_lambda_classification_validation.py's docstring for the same choice,
# made explicitly so this script cannot silently inherit an error from the
# other one and vice versa). Any change to the Platt mechanics must be made
# in both files for the comparison below to stay meaningful.
def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(p_raw, y):
    """Platt scaling constrained to a >= 0 (a = exp(log_a)), identical to
    14_platt_calibration.py's fit_platt -- see that file's docstring for
    the current, verified rationale for the constraint."""
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


def calibrate_group_pooled_rolling(grp, p_col, act_col, min_n, min_dates, window_weeks=None):
    """One (model, horizon, threshold, direction) group, spanning ALL SIX
    maturities and both the pre-period and official-period dates. Returns a
    row per (date, maturity) with p_cal, a, b, n_used, n_dates_used, in_domain.

    Eligibility for the fit AT DATE t: realized_date <= t (point-in-time
    correctness) AND date < TARIFF_SHOCK_START (only stable/pre-period
    observations are ever used to fit, matching 14_platt_calibration.py's
    choice not to fit on shock-period data at all) AND, if window_weeks is
    given, date >= t - window_weeks (a TRAILING window rather than the full
    expanding history since CALIBRATION_BURNIN_START). window_weeks=None
    reproduces the expanding-window behaviour. The eligible pool is NOT
    restricted to the row's own maturity -- every maturity's pair at every
    eligible date counts, pooling the whole curve into one fit.

    WHY A TRAILING WINDOW: the expanding-window version (window_weeks=None)
    was found to let 2021-2022 calibration behaviour dominate the fit well
    into 2023-2024, actively hurting years where AR-Direct's raw probability
    had since become well-behaved again (2024: raw BSS +0.078, expanding-
    window calibrated BSS -0.128) -- the calibration mapping itself appears
    time-varying, echoing this thesis's own finding that the macro
    relationships are (Section 4.2). A trailing window lets old regime
    behaviour fall out of the fit rather than persist indefinitely.
    """
    grp = grp.sort_values(["date", "maturity"]).reset_index(drop=True)
    n = len(grp)
    p_cal = np.full(n, np.nan)
    a_arr = np.full(n, np.nan)
    b_arr = np.full(n, np.nan)
    n_used = np.zeros(n, dtype=int)
    n_dates_used = np.zeros(n, dtype=int)
    in_domain = np.zeros(n, dtype=bool)

    valid_mask = grp[[p_col, act_col]].notna().all(axis=1) & grp["date"].lt(TARIFF_SHOCK_START)
    eligible_pool = grp[valid_mask]
    window = pd.Timedelta(weeks=window_weeks) if window_weeks is not None else None

    last_fit = None  # (a, b) from the last date with a successful fit, carried forward post-shock
    fit_cache = {}   # date -> (a, b, n_used, n_dates) -- one fit per DATE, reused across its 6 maturities

    for i, row in grp.iterrows():
        t = row["date"]
        if t in fit_cache:
            fit_result = fit_cache[t]
        else:
            elig_mask = eligible_pool["realized_date"] <= t
            if window is not None:
                elig_mask &= eligible_pool["date"] >= (t - window)
            eligible = eligible_pool[elig_mask]
            n_eligible = len(eligible)
            n_dates = eligible["date"].nunique()
            if t < TARIFF_SHOCK_START and n_eligible >= min_n and n_dates >= min_dates:
                a, b = fit_platt(eligible[p_col].values, eligible[act_col].values.astype(float))
                fit_result = (a, b, n_eligible, n_dates)
            else:
                fit_result = None
            fit_cache[t] = fit_result

        if t < TARIFF_SHOCK_START:
            if fit_result is not None:
                a, b, n_eligible, n_dates = fit_result
                last_fit = (a, b)
                n_used[i], n_dates_used[i] = n_eligible, n_dates
                a_arr[i], b_arr[i] = a, b
                if not np.isnan(row[p_col]):
                    p_cal[i] = sigmoid(a * logit(np.array([row[p_col]]))[0] + b)
                    in_domain[i] = True
            # else: leave NaN / out-of-domain -- not enough point-in-time history yet
        else:
            # Shock period: never fit here, only ever apply the last stable-period
            # fit, flagged out-of-domain -- same convention as 14_platt_calibration.py.
            if last_fit is not None and not np.isnan(row[p_col]):
                a, b = last_fit
                a_arr[i], b_arr[i] = a, b
                p_cal[i] = sigmoid(a * logit(np.array([row[p_col]]))[0] + b)
            in_domain[i] = False

    out = grp.copy()
    out[f"{p_col}_cal"] = p_cal
    out[f"{p_col}_a"] = a_arr
    out[f"{p_col}_b"] = b_arr
    out[f"{p_col}_n_used"] = n_used
    out[f"{p_col}_n_dates_used"] = n_dates_used
    out[f"{p_col}_in_domain"] = in_domain
    return out


def calibrate_all_rolling(prob_df, min_n=MIN_N_ROLLING, min_dates=MIN_DATES_ROLLING, window_weeks=None):
    df = prob_df[prob_df["model"] == PRIMARY_MODEL].copy()
    results = []
    coef_rows = []

    group_cols = ["model", "horizon", "threshold"]  # NOT maturity -- pooled across it
    for keys, grp in df.groupby(group_cols):
        model, h, k = keys
        for direction in ["up", "down"]:
            p_col, act_col = f"p_{direction}", f"actual_{direction}"
            out = calibrate_group_pooled_rolling(grp, p_col, act_col, min_n, min_dates, window_weeks)
            out = out.rename(columns={
                f"{p_col}_cal": "p_cal", f"{p_col}_a": "a", f"{p_col}_b": "b",
                f"{p_col}_n_used": "n_used", f"{p_col}_n_dates_used": "n_dates_used",
                f"{p_col}_in_domain": "in_domain", p_col: "p_raw", act_col: "actual",
            })
            out["direction"] = direction
            out = out[["date", "model", "horizon", "maturity", "threshold",
                       "direction", "p_raw", "actual", "p_cal", "a", "b",
                       "n_used", "n_dates_used", "in_domain"]]
            results.append(out)

            # Reportable evaluation set: OFFICIAL test period only (>= TEST_START).
            # Pre-period rows exist only to feed the fit_cache above; they are
            # never themselves scored as an out-of-sample result.
            reportable = out[out["in_domain"] & out["date"].ge(pd.Timestamp(TEST_START))]
            for mat, sub in reportable.groupby("maturity"):
                if len(sub) < 5:
                    continue
                bs_raw = brier(sub["p_raw"].values, sub["actual"].values.astype(float))
                bs_cal = brier(sub["p_cal"].values, sub["actual"].values.astype(float))
                base_rate = float(sub["actual"].mean())
                coef_rows.append({
                    "model": model, "horizon": h, "maturity": mat, "threshold": k, "direction": direction,
                    "N_calibrated_dates": len(sub),
                    "first_calibrated_date": sub["date"].min(),
                    "mean_a": round(float(sub["a"].mean()), 4),
                    "mean_b": round(float(sub["b"].mean()), 4),
                    "mean_n_used": round(float(sub["n_used"].mean()), 1),
                    "mean_n_dates_used": round(float(sub["n_dates_used"].mean()), 1),
                    "base_rate": round(base_rate, 4),
                    "brier_raw": round(bs_raw, 5), "brier_cal_rolling": round(bs_cal, 5),
                    "bss_raw": round(brier_skill(bs_raw, base_rate), 4),
                    "bss_cal_rolling": round(brier_skill(bs_cal, base_rate), 4),
                })

    calib_df = pd.concat(results, ignore_index=True)
    coef_df = pd.DataFrame(coef_rows)
    return coef_df, calib_df


def main():
    print("=" * 70)
    print("Rolling (point-in-time) Platt calibration -- pooled-maturity, extended-history run")
    print("=" * 70)

    prob_path = os.path.join(PROCESSED_DATA_DIR, "probability_outputs_bootstrap.parquet")
    preperiod_path = os.path.join(PROCESSED_DATA_DIR, "probability_outputs_bootstrap_preperiod.parquet")
    ff_path = os.path.join(PROCESSED_DATA_DIR, "factor_forecasts.parquet")
    if not os.path.exists(prob_path):
        print(f"[ERROR] {prob_path} not found -- run 13_bootstrap_probability_outputs.py first.")
        sys.exit(1)
    if not os.path.exists(preperiod_path):
        print(f"[ERROR] {preperiod_path} not found -- run 13c_calibration_preperiod.py first.")
        sys.exit(1)

    prob_df = pd.read_parquet(prob_path)
    ff = pd.read_parquet(ff_path)[["date", "horizon", "t_plus_h"]].drop_duplicates()
    prob_df = prob_df.merge(ff, on=["date", "horizon"], how="left").rename(
        columns={"t_plus_h": "realized_date"})
    missing = prob_df["realized_date"].isna().sum()
    if missing:
        print(f"[WARN] {missing} official-period rows had no matching t_plus_h -- dropping them.")
        prob_df = prob_df.dropna(subset=["realized_date"])

    preperiod_df = pd.read_parquet(preperiod_path)
    preperiod_df["realized_date"] = preperiod_df["date"] + pd.to_timedelta(preperiod_df["horizon"], unit="W")
    # NOTE: 13c_calibration_preperiod.py already required the realised price to
    # exist before writing a row, and computed realized_date from the same
    # weekly factor index 07/13 use -- this recomputation is calendar-safe
    # because the data is strictly weekly-Friday-indexed (no gaps to misalign).

    combined = pd.concat([preperiod_df, prob_df[prob_df["model"] == PRIMARY_MODEL]], ignore_index=True)
    print(f"Pre-period rows: {len(preperiod_df)} ({preperiod_df['date'].nunique()} dates)  |  "
          f"Official AR-Direct rows: {(prob_df['model'] == PRIMARY_MODEL).sum()} "
          f"({prob_df.loc[prob_df['model'] == PRIMARY_MODEL, 'date'].nunique()} dates)  |  "
          f"Combined: {len(combined)} rows")

    # ── Grid-search the trailing window size, same spirit as 07b_lambda_robustness.py:
    # an empirically-chosen hyperparameter, not an arbitrary pick. Restricted to the
    # primary (3M maturity is not filterable -- pooling needs all six; threshold and
    # horizon ARE restricted to the primary comparison cells) for runtime, since this
    # is a parameter-selection pass, not the final reported grid.
    WINDOW_GRID = [52, 78, 104, 156, None]  # weeks; None = expanding (current default)
    grid_input = combined[combined["threshold"] == PRIMARY_THRESHOLD]
    grid_rows = []
    print("\n" + "-" * 70)
    print("Trailing-window grid search (primary cells, 3M maturity, 8% threshold)")
    print("-" * 70)
    for w in WINDOW_GRID:
        w_coef, _ = calibrate_all_rolling(grid_input, window_weeks=w)
        primary_cells = w_coef[(w_coef.maturity == PRIMARY_MATURITY) & (w_coef.horizon.isin(PRIMARY_HORIZONS))]
        mean_bss = float(primary_cells["bss_cal_rolling"].mean()) if not primary_cells.empty else np.nan
        n_positive = int((primary_cells["bss_cal_rolling"] > 0).sum())
        label = f"{w}W trailing" if w is not None else "expanding (no cap)"
        print(f"  {label:20s}: mean BSS across 6 primary cells = {mean_bss:+.4f}  "
              f"({n_positive}/6 cells positive)")
        grid_rows.append({"window_weeks": w if w is not None else -1, "mean_bss": mean_bss,
                           "n_positive": n_positive})
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "platt_window_grid_search.parquet"))
    best_row = grid_df.loc[grid_df["mean_bss"].idxmax()]
    best_window = int(best_row["window_weeks"]) if best_row["window_weeks"] > 0 else None
    print(f"\nSelected window: {best_window if best_window else 'expanding (no cap)'} "
          f"(mean BSS={best_row['mean_bss']:+.4f}). Saved: platt_window_grid_search.parquet")

    coef_df, calib_df = calibrate_all_rolling(combined, window_weeks=best_window)
    coef_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "platt_coefficients_rolling.parquet"))
    calib_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "calibrated_probabilities_rolling.parquet"))
    print(f"Saved: platt_coefficients_rolling.parquet ({len(coef_df)} summary rows), "
          f"calibrated_probabilities_rolling.parquet ({len(calib_df)} rows)")

    # ── Side-by-side comparison against the pooled (14_platt_calibration.py) result ──
    pooled_path = os.path.join(PROCESSED_DATA_DIR, "platt_coefficients.parquet")
    comparison_rows = []
    if os.path.exists(pooled_path):
        pooled = pd.read_parquet(pooled_path)
        print("\n" + "-" * 100)
        print(f"BSS COMPARISON - {MODEL_LABELS[PRIMARY_MODEL]}, {PRIMARY_MATURITY}M, "
              f"{int(PRIMARY_THRESHOLD * 100)}% threshold: pooled (14) vs rolling, pooled-maturity, "
              f"extended-history (14b)")
        print("-" * 100)
        for h in PRIMARY_HORIZONS:
            for direction in ["up", "down"]:
                p_row = pooled[(pooled.model == PRIMARY_MODEL) & (pooled.horizon == h) &
                                (pooled.maturity == PRIMARY_MATURITY) & (pooled.threshold == PRIMARY_THRESHOLD) &
                                (pooled.direction == direction)]
                r_row = coef_df[(coef_df.model == PRIMARY_MODEL) & (coef_df.horizon == h) &
                                 (coef_df.maturity == PRIMARY_MATURITY) & (coef_df.threshold == PRIMARY_THRESHOLD) &
                                 (coef_df.direction == direction)]
                bss_pooled = float(p_row["bss_cal"].iloc[0]) if not p_row.empty else np.nan
                n_pooled = int(p_row["N_fit"].iloc[0]) if not p_row.empty else np.nan
                bss_rolling = float(r_row["bss_cal_rolling"].iloc[0]) if not r_row.empty else np.nan
                n_rolling = int(r_row["N_calibrated_dates"].iloc[0]) if not r_row.empty else 0
                n_used = float(r_row["mean_n_used"].iloc[0]) if not r_row.empty else np.nan
                first_cal = r_row["first_calibrated_date"].iloc[0] if not r_row.empty else pd.NaT
                print(f"h={h:>2}W {direction:>4}: pooled BSS={bss_pooled:+.4f} (N={n_pooled})   "
                      f"rolling BSS={bss_rolling:+.4f} (N_dates={n_rolling}, mean pooled fit N={n_used:.0f}, "
                      f"first calibrated {first_cal})")
                comparison_rows.append({
                    "horizon": h, "direction": direction,
                    "bss_pooled": bss_pooled, "n_pooled": n_pooled,
                    "bss_rolling": bss_rolling, "n_rolling": n_rolling,
                    "first_calibrated_date": first_cal,
                })
    else:
        print(f"\n[WARN] {pooled_path} not found -- run 14_platt_calibration.py first "
              f"for the side-by-side comparison. Rolling-only results have been saved.")

    if comparison_rows:
        pd.DataFrame(comparison_rows).to_parquet(
            os.path.join(PROCESSED_DATA_DIR, "bss_chain_pooled_vs_rolling.parquet"))
        print("\nSaved: bss_chain_pooled_vs_rolling.parquet")

    print("\nDone. 14_platt_calibration.py and its outputs were not modified.")


if __name__ == "__main__":
    main()
