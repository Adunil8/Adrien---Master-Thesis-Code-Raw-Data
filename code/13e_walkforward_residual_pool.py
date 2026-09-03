"""
13e_walkforward_residual_pool.py - Builds the genuinely walk-forward,
point-in-time-honest residual pool for models 1 (AR-Direct), 2 (LP+Macro),
and 3 (LP+Macro+EW), replacing the single-fit-per-window construction
disclosed as an idealisation in Section 3.8. Model 0 (Random Walk) is not
included: it has no fitted coefficient, so its residual pool (an actual
realised difference, not an estimate) was never idealised in the first
place -- there is nothing for this fix to correct there.

STANDALONE / NON-DESTRUCTIVE: does not touch 07/13/14/15/09 or their
outputs. Produces its own file, combined from two segments:

  (1) PRE-TEST segment (RESIDUAL_POOL_BURNIN_START, config.py, 2012-01-01,
      through just before TEST_START): freshly computed here. At every
      origin s in this range, refits each model using ONLY factors/macro
      data up to and including s (mirrors 07_forecast_evaluation.py's own
      lp_forecasts_at_t exactly, just walked earlier and further than the
      officially reported test period), forecasts beta_{s+h}, and records
      the single honest residual: actual beta_{s+h} minus that forecast.
      2006-2011 is deliberately excluded (RESIDUAL_POOL_BURNIN_START), per
      13d_walkforward_stability_diagnostic.py's finding that all three
      models' coefficients are still visibly unstable in that stretch
      (median week-to-week coefficient change 0.02-0.53, only settling into
      a flat 0.003-0.03 regime from 2012 onward) -- a criterion chosen by
      watching the coefficients alone, never by checking any forecast-
      accuracy, BSS, or AUC result.

  (2) OFFICIAL TEST PERIOD segment (TEST_START onward): NOT recomputed --
      07_forecast_evaluation.py already re-estimates at every test date
      using only data up to that date, so factor_forecasts.parquet's own
      beta_hat is already exactly the quantity needed. This segment is
      just: actual beta at t+h (ns_factors.parquet) minus that already-
      correct forecast (factor_forecasts.parquet). No new model fitting.

Each origin contributes exactly ONE residual per (model, horizon) -- unlike
the current production pool, which pools MANY origins' residuals but all
computed from a single, later, "future-informed" coefficient. Here it is
reversed: many origins, each with its OWN period-appropriate coefficient.

Output: data/processed/walkforward_residuals.parquet
  columns: date (origin s), horizon, model, resid_beta0/1/2, realized_date,
           segment ("preperiod_fresh" | "official_reused")

Run: python code/13e_walkforward_residual_pool.py   (~30-90 seconds)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import (
    TEST_START, RESIDUAL_POOL_BURNIN_START, FORECAST_HORIZONS_WEEKS,
    PROCESSED_DATA_DIR, MACRO_CHANGE_COLS, MACRO_LEVEL_COLS, MACRO_VAR_SPEC,
    LAMBDA_EW,
)
from utils import build_lp_matrices, fit_lp

FACTOR_COLS = ["beta0", "beta1", "beta2"]


# ── Duplicated, not imported, from 07_forecast_evaluation.py (models 1-3
# branches only -- model 0/Random Walk needs no fix, see module docstring).
# Same isolation rationale as every other numbered script in this codebase
# (see 02c_lambda_classification_validation.py's docstring). ──────────────
def _build_forecast_row(factors_train, macro_train, p, q, macro_cols):
    combined = pd.concat(
        [factors_train[FACTOR_COLS]] + ([macro_train[macro_cols]] if macro_cols else []),
        axis=1,
    ).dropna()
    if len(combined) < max(p, q):
        return None
    ar_block = []
    for lag in range(p):
        ar_block.extend(combined.iloc[-(lag + 1)][FACTOR_COLS].values)
    macro_block = []
    if macro_cols:
        for lag in range(q):
            macro_block.extend(combined.iloc[-(lag + 1)][macro_cols].values)
    return np.array([[1.0] + ar_block + macro_block])


def forecasts_at_origin(factors_train, macro_train, macro_cols, p, q, lambda_ew, h):
    """Models 1, 2, 3 forecasts of beta_{s+h}, made using only data through
    origin s. Falls back to the last observed factors (random-walk-like) if
    there is not yet enough history for a given model at this origin."""
    last_obs = factors_train[FACTOR_COLS].iloc[-1].values
    out = {1: last_obs.copy(), 2: last_obs.copy(), 3: last_obs.copy()}

    X_ar, Y_ar, _ = build_lp_matrices(factors_train, None, p=p, q=0, h=h, macro_cols=None)
    if X_ar is not None and len(X_ar) >= p + 2:
        coef1 = fit_lp(X_ar, Y_ar)
        x_now = _build_forecast_row(factors_train, None, p, 0, None)
        if x_now is not None:
            out[1] = (x_now @ coef1).ravel()

    X_lp, Y_lp, _ = build_lp_matrices(factors_train, macro_train, p=p, q=q, h=h, macro_cols=macro_cols)
    if X_lp is not None and len(X_lp) >= p + q * len(macro_cols) + 2:
        x_now_m = _build_forecast_row(factors_train, macro_train, p, q, macro_cols)
        if x_now_m is not None:
            coef2 = fit_lp(X_lp, Y_lp)
            out[2] = (x_now_m @ coef2).ravel()
            T = len(X_lp)
            w = lambda_ew ** np.arange(T - 1, -1, -1)
            coef3 = fit_lp(X_lp, Y_lp, weights=w)
            out[3] = (x_now_m @ coef3).ravel()

    return out


def build_preperiod_segment(factors, macro, macro_cols, p, q, lambda_ew):
    origins = factors.loc[RESIDUAL_POOL_BURNIN_START:TEST_START].index
    origins = origins[origins < pd.Timestamp(TEST_START)]
    print(f"Pre-test segment: {len(origins)} origins, {origins.min().date()} -> {origins.max().date()}")

    records = []
    n = len(origins)
    for i, s in enumerate(origins):
        if i % 100 == 0:
            print(f"  {i}/{n}...", end=" ", flush=True)
        factors_train = factors.loc[:s]
        macro_train = macro.loc[:s]

        for h in FORECAST_HORIZONS_WEEKS:
            future_idx = factors.index[factors.index > s]
            if len(future_idx) < h:
                continue
            s_ph = future_idx[h - 1]
            actual = factors.loc[s_ph, FACTOR_COLS].values
            fc = forecasts_at_origin(factors_train, macro_train, macro_cols, p, q, lambda_ew, h)
            for m in [1, 2, 3]:
                resid = actual - fc[m]
                records.append({
                    "date": s, "horizon": h, "model": m,
                    "resid_beta0": resid[0], "resid_beta1": resid[1], "resid_beta2": resid[2],
                    "realized_date": s_ph, "segment": "preperiod_fresh",
                })
    print("done.")
    return pd.DataFrame(records)


def build_official_segment(factors):
    ff = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "factor_forecasts.parquet"))
    ff = ff[ff["model"].isin([1, 2, 3])].copy()
    actual = factors[FACTOR_COLS].rename(columns={c: f"actual_{c}" for c in FACTOR_COLS})
    ff = ff.merge(actual, left_on="t_plus_h", right_index=True, how="left")
    ff = ff.dropna(subset=[f"actual_{c}" for c in FACTOR_COLS])

    ff["resid_beta0"] = ff["actual_beta0"] - ff["beta0_hat"]
    ff["resid_beta1"] = ff["actual_beta1"] - ff["beta1_hat"]
    ff["resid_beta2"] = ff["actual_beta2"] - ff["beta2_hat"]
    ff["segment"] = "official_reused"
    out = ff.rename(columns={"t_plus_h": "realized_date"})[
        ["date", "horizon", "model", "resid_beta0", "resid_beta1", "resid_beta2",
         "realized_date", "segment"]]
    print(f"Official-test segment (reused from factor_forecasts.parquet, no new fitting): "
          f"{len(out)} rows, {out['date'].nunique()} dates, "
          f"{out['date'].min().date()} -> {out['date'].max().date()}")
    return out


def main():
    print("=" * 70)
    print("Walk-forward residual pool: models 1-3, burn-in "
          f"{RESIDUAL_POOL_BURNIN_START} (config.py)")
    print("=" * 70)

    factors = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet"))[FACTOR_COLS]
    macro_changes = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro_changes.parquet"))
    macro_levels = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro.parquet"))
    macro = pd.concat([macro_changes[MACRO_CHANGE_COLS], macro_levels[MACRO_LEVEL_COLS]], axis=1)
    macro_cols = MACRO_VAR_SPEC
    lag_sel = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "lag_selection.parquet"))
    p = int(lag_sel["recommended"].iloc[0])
    q = 1
    opt_path = os.path.join(PROCESSED_DATA_DIR, "lambda_ew_optimal.parquet")
    lambda_ew = float(pd.read_parquet(opt_path)["lambda_ew_optimal"].iloc[0]) if os.path.exists(opt_path) else LAMBDA_EW
    print(f"p={p}, q={q}, lambda_ew={lambda_ew:.4f}\n")

    preperiod = build_preperiod_segment(factors, macro, macro_cols, p, q, lambda_ew)
    official = build_official_segment(factors)

    combined = pd.concat([preperiod, official], ignore_index=True)
    out_path = os.path.join(PROCESSED_DATA_DIR, "walkforward_residuals.parquet")
    combined.to_parquet(out_path)
    print(f"\nSaved: {out_path}  ({len(combined)} rows)")
    for m in [1, 2, 3]:
        sub = combined[combined.model == m]
        print(f"  Model {m}: {sub['date'].nunique()} distinct origins, "
              f"{sub['date'].min().date()} -> {sub['date'].max().date()}")

    print("\nDone. 07/13/14/15/09 and their outputs were not modified.")


if __name__ == "__main__":
    main()
