"""
09d_deal_platt_rolling.py - Genuinely point-in-time-honest Platt calibration
for the deal-specific, roll-down/volatility-threshold probabilities (09c),
mirroring 14b_platt_calibration_rolling.py's fix for the general case rather
than assuming by analogy that the same conclusion carries over.

WHY THIS SCRIPT: 09c's own Platt fit is still the in-sample POOLED
methodology (fit on the whole stable period, apply back across it) --
exactly the construction 14b showed does not survive honest testing in the
general case. This checks that directly for the deal-specific case instead
of assuming it.

METHOD, same as 14b's extended-history version:
  (1) Extend the deal's raw probability series back to
      RESIDUAL_POOL_BURNIN_START (2012-01-01, config.py), computing genuine
      walk-forward forecasts at each pre-period origin (models 1-3 reuse
      13e's forecasts_at_origin logic; model 0 needs no fitting).
  (2) Fit Platt at each OFFICIAL test date t using only pool entries whose
      OUTCOME is already known as of t (realized_date <= t) -- the same
      embargo rule used throughout this fix.
  (3) No maturity-pooling option here (the deal has one specific tenor,
      unlike the six-maturity constant-maturity grid), so this is the
      single-cell analogue of 14b's v2, not v1 -- extended history but no
      cross-maturity pooling to inflate N further.

STANDALONE / NON-DESTRUCTIVE: does not touch 09/09a/09c/13/14 or their
outputs.

Inputs : data/processed/walkforward_residuals.parquet   (13e, horizon=13 rows)
         data/processed/ns_factors.parquet
         data/processed/curves.parquet
         data/processed/lme_copper_cash.parquet
         data/processed/deal_rolldown_probabilities_walkforward.parquet (09c, official period)
         data/processed/deal_rolldown_calibrated_walkforward.parquet (09c, for BSS comparison)
Outputs: data/processed/deal_platt_rolling_comparison.parquet

Run: python code/09d_deal_platt_rolling.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import PROCESSED_DATA_DIR, LAMBDA_NS, TEST_START, RESIDUAL_POOL_BURNIN_START
from utils import build_lp_matrices, fit_lp, trailing_vol_reference

MODEL_NAMES = {0: "Random Walk", 1: "AR-Direct", 2: "LP+Macro", 3: "LP+Macro+EW"}
DEAL_MATURITY = 3
DEAL_HORIZON = 13
VOL_LOOKBACK_WEEKS = 52
WEEKS_PER_MONTH = 52 / 12
TAU_REM = DEAL_MATURITY - DEAL_HORIZON / WEEKS_PER_MONTH
N_SAMPLES = 1000
EPS = 1e-4
TARIFF_SHOCK_START = pd.Timestamp("2025-04-11")
FACTOR_COLS = ["beta0", "beta1", "beta2"]
MIN_N_ROLLING = 20


def ns_loadings_vec(tau_scalar, lambda_ns):
    lt = tau_scalar / lambda_ns
    if lt < 1e-8:
        return np.array([1.0, 1.0, 0.0])
    s_load = (1 - np.exp(-lt)) / lt
    return np.array([1.0, s_load, s_load - np.exp(-lt)])


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


# ── Duplicated from 13e_walkforward_residual_pool.py (models 1-3 branch) ──
def _build_forecast_row(factors_train, macro_train, p, q, macro_cols):
    combined = pd.concat(
        [factors_train[FACTOR_COLS]] + ([macro_train[macro_cols]] if macro_cols else []), axis=1
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


def build_preperiod_deal_probs(factors, macro, macro_cols, p, q, lambda_ew,
                                curves, cash, sigma_series, loadings_rem, wf13, rng):
    """Same roll-down/Cash/sigma mechanics as 09c's compute_for_model, but at
    pre-period origins (2012 -> just before TEST_START), computing each
    origin's forecast fresh rather than reading factor_forecasts.parquet
    (which only has official-period rows)."""
    entry_col = f"m{DEAL_MATURITY:02d}"
    origins = factors.loc[RESIDUAL_POOL_BURNIN_START:TEST_START].index
    origins = origins[origins < pd.Timestamp(TEST_START)]
    print(f"Pre-period deal probabilities: {len(origins)} origins")

    records = []
    for i, s in enumerate(origins):
        if i % 100 == 0:
            print(f"  {i}/{len(origins)}...", end=" ", flush=True)
        if s not in curves.index or entry_col not in curves.columns:
            continue
        p0 = curves.loc[s, entry_col]
        if pd.isna(p0) or p0 <= 0:
            continue
        sigma_t = sigma_series.get(s, np.nan)
        if pd.isna(sigma_t):
            continue

        future_idx = factors.index[factors.index > s]
        if len(future_idx) < DEAL_HORIZON:
            continue
        s_ph = future_idx[DEAL_HORIZON - 1]

        idx_nearest = cash.index.get_indexer([s_ph], method="nearest", tolerance=pd.Timedelta(days=3))[0]
        actual_price = np.nan if idx_nearest == -1 else cash.loc[cash.index[idx_nearest]]

        factors_train = factors.loc[:s]
        macro_train = macro.loc[:s]
        fc = forecasts_at_origin(factors_train, macro_train, macro_cols, p, q, lambda_ew, DEAL_HORIZON)
        # Model 0: no fitting -- last observed factors
        fc[0] = factors_train[FACTOR_COLS].iloc[-1].values

        for m in [0, 1, 2, 3]:
            if m == 0:
                eligible = None  # built separately below from realised differences
                pool = None
            else:
                pool_m = wf13[wf13.model == m]
                eligible = pool_m[pool_m["realized_date"] <= s]
                if len(eligible) < 5:
                    continue
                pool = eligible[["resid_beta0", "resid_beta1", "resid_beta2"]].values
            if m == 0:
                combined = factors[FACTOR_COLS].dropna()
                diffs = (combined.shift(-DEAL_HORIZON) - combined).dropna()
                diffs = diffs[diffs.index <= s - pd.Timedelta(weeks=DEAL_HORIZON)]
                if len(diffs) < 5:
                    continue
                pool = diffs.values

            idx = rng.integers(0, len(pool), size=N_SAMPLES)
            sim_betas = fc[m][None, :] + pool[idx]
            sim_prices = np.exp(sim_betas @ loadings_rem)

            up_barrier, down_barrier = p0 * (1 + sigma_t), p0 * (1 - sigma_t)
            p_up = float(np.mean(sim_prices > up_barrier))
            p_down = float(np.mean(sim_prices < down_barrier))
            if np.isnan(actual_price):
                act_up, act_down = np.nan, np.nan
            else:
                act_up = 1 if actual_price > up_barrier else 0
                act_down = 1 if actual_price < down_barrier else 0

            records.append({"model": m, "date": s, "p_up": round(p_up, 4), "p_down": round(p_down, 4),
                             "actual_up": act_up, "actual_down": act_down, "realized_date": s_ph})
    print("done.")
    return pd.DataFrame(records)


def rolling_platt_for_model(combined, model_id, min_n=MIN_N_ROLLING):
    """Genuinely point-in-time-honest Platt: at each official test date t,
    fit on eligible history only (realized_date <= t, date < shock start)."""
    sub = combined[combined.model == model_id].sort_values("date").reset_index(drop=True)
    results = {}
    for direction in ["up", "down"]:
        p_col, act_col = f"p_{direction}", f"actual_{direction}"
        valid_mask = sub[[p_col, act_col]].notna().all(axis=1) & sub["date"].lt(TARIFF_SHOCK_START)
        eligible_pool = sub[valid_mask]
        rows = []
        for _, row in sub[sub["date"] >= pd.Timestamp(TEST_START)].iterrows():
            t = row["date"]
            eligible = eligible_pool[eligible_pool["realized_date"] <= t]
            if len(eligible) < min_n or t >= TARIFF_SHOCK_START:
                rows.append({"date": t, "p_raw": row[p_col], "p_cal": np.nan,
                             "actual": row[act_col], "in_domain": False, "n_used": len(eligible)})
                continue
            a, b = fit_platt(eligible[p_col].values, eligible[act_col].values.astype(float))
            p_cal = sigmoid(a * logit(np.array([row[p_col]]))[0] + b) if not np.isnan(row[p_col]) else np.nan
            rows.append({"date": t, "p_raw": row[p_col], "p_cal": p_cal, "actual": row[act_col],
                         "in_domain": True, "n_used": len(eligible), "a": a, "b": b})
        results[direction] = pd.DataFrame(rows)
    return results


def main():
    print("=" * 78)
    print("Point-in-time-honest Platt calibration, deal-specific (roll-down/vol-threshold)")
    print("=" * 78)

    factors = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet"))[FACTOR_COLS]
    from config import MACRO_CHANGE_COLS, MACRO_LEVEL_COLS, MACRO_VAR_SPEC, LAMBDA_EW
    macro_changes = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro_changes.parquet"))
    macro_levels = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro.parquet"))
    macro = pd.concat([macro_changes[MACRO_CHANGE_COLS], macro_levels[MACRO_LEVEL_COLS]], axis=1)
    macro_cols = MACRO_VAR_SPEC
    lag_sel = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "lag_selection.parquet"))
    p = int(lag_sel["recommended"].iloc[0])
    q = 1
    opt_path = os.path.join(PROCESSED_DATA_DIR, "lambda_ew_optimal.parquet")
    lambda_ew = float(pd.read_parquet(opt_path)["lambda_ew_optimal"].iloc[0]) if os.path.exists(opt_path) else LAMBDA_EW

    curves = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "curves.parquet"))
    curves.index = pd.to_datetime(curves.index)
    cash = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "lme_copper_cash.parquet"))["cash"]
    cash.index = pd.to_datetime(cash.index)
    sigma_series = trailing_vol_reference(cash, horizon_weeks=DEAL_HORIZON, lookback_weeks=VOL_LOOKBACK_WEEKS)
    loadings_rem = ns_loadings_vec(TAU_REM, LAMBDA_NS)
    rng = np.random.default_rng(42)

    wf = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "walkforward_residuals.parquet"))
    wf13 = wf[wf.horizon == DEAL_HORIZON]

    preperiod = build_preperiod_deal_probs(factors, macro, macro_cols, p, q, lambda_ew,
                                            curves, cash, sigma_series, loadings_rem, wf13, rng)

    official = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "deal_rolldown_probabilities_walkforward.parquet"))
    official["realized_date"] = official["date"] + pd.to_timedelta(DEAL_HORIZON, unit="W")
    official = official[["model", "date", "p_up", "p_down", "actual_up", "actual_down", "realized_date"]]

    combined = pd.concat([preperiod, official], ignore_index=True)
    print(f"\nCombined series: {len(combined)} rows, "
          f"{combined['date'].min().date()} -> {combined['date'].max().date()}")
    combined.to_parquet(os.path.join(PROCESSED_DATA_DIR, "deal_platt_rolling_combined_input.parquet"))
    print("Saved: deal_platt_rolling_combined_input.parquet (verification: the exact series the rolling fit ran on)")

    orig_calib = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "deal_rolldown_calibrated_walkforward.parquet"))

    print("\n" + "-" * 100)
    print("BSS COMPARISON - raw vs in-sample pooled Platt (09c) vs point-in-time-honest rolling Platt")
    print("-" * 100)
    comparison_rows = []
    for model_id in [0, 1, 2, 3]:
        rolling = rolling_platt_for_model(combined, model_id)
        for direction in ["up", "down"]:
            df = rolling[direction]
            in_dom = df[df["in_domain"]].dropna(subset=["p_raw", "p_cal", "actual"])
            if len(in_dom) < 5:
                print(f"{MODEL_NAMES[model_id]:<14} {direction:>4}: insufficient in-domain dates for rolling test")
                continue
            base_rate = float(in_dom["actual"].mean())
            bs_raw = brier(in_dom["p_raw"].values, in_dom["actual"].values.astype(float))
            bs_roll = brier(in_dom["p_cal"].values, in_dom["actual"].values.astype(float))
            bss_raw = brier_skill(bs_raw, base_rate)
            bss_roll = brier_skill(bs_roll, base_rate)

            pooled_row = orig_calib[(orig_calib.model == model_id)]
            pooled_row = pooled_row[pooled_row["in_calibration_domain"] & pooled_row["date"].lt(TARIFF_SHOCK_START)]
            act_col = f"actual_{direction}"
            pcal_col = f"p_{direction}_cal"
            praw_col = f"p_{direction}"
            valid_pooled = pooled_row.dropna(subset=[pcal_col, act_col])
            bss_pooled = np.nan
            if len(valid_pooled) >= 5:
                br_p = float(valid_pooled[act_col].mean())
                bs_p = brier(valid_pooled[pcal_col].values, valid_pooled[act_col].values.astype(float))
                bss_pooled = brier_skill(bs_p, br_p)

            print(f"{MODEL_NAMES[model_id]:<14} {direction:>4}: raw BSS={bss_raw:+.4f}   "
                  f"pooled-Platt BSS={bss_pooled:+.4f}   rolling-honest-Platt BSS={bss_roll:+.4f}  "
                  f"(N_in_domain={len(in_dom)})")
            comparison_rows.append({"model": model_id, "direction": direction, "bss_raw": bss_raw,
                                     "bss_pooled_platt": bss_pooled, "bss_rolling_platt": bss_roll,
                                     "n_in_domain": len(in_dom)})

            if model_id == 1 and direction == "up":
                df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "deal_platt_rolling_ar_direct_up_trajectory.parquet"))
                print("  Saved: deal_platt_rolling_ar_direct_up_trajectory.parquet (verification: full per-date a/b/n_used)")

    pd.DataFrame(comparison_rows).to_parquet(
        os.path.join(PROCESSED_DATA_DIR, "deal_platt_rolling_comparison.parquet"))
    print("\nSaved: deal_platt_rolling_comparison.parquet")
    print("\nDone. 09/09a/09c/13/14 and their outputs were not modified.")


if __name__ == "__main__":
    main()
