"""
13c_calibration_preperiod.py - Extends the AR-Direct walk-forward bootstrap
probability history back to CALIBRATION_BURNIN_START (2021-01-01, config.py),
two years before the official TEST_START (2023-01-01).

WHY THIS EXISTS: 14b_platt_calibration_rolling.py's point-in-time-correct
Platt fit found the officially reported test period alone gives too little
history to fit stably -- N starts at 0 in January 2023 and only clears a
usable floor after several months, with a documented small-N MLE blow-up
along the way (a >~1000 at one date, see conversation record). This script
generates the SAME kind of genuine, one-step-ahead-honest AR-Direct forecast
that 07_forecast_evaluation.py generates for 2023-2026, just starting two
years earlier, purely so 14b has real pre-2023 history to draw on. It is
otherwise the identical procedure: at each date t, the model is re-estimated
using only data available up to t (expanding window), so every pre-period
record here is exactly as point-in-time-honest as the official ones.

WHAT THIS DOES NOT DO: it does not change TRAIN_END or TEST_START, and no
number in Chapters 4-5 (RMSE, hit rate, AUC) is ever computed over this
window -- it is calibration-fitting history only, never reported as an
out-of-sample result in its own right. It also does not touch the residual-
pool idealisation disclosed in Section 3.8 (the single-fit-per-window
resampling): this script reproduces that SAME construction, unchanged, for
the pre-period, rather than fixing it -- that is a separate, deferred task.

SCOPE: AR-Direct (model 1) only. This is the only model 14b ever calibrates
(Section 4.5's AUC test restricts Platt to it), so generating pre-period
probabilities for Random Walk / LP+Macro / LP+Macro+EW would never be used
downstream -- skipped to keep this script's runtime proportionate. Because
AR-Direct does not use macro variables, macro data is not even loaded here.

DUPLICATED LOGIC, NOT IMPORTED: the AR-Direct forecast step mirrors
07_forecast_evaluation.py's lp_forecasts_at_t/_build_forecast_row (model-1
branch only), and the bootstrap-residual step mirrors
13_bootstrap_probability_outputs.py's sigma_and_resid_for_window/
sample_curves_bootstrap (model-1 branch only). Numbered scripts in this
codebase are standalone entry points that duplicate rather than cross-import
shared logic (see 02c_lambda_classification_validation.py's docstring for
the same choice, made so no script can silently inherit a bug from another).

Inputs : data/processed/ns_factors.parquet
         data/processed/curves.parquet
         data/processed/lag_selection.parquet
Outputs: data/processed/probability_outputs_bootstrap_preperiod.parquet
         (same schema as probability_outputs_bootstrap.parquet, model=1 only,
          dates in [CALIBRATION_BURNIN_START, TEST_START))

Run: python code/13c_calibration_preperiod.py   (~30-60 seconds)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import (
    MATURITIES_MONTHS, LAMBDA_NS, TEST_START, CALIBRATION_BURNIN_START,
    FORECAST_HORIZONS_WEEKS, MARGIN_THRESHOLDS, PROCESSED_DATA_DIR,
)
from utils import ns_curve, build_lp_matrices, fit_lp

N_SAMPLES = 1000
MODEL_ID = 1  # AR-Direct only, see module docstring


def ar_direct_forecast_and_resid(factors_train: pd.DataFrame, p: int, h: int):
    """AR-Direct (model 1) branch of 07_forecast_evaluation.py's
    lp_forecasts_at_t, plus the residual-pool construction from
    13_bootstrap_probability_outputs.py's sigma_and_resid_for_window --
    combined here since this script only ever needs model 1.

    Returns (mu_beta, resid) where resid is the (n, 3) array of in-window
    residuals used for bootstrap resampling. Falls back to mu_beta = last
    observed factors (random-walk) if there is not yet enough history.
    """
    factor_cols = ["beta0", "beta1", "beta2"]
    last_obs = factors_train[factor_cols].iloc[-1].values

    X_ar, Y_ar, _ = build_lp_matrices(factors_train, None, p=p, q=0, h=h, macro_cols=None)
    if X_ar is None or len(X_ar) < p + 2:
        return last_obs, np.zeros((1, 3))

    coef = fit_lp(X_ar, Y_ar)
    resid = Y_ar - X_ar @ coef

    combined = factors_train[factor_cols].dropna()
    ar_block = []
    for lag in range(p):
        ar_block.extend(combined.iloc[-(lag + 1)].values)
    x_now = np.array([[1.0] + ar_block])
    mu_beta = (x_now @ coef).ravel()
    return mu_beta, resid


def sample_curves_bootstrap(mu_beta, resid_pool, tau, lambda_ns, n_samples, rng):
    """Uniform-resampling case only (model 1 is never WLS-fit) -- identical
    to 13_bootstrap_probability_outputs.py's sample_curves_bootstrap with
    resid_weights=None."""
    n_pool = len(resid_pool)
    if n_pool < 5:
        sigma = np.cov(resid_pool.T) if n_pool > 3 else np.eye(3) * 1e-4
        try:
            betas = rng.multivariate_normal(mu_beta, sigma, size=n_samples)
        except np.linalg.LinAlgError:
            betas = rng.multivariate_normal(mu_beta, np.diag(np.diag(sigma)), size=n_samples)
    else:
        idx = rng.integers(0, n_pool, size=n_samples)
        betas = mu_beta[None, :] + resid_pool[idx]
    return np.exp(np.array([ns_curve(b, tau, lambda_ns) for b in betas]))


def main():
    print("=" * 70)
    print("Pre-period (calibration burn-in) AR-Direct bootstrap probabilities")
    print(f"Window: {CALIBRATION_BURNIN_START} -> just before {TEST_START}")
    print("=" * 70)

    factors = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet"))[
        ["beta0", "beta1", "beta2"]]
    copper = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "curves.parquet"))
    lag_sel = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "lag_selection.parquet"))
    p = int(lag_sel["recommended"].iloc[0])
    tau = np.array(MATURITIES_MONTHS, dtype=float)

    preperiod_dates = factors.loc[CALIBRATION_BURNIN_START:TEST_START].index
    preperiod_dates = preperiod_dates[preperiod_dates < pd.Timestamp(TEST_START)]
    print(f"p={p} | {len(preperiod_dates)} pre-period dates | model=AR-Direct only")

    rng = np.random.default_rng(42)  # same seed as 13_bootstrap_probability_outputs.py
    records = []
    n = len(preperiod_dates)

    for i, t in enumerate(preperiod_dates):
        if i % 20 == 0:
            print(f"  {i}/{n}...", end=" ", flush=True)

        factors_train = factors.loc[:t]
        if t not in copper.index:
            continue
        current_prices = copper.loc[t].values

        for h in FORECAST_HORIZONS_WEEKS:
            future_idx = factors.index[factors.index > t]
            if len(future_idx) < h:
                continue  # not enough history AFTER t yet to know the realised outcome
            t_ph = future_idx[h - 1]
            if t_ph not in copper.index:
                continue
            realized_prices = copper.loc[t_ph].values

            mu_beta, resid = ar_direct_forecast_and_resid(factors_train, p, h)
            curves = sample_curves_bootstrap(mu_beta, resid, tau, LAMBDA_NS, N_SAMPLES, rng)

            for j, mat in enumerate(MATURITIES_MONTHS):
                if j >= len(current_prices) or np.isnan(current_prices[j]):
                    continue
                p0 = current_prices[j]
                pr = realized_prices[j] if j < len(realized_prices) else np.nan
                if np.isnan(pr):
                    continue

                for k in MARGIN_THRESHOLDS:
                    up_barrier = p0 * (1 + k)
                    down_barrier = p0 * (1 - k)
                    sample_col = curves[:, j]
                    records.append({
                        "date": t, "horizon": h, "model": MODEL_ID, "maturity": mat,
                        "threshold": k,
                        "p_up": round(float(np.mean(sample_col > up_barrier)), 4),
                        "p_down": round(float(np.mean(sample_col < down_barrier)), 4),
                        "actual_up": 1 if pr > up_barrier else 0,
                        "actual_down": 1 if pr < down_barrier else 0,
                    })
    print("done.")

    out_df = pd.DataFrame(records)
    out_path = os.path.join(PROCESSED_DATA_DIR, "probability_outputs_bootstrap_preperiod.parquet")
    out_df.to_parquet(out_path)
    print(f"Saved: {out_path}  ({len(out_df)} rows, {out_df['date'].nunique()} distinct dates, "
          f"{out_df['date'].min().date()} -> {out_df['date'].max().date()})")


if __name__ == "__main__":
    main()
