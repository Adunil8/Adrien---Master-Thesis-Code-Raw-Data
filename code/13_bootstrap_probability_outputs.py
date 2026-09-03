"""
13_bootstrap_probability_outputs.py - Full-grid parallel probability pipeline
using empirical bootstrap Monte Carlo sampling instead of Gaussian.

STANDALONE / NON-DESTRUCTIVE (see 12_bootstrap_vs_gaussian_test.py for the
head-to-head motivation and the primary-grid comparison). This script does
NOT touch 08_probability_outputs.py or any of its outputs. It reproduces
08's full-grid probability computation (all 4 models x 6 horizons x 6
maturities x 4 thresholds x 2 directions) but draws Monte Carlo curves from
the empirical LP residual pool (bootstrap) rather than N(mu_h, Sigma_h).

Outputs (separate names - production files untouched):
    data/processed/probability_outputs_bootstrap.parquet
    data/processed/brier_scores_bootstrap.parquet
    data/processed/rolling_lp_residuals.pkl   (residual pools, reused by
                                                 14_platt_calibration.py)

Run: python code/13_bootstrap_probability_outputs.py   (~2-4 minutes)
"""

import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import (
    MATURITIES_MONTHS, LAMBDA_NS, LAMBDA_EW,
    FORECAST_HORIZONS_WEEKS, MARGIN_THRESHOLDS,
    PROCESSED_DATA_DIR,
    MACRO_CHANGE_COLS, MACRO_LEVEL_COLS, MACRO_VAR_SPEC,
)
from utils import ns_curve, build_lp_matrices, fit_lp

N_SAMPLES = 1000


def sigma_and_resid_for_window(factors_w, macro_w, macro_cols, p, q, lambda_ew, horizons):
    """Same construction as 08_probability_outputs._sigma_from_window, but also
    retains the raw residual vectors (needed for bootstrap sampling).

    Also returns resid_weights: for Model 3 (LP+Macro+EW) this is the SAME
    per-observation weight vector w = lambda_ew^(T-1-t) used to fit the point
    forecast (coef_ew). Passing it through to the bootstrap sampling step
    keeps the point estimate and the uncertainty band internally consistent
    -- both then reflect the same recency-weighted view of history, rather
    than a recency-weighted mean built on top of a uniformly-resampled
    (i.e. full 17-year, equally-weighted) error distribution. Models 0-2 are
    not fit by WLS, so every historical residual is equally representative
    of the window used to compute their point forecast: resid_weights is
    None for those, and sample_curves_bootstrap falls back to uniform
    resampling, which is the correct choice for an unweighted model.
    """
    factor_cols = ["beta0", "beta1", "beta2"]
    resid = {}
    resid_weights = {}

    for h in horizons:
        combined_f = factors_w[factor_cols].dropna()
        Y_rw = combined_f.shift(-h).dropna()
        X_rw = combined_f.loc[Y_rw.index]
        resid[(0, h)] = Y_rw.values - X_rw.values
        resid_weights[(0, h)] = None

        X_ar, Y_ar, _ = build_lp_matrices(factors_w, macro_w, p, 0, h, macro_cols=None)
        X_lp, Y_lp, _ = build_lp_matrices(factors_w, macro_w, p, q, h, macro_cols=macro_cols)

        for mid, X, Y in [(1, X_ar, Y_ar), (2, X_lp, Y_lp)]:
            if X is None or len(X) < 10:
                resid[(mid, h)] = np.zeros((1, 3))
            else:
                coef = fit_lp(X, Y)
                resid[(mid, h)] = Y - X @ coef
            resid_weights[(mid, h)] = None

        if X_lp is not None and len(X_lp) >= 10:
            T_w = len(X_lp)
            w = lambda_ew ** np.arange(T_w - 1, -1, -1)
            coef_ew = fit_lp(X_lp, Y_lp, weights=w)
            resid[(3, h)] = Y_lp - X_lp @ coef_ew
            resid_weights[(3, h)] = w
        else:
            resid[(3, h)] = resid.get((2, h), np.zeros((1, 3)))
            resid_weights[(3, h)] = None

    return resid, resid_weights


def sample_curves_bootstrap(mu_beta, resid_pool, tau, lambda_ns, n_samples, rng, resid_weights=None):
    """Bootstrap-resample residuals around mu_beta to build a Monte Carlo
    distribution of forecasted curves. resid_weights, when given, must align
    1:1 with resid_pool's rows and should be the same weights used to
    estimate mu_beta (see sigma_and_resid_for_window docstring) -- this is
    what keeps a weighted point forecast and its uncertainty band mutually
    consistent. None (the default, and always the case for Models 0-2)
    resamples uniformly, correct for an unweighted point forecast."""
    n_pool = len(resid_pool)
    if n_pool < 5:
        sigma = np.cov(resid_pool.T) if n_pool > 3 else np.eye(3) * 1e-4
        try:
            betas = rng.multivariate_normal(mu_beta, sigma, size=n_samples)
        except np.linalg.LinAlgError:
            betas = rng.multivariate_normal(mu_beta, np.diag(np.diag(sigma)), size=n_samples)
    else:
        if resid_weights is not None and len(resid_weights) == n_pool:
            probs = resid_weights / resid_weights.sum()
            idx = rng.choice(n_pool, size=n_samples, p=probs)
        else:
            idx = rng.integers(0, n_pool, size=n_samples)
        betas = mu_beta[None, :] + resid_pool[idx]
    return np.exp(np.array([ns_curve(b, tau, lambda_ns) for b in betas]))


def compute_rolling_resid(factors, macro, macro_cols, p, q, lambda_ew, horizons, test_dates):
    rolling_resid = {}
    rolling_weights = {}
    n = len(test_dates)
    print(f"Computing rolling residual pools for {n} test dates x {len(horizons)} horizons x 4 models...")
    for i, t in enumerate(test_dates):
        if i % 30 == 0:
            print(f"  {i}/{n}...", end=" ", flush=True)
        t = pd.Timestamp(t)
        resid_t, weights_t = sigma_and_resid_for_window(
            factors.loc[:t], macro.loc[:t], macro_cols, p, q, lambda_ew, horizons)
        for (mid, h), arr in resid_t.items():
            rolling_resid[(mid, h, t)] = arr
        for (mid, h), w in weights_t.items():
            rolling_weights[(mid, h, t)] = w
    print("done.")
    return rolling_resid, rolling_weights


def get_resid(rolling_resid, model_id, h, date):
    return rolling_resid.get((model_id, h, pd.Timestamp(date)), np.zeros((1, 3)))


def get_resid_weights(rolling_weights, model_id, h, date):
    return rolling_weights.get((model_id, h, pd.Timestamp(date)), None)


def compute_probability_outputs_bootstrap(factor_forecasts_df, copper_df, rolling_resid, rolling_weights,
                                           tau, lambda_ns, thresholds, n_samples=N_SAMPLES):
    # Each model gets its own independent RNG stream. A single generator
    # advanced across all four models in sequence would let Model 3's
    # weighted resampling (rng.choice with p=probs, vs the other three
    # models' rng.integers) consume a different number of draws from the
    # shared stream each iteration, silently shifting the random numbers
    # Models 0-2 receive on the NEXT (date, horizon) iteration, purely
    # because of an implementation detail belonging to a different model.
    # A model's reported numbers should be a deterministic function of its
    # own data and code alone. Spawning one independent child generator
    # per model_id from a single seeded parent removes that coupling
    # entirely.
    rngs = dict(zip([0, 1, 2, 3], np.random.default_rng(42).spawn(4)))
    records = []
    dates = factor_forecasts_df["date"].unique()
    n_total = len(dates)
    print(f"\nComputing bootstrap probability outputs for {n_total} test dates...")

    for i, date_t in enumerate(sorted(dates)):
        if i % 30 == 0:
            print(f"  {i}/{n_total}...", end=" ", flush=True)
        if date_t not in copper_df.index:
            continue
        current_prices = copper_df.loc[date_t].values

        for h in FORECAST_HORIZONS_WEEKS:
            mask = (factor_forecasts_df["date"] == date_t) & (factor_forecasts_df["horizon"] == h)
            sub = factor_forecasts_df[mask]
            t_ph_candidates = sub["t_plus_h"].values
            if len(t_ph_candidates) == 0:
                continue
            t_ph = pd.Timestamp(t_ph_candidates[0])
            if t_ph not in copper_df.index:
                continue
            realized_prices = copper_df.loc[t_ph].values

            for model_id in [0, 1, 2, 3]:
                sub_m = sub[sub["model"] == model_id]
                if sub_m.empty:
                    continue
                mu_beta = sub_m[["beta0_hat", "beta1_hat", "beta2_hat"]].values[0]
                resid_h = get_resid(rolling_resid, model_id, h, date_t)
                weights_h = get_resid_weights(rolling_weights, model_id, h, date_t)
                curves = sample_curves_bootstrap(mu_beta, resid_h, tau, lambda_ns, n_samples, rngs[model_id],
                                                  resid_weights=weights_h)

                for j, mat in enumerate(MATURITIES_MONTHS):
                    if j >= len(current_prices) or np.isnan(current_prices[j]):
                        continue
                    p0 = current_prices[j]
                    pr = realized_prices[j] if j < len(realized_prices) else np.nan

                    for k in thresholds:
                        up_barrier   = p0 * (1 + k)
                        down_barrier = p0 * (1 - k)
                        sample_col = curves[:, j]
                        p_up   = float(np.mean(sample_col > up_barrier))
                        p_down = float(np.mean(sample_col < down_barrier))
                        act_up   = (1 if pr > up_barrier else 0) if not np.isnan(pr) else np.nan
                        act_down = (1 if pr < down_barrier else 0) if not np.isnan(pr) else np.nan
                        records.append({
                            "date": date_t, "horizon": h, "model": model_id, "maturity": mat,
                            "threshold": k, "p_up": round(p_up, 4), "p_down": round(p_down, 4),
                            "actual_up": act_up, "actual_down": act_down,
                        })
    print("done.")
    return pd.DataFrame(records)


def compute_brier_scores(prob_df):
    rows = []
    for (model, h, mat, k), grp in prob_df.groupby(["model", "horizon", "maturity", "threshold"]):
        for direction in ["up", "down"]:
            p_col, act_col = f"p_{direction}", f"actual_{direction}"
            valid = grp[[p_col, act_col]].dropna()
            if len(valid) < 5:
                continue
            bs = float(np.mean((valid[p_col] - valid[act_col]) ** 2))
            base_rate = float(valid[act_col].mean())
            bs_clim = float(base_rate * (1 - base_rate))
            bss = 1 - bs / bs_clim if bs_clim > 0 else np.nan
            rows.append({
                "model": model, "horizon": h, "maturity": mat, "threshold": k,
                "direction": direction, "brier_score": round(bs, 5), "brier_skill": round(bss, 4),
                "base_rate": round(base_rate, 4), "N": len(valid),
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("Full-grid bootstrap probability pipeline (standalone, non-destructive)")
    print("=" * 70)

    factors = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet"))[["beta0", "beta1", "beta2"]]
    copper  = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "curves.parquet"))
    macro_changes = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro_changes.parquet"))
    macro_levels  = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro.parquet"))
    macro = pd.concat([macro_changes[MACRO_CHANGE_COLS], macro_levels[MACRO_LEVEL_COLS]], axis=1)
    macro_cols = MACRO_VAR_SPEC

    factor_forecasts = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "factor_forecasts.parquet"))
    lag_sel = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "lag_selection.parquet"))
    opt_path = os.path.join(PROCESSED_DATA_DIR, "lambda_ew_optimal.parquet")
    lambda_ew = float(pd.read_parquet(opt_path)["lambda_ew_optimal"].iloc[0]) if os.path.exists(opt_path) else LAMBDA_EW
    p = int(lag_sel["recommended"].iloc[0])
    q = 1
    tau = np.array(MATURITIES_MONTHS, dtype=float)

    print(f"p={p}, lambda_ew={lambda_ew:.4f}")

    test_dates = sorted(factor_forecasts["date"].unique())
    resid_path = os.path.join(PROCESSED_DATA_DIR, "rolling_lp_residuals.pkl")
    rolling_resid, rolling_weights = compute_rolling_resid(
        factors, macro, macro_cols, p, q, lambda_ew, FORECAST_HORIZONS_WEEKS, test_dates)
    with open(resid_path, "wb") as f:
        pickle.dump({"resid": rolling_resid, "weights": rolling_weights}, f)
    print(f"Saved: {resid_path}")

    # ── Forecast-error diagnostics, all four models (Ljung-Box, ARCH, JB) ──
    # main.qmd's Section 4.3 needs the full E1/E2/E3 battery on the four
    # models (Random Walk, AR-Direct, LP+Macro, LP+Macro+EW) that actually
    # produce every forecast and probability output in this thesis, not
    # just on the separate diagnostic VAR+macro+EW (lambda=0.97). All four
    # are checked here: Random Walk's and LP+Macro's forecast errors sit in
    # the same residual pool and are used the same way downstream (e.g.
    # Random Walk's own bootstrapped probabilities, Chapter 5), so there is
    # no reason to check only a subset. Random Walk has no fitted coefficients, so "residual" here
    # means its raw forecast error, F(t+h)-F(t); the same three tests still
    # ask a meaningful question of it (is that error serially correlated,
    # volatility-clustered, non-normal), since that is exactly what its own
    # bootstrap probability construction assumes away by resampling rather
    # than using a closed form. VIF is a property of the shared regressor
    # set alone, so the VAR's E4 number is unaffected either way, but serial
    # correlation and ARCH are properties of a specific model's own errors,
    # and these differ from the VAR's, most obviously for serial correlation:
    # LP fits each horizon directly (Jorda 2005), so an h=13 or h=26-week
    # regression uses heavily overlapping windows between consecutive dates,
    # which mechanically induces serial correlation a VAR's 1-step residuals
    # never faces. Reporting only the VAR's numbers would understate this for
    # exactly the models the thesis's conclusions rest on. Checked here
    # directly on the residual pool the bootstrap itself draws from, at the
    # two primary horizons, using each (model, horizon)'s most mature
    # snapshot (latest available date, largest in-sample window).
    print("\n" + "-" * 65)
    print("Forecast-error diagnostics, all four models (LJB, ARCH, JB) - not the Section 4.3 VAR")
    print("-" * 65)
    from scipy.stats import jarque_bera, chi2
    from scipy.special import gammaln
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

    def chi2_logsf_stable(stat, df):
        """Natural-log survival function for chi2(df) at `stat`, robust to
        the regime where BOTH scipy.stats.chi2.sf and .logsf silently
        underflow to 0.0/-inf. This happens here well before any p-value
        of practical interest is reached: e.g. stat=1900, df=10 already
        breaks scipy's own logsf, while this thesis's Ljung-Box statistics
        at h=13W reach ~4600. Falls back to the standard upper-incomplete-
        -gamma asymptotic tail expansion (Abramowitz & Stegun 6.5.32),
        exact in the limit stat >> df, which is exactly the only regime
        this fallback is ever invoked in. Validated against scipy directly
        on the range where scipy itself still works (matches to machine
        precision up to stat~1400, df=10), before trusting it further into
        the tail scipy cannot reach at all.
        """
        raw = chi2.logsf(stat, df)
        if np.isfinite(raw):
            return raw
        x, s = stat / 2.0, df / 2.0
        series = 1.0 + (s - 1) / x + (s - 1) * (s - 2) / x**2 + (s - 1) * (s - 2) * (s - 3) / x**3
        ln_upper_gamma = (s - 1) * np.log(x) - x + np.log(series)
        return ln_upper_gamma - gammaln(s)

    model_labels_jb = {0: "RandomWalk", 1: "AR-Direct", 2: "LP+Macro", 3: "LP+Macro+EW"}
    factor_labels_jb = ["beta0", "beta1", "beta2"]
    diag_rows = []
    for m in [0, 1, 2, 3]:
        for h in [4, 13]:
            sub_keys = [k for k in rolling_resid if k[0] == m and k[1] == h]
            if not sub_keys:
                continue
            latest = max(sub_keys, key=lambda k: k[2])
            arr = rolling_resid[latest]
            for i, fname in enumerate(factor_labels_jb):
                series = arr[:, i]
                jb_stat, jb_pval = jarque_bera(series)
                jb_log10_pval = chi2_logsf_stable(jb_stat, 2) / np.log(10)
                lb = acorr_ljungbox(series, lags=10, return_df=True)
                # Worst (most significant) lag by direct p-value normally
                # picks this out correctly; once every lag has underflowed
                # to a tied 0.0 (h=13W here), idxmin() would instead just
                # return the first lag arbitrarily, so the selection itself
                # is redone in log space, which never underflows the same way.
                lb_log10_by_lag = pd.Series(
                    {lag: chi2_logsf_stable(row_.lb_stat, lag) / np.log(10)
                     for lag, row_ in lb.iterrows()}
                )
                best_lag = lb_log10_by_lag.idxmin()
                lb_log10_pval = lb_log10_by_lag[best_lag]
                lb_pval = lb.loc[best_lag, "lb_pvalue"]
                try:
                    lm_stat, lm_pval, _, _ = het_arch(series, nlags=5)
                    arch_log10_pval = chi2_logsf_stable(lm_stat, 5) / np.log(10)
                except Exception:
                    lm_pval, arch_log10_pval = np.nan, np.nan
                # NOTE: lb_pval/arch_pval are stored at full float precision,
                # matching jb_pval, not rounded to 4dp. These tests reject at
                # extreme significance (many cells underflow chi2.sf to exact
                # 0.0 at very high df/stat, e.g. h=13W; others are merely tiny,
                # e.g. 1e-46 to 1e-310) and rounding to 4dp silently collapsed
                # every non-underflowed cell to a spurious "0.0000" too, hiding
                # genuine ~250-orders-of-magnitude variation across rows behind
                # what looked like an identical result everywhere (caught via
                # direct comparison against jb_pval's unrounded column, which
                # showed real cell-to-cell variation the other two did not).
                # The *_log10_pval companions cover the further, rarer case
                # where the true p-value underflows even a float64 (~1e-308
                # floor), e.g. Ljung-Box at h=13W reaches ~1e-500 to 1e-990,
                # so the direct float genuinely cannot represent it at all;
                # display code should fall back to these whenever the direct
                # value is exactly 0.0, instead of reporting a bare "0".
                row = {
                    "model": model_labels_jb[m], "horizon_weeks": h, "factor": fname,
                    "n_obs": arr.shape[0], "as_of": latest[2],
                    "jb_stat": round(float(jb_stat), 2), "jb_pval": float(jb_pval),
                    "jb_log10_pval": float(jb_log10_pval),
                    "normal": jb_pval > 0.05,
                    "lb_pval": float(lb_pval), "lb_log10_pval": float(lb_log10_pval),
                    "serial_corr": lb_pval < 0.05,
                    "arch_pval": float(lm_pval) if not np.isnan(lm_pval) else None,
                    "arch_log10_pval": float(arch_log10_pval) if not np.isnan(arch_log10_pval) else None,
                    "arch_effect": (lm_pval < 0.05) if not np.isnan(lm_pval) else None,
                }
                diag_rows.append(row)
                print(f"  {model_labels_jb[m]:12s} h={h:2d}W {fname}: "
                      f"JB p={jb_pval:.4g} (normal={jb_pval > 0.05})  "
                      f"LB p={lb_pval:.4g} (serial_corr={lb_pval < 0.05})  "
                      f"ARCH p={lm_pval:.4g} (effect={lm_pval < 0.05})")
    diag_df = pd.DataFrame(diag_rows)
    diag_path = os.path.join(PROCESSED_DATA_DIR, "lp_residual_diagnostics.parquet")
    diag_df.to_parquet(diag_path)
    print(f"Saved: {diag_path}")

    prob_df = compute_probability_outputs_bootstrap(
        factor_forecasts, copper, rolling_resid, rolling_weights, tau, LAMBDA_NS, MARGIN_THRESHOLDS)
    prob_path = os.path.join(PROCESSED_DATA_DIR, "probability_outputs_bootstrap.parquet")
    prob_df.to_parquet(prob_path)
    print(f"Saved: {prob_path}  ({len(prob_df)} rows)")

    brier_df = compute_brier_scores(prob_df)
    brier_path = os.path.join(PROCESSED_DATA_DIR, "brier_scores_bootstrap.parquet")
    brier_df.to_parquet(brier_path)
    print(f"Saved: {brier_path}")

    # Print same-format summary as 08, for direct comparison
    print("\n" + "-" * 65)
    print("Brier Skill Score (bootstrap) - P(dF > +8%), 3M maturity")
    print("-" * 65)
    model_labels = {0: "RW", 1: "AR-Direct", 2: "LP+Macro", 3: "LP+Macro+EW"}
    for h in [4, 13, 26]:
        sub = brier_df[(brier_df.maturity == 3) & (brier_df.threshold == 0.08) &
                        (brier_df.direction == "up") & (brier_df.horizon == h)].sort_values("model")
        if sub.empty:
            continue
        print(f"\nh={h}W:")
        for _, row in sub.iterrows():
            m = model_labels.get(row["model"], str(row["model"]))
            print(f"  {m:14s} BS={row['brier_score']:.4f}  BSS={row['brier_skill']:+.4f}  N={row['N']}")

    print("\nDone. No production files (08_probability_outputs.py outputs) were modified.")
    print("Next: run code/14_platt_calibration.py to fit calibration on top of this output.")


if __name__ == "__main__":
    main()
