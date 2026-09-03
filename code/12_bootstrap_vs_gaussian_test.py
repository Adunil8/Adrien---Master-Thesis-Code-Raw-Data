"""
12_bootstrap_vs_gaussian_test.py - Robustness test: Gaussian vs empirical
bootstrap Monte Carlo sampling for the probability outputs.

STANDALONE / NON-DESTRUCTIVE: this script does NOT modify 08_probability_
outputs.py or any of its saved outputs (data/processed/probability_outputs.
parquet, brier_scores.parquet, rolling_lp_sigma.pkl are all left untouched).
It recomputes its own rolling residual pool independently and writes to
SEPARATELY NAMED files:
    data/processed/probability_outputs_bootstrap.parquet
    data/processed/brier_scores_bootstrap.parquet
    data/processed/gaussian_vs_bootstrap_comparison.parquet

MOTIVATION
----------
Section 4.3 (Jarque-Bera, test E3) already shows LP residuals reject
normality (high excess kurtosis, esp. beta2). 08_probability_outputs.py's
Monte Carlo sampler currently draws from N(mu_h, Sigma_h) - a Gaussian
assumption the thesis's own diagnostics call into question. This script
tests the alternative: draw directly from the empirical (in-sample) LP
residual pool (a nonparametric bootstrap), which has the correct historical
shape (skew, kurtosis) by construction, and compares calibration (Brier
Skill Score, reliability) against the existing Gaussian approach.

This is presented as a robustness check (parallel to the Svensson-vs-NS and
lambda-grid checks already in Appendix IV), not a silent replacement of the
production pipeline. If the bootstrap materially improves calibration, that
becomes the basis for a considered decision - made explicitly, together -
about whether to promote it into 08_probability_outputs.py.

Run: python code/12_bootstrap_vs_gaussian_test.py   (~1-2 minutes)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from config import (
    MATURITIES_MONTHS, LAMBDA_NS,
    FORECAST_HORIZONS_WEEKS, HORIZON_LABELS,
    MARGIN_THRESHOLDS,
    PROCESSED_DATA_DIR,
    MACRO_CHANGE_COLS, MACRO_LEVEL_COLS, MACRO_VAR_SPEC,
)
from utils import ns_curve, build_lp_matrices, fit_lp

N_SAMPLES = 1000
PRIMARY_MODEL = 3        # LP + Macro + EW - the deployed model (see 09_stylised_deal.py)
PRIMARY_MATURITY = 3
PRIMARY_THRESHOLD = 0.08
PRIMARY_HORIZONS = [4, 13, 26]
MODEL_LABELS = {0: "Random Walk", 1: "AR-Direct", 2: "LP + Macro", 3: "LP + Macro + EW"}


# ── Rolling residual pool (self-contained copy - mirrors 08's Sigma_h logic,
#    but retains the raw residual vectors instead of collapsing to covariance) ──

def sigma_and_resid_for_window(factors_w, macro_w, macro_cols, p, q, lambda_ew, horizons):
    factor_cols = ["beta0", "beta1", "beta2"]
    sigma, resid = {}, {}

    for h in horizons:
        combined_f = factors_w[factor_cols].dropna()
        Y_rw = combined_f.shift(-h).dropna()
        X_rw = combined_f.loc[Y_rw.index]
        res0 = Y_rw.values - X_rw.values
        sigma[(0, h)] = np.cov(res0.T) if len(res0) > 3 else np.eye(3) * 1e-4
        resid[(0, h)] = res0 if len(res0) > 0 else np.zeros((1, 3))

        X_ar, Y_ar, _ = build_lp_matrices(factors_w, macro_w, p, 0, h, macro_cols=None)
        X_lp, Y_lp, _ = build_lp_matrices(factors_w, macro_w, p, q, h, macro_cols=macro_cols)

        for mid, X, Y in [(1, X_ar, Y_ar), (2, X_lp, Y_lp)]:
            if X is None or len(X) < 10:
                sigma[(mid, h)] = np.eye(3) * 1e-4
                resid[(mid, h)] = np.zeros((1, 3))
                continue
            coef = fit_lp(X, Y)
            res  = Y - X @ coef
            sigma[(mid, h)] = np.cov(res.T) if len(res) > 3 else np.eye(3) * 1e-4
            resid[(mid, h)] = res

        if X_lp is not None and len(X_lp) >= 10:
            T_w = len(X_lp)
            w = lambda_ew ** np.arange(T_w - 1, -1, -1)
            coef_ew = fit_lp(X_lp, Y_lp, weights=w)
            res_ew = Y_lp - X_lp @ coef_ew
            sigma[(3, h)] = np.cov(res_ew.T) if len(res_ew) > 3 else np.eye(3) * 1e-4
            resid[(3, h)] = res_ew
        else:
            sigma[(3, h)] = sigma.get((2, h), np.eye(3) * 1e-4)
            resid[(3, h)] = resid.get((2, h), np.zeros((1, 3)))

    return sigma, resid


def sample_curves_gaussian(mu_beta, sigma_h, tau, lambda_ns, n_samples, rng):
    try:
        betas = rng.multivariate_normal(mu_beta, sigma_h, size=n_samples)
    except np.linalg.LinAlgError:
        betas = rng.multivariate_normal(mu_beta, np.diag(np.diag(sigma_h)), size=n_samples)
    return np.exp(np.array([ns_curve(b, tau, lambda_ns) for b in betas]))


def sample_curves_bootstrap(mu_beta, resid_pool, tau, lambda_ns, n_samples, rng):
    """
    Nonparametric bootstrap: draw n_samples residual VECTORS (with replacement,
    preserving cross-factor correlation) from the actual in-sample LP residual
    pool, add to the point forecast mu_beta. No distributional assumption.
    """
    n_pool = len(resid_pool)
    if n_pool < 5:
        return sample_curves_gaussian(mu_beta, np.cov(resid_pool.T) if n_pool > 3
                                       else np.eye(3) * 1e-4, tau, lambda_ns, n_samples, rng)
    idx = rng.integers(0, n_pool, size=n_samples)
    betas = mu_beta[None, :] + resid_pool[idx]
    return np.exp(np.array([ns_curve(b, tau, lambda_ns) for b in betas]))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def brier_skill(bs, base_rate):
    bs_clim = base_rate * (1 - base_rate)
    return 1 - bs / bs_clim if bs_clim > 0 else np.nan


def main():
    print("=" * 70)
    print("Robustness test - Gaussian vs empirical bootstrap Monte Carlo sampling")
    print("(standalone - production files untouched)")
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
    from config import LAMBDA_EW
    lambda_ew = float(pd.read_parquet(opt_path)["lambda_ew_optimal"].iloc[0]) if os.path.exists(opt_path) else LAMBDA_EW
    p = int(lag_sel["recommended"].iloc[0])
    q = 1
    tau = np.array(MATURITIES_MONTHS, dtype=float)

    test_dates = sorted(factor_forecasts["date"].unique())
    n_td = len(test_dates)
    print(f"p={p}, lambda_ew={lambda_ew:.4f}, {n_td} test dates\n")

    rng = np.random.default_rng(42)
    records = []
    print("Computing rolling residual pools + Monte Carlo draws "
          f"(both methods, {n_td} dates)...")

    for i, date_t in enumerate(test_dates):
        if i % 30 == 0:
            print(f"  {i}/{n_td}...", end=" ", flush=True)
        date_t = pd.Timestamp(date_t)
        if date_t not in copper.index:
            continue
        current_prices = copper.loc[date_t].values

        sigma_t, resid_t = sigma_and_resid_for_window(
            factors.loc[:date_t], macro.loc[:date_t], macro_cols, p, q, lambda_ew,
            PRIMARY_HORIZONS,
        )

        for h in PRIMARY_HORIZONS:
            mask = (factor_forecasts["date"] == date_t) & (factor_forecasts["horizon"] == h) \
                   & (factor_forecasts["model"] == PRIMARY_MODEL)
            sub = factor_forecasts[mask]
            if sub.empty:
                continue
            t_ph_vals = sub["t_plus_h"].values
            if len(t_ph_vals) == 0:
                continue
            t_ph = pd.Timestamp(t_ph_vals[0])
            if t_ph not in copper.index:
                continue
            realized_prices = copper.loc[t_ph].values
            mu_beta = sub[["beta0_hat", "beta1_hat", "beta2_hat"]].values[0]

            sig_h   = sigma_t.get((PRIMARY_MODEL, h), np.eye(3) * 1e-4)
            resid_h = resid_t.get((PRIMARY_MODEL, h), np.zeros((1, 3)))

            curves_g = sample_curves_gaussian(mu_beta, sig_h, tau, LAMBDA_NS, N_SAMPLES, rng)
            curves_b = sample_curves_bootstrap(mu_beta, resid_h, tau, LAMBDA_NS, N_SAMPLES, rng)

            j = MATURITIES_MONTHS.index(PRIMARY_MATURITY)
            if j >= len(current_prices) or np.isnan(current_prices[j]):
                continue
            p0 = current_prices[j]
            pr = realized_prices[j] if j < len(realized_prices) else np.nan
            if np.isnan(pr):
                continue

            up_barrier   = p0 * (1 + PRIMARY_THRESHOLD)
            down_barrier = p0 * (1 - PRIMARY_THRESHOLD)

            records.append({
                "date": date_t, "horizon": h,
                "p_up_gauss":   float(np.mean(curves_g[:, j] > up_barrier)),
                "p_down_gauss": float(np.mean(curves_g[:, j] < down_barrier)),
                "p_up_boot":    float(np.mean(curves_b[:, j] > up_barrier)),
                "p_down_boot":  float(np.mean(curves_b[:, j] < down_barrier)),
                "actual_up":   1 if pr > up_barrier else 0,
                "actual_down": 1 if pr < down_barrier else 0,
            })

    print("done.\n")
    df = pd.DataFrame(records)

    print("─" * 90)
    print(f"COMPARISON - {MODEL_LABELS[PRIMARY_MODEL]}, {PRIMARY_MATURITY}M maturity, "
          f"{int(PRIMARY_THRESHOLD*100)}% threshold")
    print("Method 1 = current production (Gaussian Monte Carlo, 08_probability_outputs.py)")
    print("Method 2 = empirical bootstrap of in-sample LP residuals (this test)")
    print("─" * 90)

    comparison_rows = []
    for h in PRIMARY_HORIZONS:
        sub = df[df.horizon == h]
        h_lbl = HORIZON_LABELS.get(h, f"{h}W")
        for direction in ["up", "down"]:
            p_g = sub[f"p_{direction}_gauss"].values
            p_b = sub[f"p_{direction}_boot"].values
            y   = sub[f"actual_{direction}"].values.astype(float)
            base_rate = y.mean()
            bs_g, bs_b = brier(p_g, y), brier(p_b, y)
            bss_g, bss_b = brier_skill(bs_g, base_rate), brier_skill(bs_b, base_rate)
            mean_p_g, mean_p_b = p_g.mean(), p_b.mean()
            print(f"\n{h_lbl:8s} P(ΔF {'>' if direction=='up' else '<'} "
                  f"{'+' if direction=='up' else '-'}{int(PRIMARY_THRESHOLD*100)}%)  "
                  f"[N={len(sub)}, base_rate={base_rate:.3f}]")
            print(f"  Gaussian : mean_pred={mean_p_g:.3f}  BS={bs_g:.4f}  BSS={bss_g:+.4f}")
            print(f"  Bootstrap: mean_pred={mean_p_b:.3f}  BS={bs_b:.4f}  BSS={bss_b:+.4f}  "
                  f"{'<-- IMPROVES' if bss_b > bss_g else '<-- worse or same'}")
            comparison_rows.append({
                "horizon": h, "horizon_lbl": h_lbl, "direction": direction, "N": len(sub),
                "base_rate": round(base_rate, 4),
                "mean_pred_gauss": round(mean_p_g, 4), "mean_pred_boot": round(mean_p_b, 4),
                "bs_gauss": round(bs_g, 5), "bs_boot": round(bs_b, 5),
                "bss_gauss": round(bss_g, 4), "bss_boot": round(bss_b, 4),
                "improves": bool(bss_b > bss_g),
            })

    comp_df = pd.DataFrame(comparison_rows)
    mean_bss_g = comp_df["bss_gauss"].mean()
    mean_bss_b = comp_df["bss_boot"].mean()
    print("\n" + "─" * 90)
    print(f"MEAN BSS across primary grid - Gaussian: {mean_bss_g:+.4f}   "
          f"Bootstrap: {mean_bss_b:+.4f}   (change: {mean_bss_b - mean_bss_g:+.4f})")
    n_improve = comp_df["improves"].sum()
    print(f"Bootstrap improves BSS in {n_improve}/{len(comp_df)} (horizon, direction) combinations")

    df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "probability_outputs_bootstrap.parquet"))
    comp_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "gaussian_vs_bootstrap_comparison.parquet"))
    print(f"\nSaved: data/processed/probability_outputs_bootstrap.parquet")
    print(f"Saved: data/processed/gaussian_vs_bootstrap_comparison.parquet")
    print("\nNo production files were modified. Review the comparison above before")
    print("deciding whether to promote the bootstrap approach into 08_probability_outputs.py.")


if __name__ == "__main__":
    main()
