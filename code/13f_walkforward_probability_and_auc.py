"""
13f_walkforward_probability_and_auc.py - Rebuilds the bootstrap probability
outputs for models 1-3 using the walk-forward, point-in-time-honest residual
pool (13e_walkforward_residual_pool.py), then re-runs the SAME AUC
diagnostic as 15_calibration_diagnostics.py on the result, so Section 5.2's
"AR-Direct is the only model with curve-wide robust skill" conclusion can be
directly re-tested rather than assumed to survive the residual-pool fix.

Model 0 (Random Walk) is carried over UNCHANGED from the existing production
file: it has no fitted coefficient, so its residual pool was never idealised
(see 13e's docstring) -- there is nothing to recompute.

MODEL 3 RESAMPLING WEIGHTING, a detail worth being explicit about: each pool
entry is already a residual computed at ITS OWN origin using an EW-weighted
fit (recency relative to THAT origin). A separate question is how much each
pooled origin should count when building the CURRENT test date's uncertainty
band. Following the same principle 13_bootstrap_probability_outputs.py uses
(the point estimate and the uncertainty band must reflect the same view of
history), Model 3's resampling is ALSO recency-weighted -- by lambda_ew
raised to the (test date - origin) gap in weeks -- so more recent origins
count more when building today's band, consistent with the point forecast
itself being recency-tuned. Models 1-2 resample uniformly, matching their
equally-weighted point estimates.

STANDALONE / NON-DESTRUCTIVE: does not touch 07/08/09/13/14/15 or their
outputs.

Inputs : data/processed/walkforward_residuals.parquet   (13e)
         data/processed/factor_forecasts.parquet          (point forecasts,
                                                             unaffected by
                                                             this fix)
         data/processed/curves.parquet
         data/processed/probability_outputs_bootstrap.parquet (model 0 only,
                                                             carried over)
Outputs: data/processed/probability_outputs_bootstrap_walkforward.parquet
         data/processed/brier_scores_bootstrap_walkforward.parquet
         data/processed/calibration_diagnostics_walkforward.parquet

Run: python code/13f_walkforward_probability_and_auc.py   (~2-4 minutes)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config import (
    MATURITIES_MONTHS, LAMBDA_NS, LAMBDA_EW, FORECAST_HORIZONS_WEEKS,
    MARGIN_THRESHOLDS, PROCESSED_DATA_DIR, HORIZON_LABELS,
)
from utils import ns_curve

N_SAMPLES = 1000
TARIFF_SHOCK_START = pd.Timestamp("2025-04-11")
PRIMARY_MATURITY, PRIMARY_THRESHOLD = 3, 0.08
MODEL_LABELS = {0: "Random Walk", 1: "AR-Direct", 2: "LP+Macro", 3: "LP+Macro+EW"}


# ── Duplicated from 13_bootstrap_probability_outputs.py ────────────────────
def sample_curves_bootstrap(mu_beta, resid_pool, tau, lambda_ns, n_samples, rng, resid_weights=None):
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


# ── Duplicated from 15_calibration_diagnostics.py ───────────────────────────
def auc(p: np.ndarray, y: np.ndarray) -> tuple[float, int, int]:
    y = y.astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan, int(n_pos), int(n_neg)
    ranks = rankdata(p)
    auc_val = (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc_val), int(n_pos), int(n_neg)


MIN_CLASS_COUNT = 5


def classify_verdict(a: float, n_pos: int, n_neg: int) -> str:
    if np.isnan(a):
        return "UNDEFINED (degenerate class - no variation to rank)"
    if min(n_pos, n_neg) < MIN_CLASS_COUNT:
        return f"INSUFFICIENT SAMPLE (min class n={min(n_pos, n_neg)})"
    if a > 0.58:
        return "GENUINE SKILL"
    if a >= 0.42:
        return "weak/none"
    return "INVERTED (worse than random)"


def compute_probability_outputs(factor_forecasts_df, copper_df, walkfwd_resid, tau, lambda_ns,
                                 thresholds, lambda_ew, n_samples=N_SAMPLES):
    rng = np.random.default_rng(42)
    records = []
    dates = sorted(factor_forecasts_df["date"].unique())
    n_total = len(dates)
    print(f"Computing walk-forward-pool probability outputs for {n_total} test dates...")

    for i, date_t in enumerate(dates):
        if i % 30 == 0:
            print(f"  {i}/{n_total}...", end=" ", flush=True)
        date_t = pd.Timestamp(date_t)
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

            pool_h = walkfwd_resid[(walkfwd_resid.horizon == h) & (walkfwd_resid.realized_date <= date_t)]

            for model_id in [1, 2, 3]:
                sub_m = sub[sub["model"] == model_id]
                if sub_m.empty:
                    continue
                mu_beta = sub_m[["beta0_hat", "beta1_hat", "beta2_hat"]].values[0]

                pool = pool_h[pool_h.model == model_id]
                resid_arr = pool[["resid_beta0", "resid_beta1", "resid_beta2"]].values
                weights = None
                if model_id == 3 and len(pool) > 0:
                    gap_weeks = (date_t - pool["date"]).dt.days / 7.0
                    weights = lambda_ew ** gap_weeks.values

                curves = sample_curves_bootstrap(mu_beta, resid_arr, tau, lambda_ns, n_samples, rng,
                                                  resid_weights=weights)

                for j, mat in enumerate(MATURITIES_MONTHS):
                    if j >= len(current_prices) or np.isnan(current_prices[j]):
                        continue
                    p0 = current_prices[j]
                    pr = realized_prices[j] if j < len(realized_prices) else np.nan

                    for k in thresholds:
                        up_barrier = p0 * (1 + k)
                        down_barrier = p0 * (1 - k)
                        sample_col = curves[:, j]
                        p_up = float(np.mean(sample_col > up_barrier))
                        p_down = float(np.mean(sample_col < down_barrier))
                        act_up = (1 if pr > up_barrier else 0) if not np.isnan(pr) else np.nan
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
            rows.append({"model": model, "horizon": h, "maturity": mat, "threshold": k,
                         "direction": direction, "brier_score": round(bs, 5),
                         "brier_skill": round(bss, 4), "base_rate": round(base_rate, 4), "N": len(valid)})
    return pd.DataFrame(rows)


def run_auc_diagnostic(prob_all):
    prob_all["date"] = pd.to_datetime(prob_all["date"])
    prob_all["period"] = np.where(prob_all["date"] >= TARIFF_SHOCK_START, "tariff_shock", "stable")
    maturities = sorted(prob_all.maturity.unique())
    rows = []
    for model_id in [0, 1, 2, 3]:
        for h in [4, 13, 26]:
            for direction in ["up", "down"]:
                p_col, act_col = f"p_{direction}", f"actual_{direction}"
                for period in ["stable", "tariff_shock"]:
                    for mat in maturities:
                        sub = prob_all[(prob_all.model == model_id) & (prob_all.horizon == h) &
                                       (prob_all.period == period) & (prob_all.maturity == mat) &
                                       (prob_all.threshold == PRIMARY_THRESHOLD)]
                        valid = sub[[p_col, act_col]].dropna()
                        if len(valid) < 10:
                            continue
                        a, npos, nneg = auc(valid[p_col].values, valid[act_col].values)
                        verdict = classify_verdict(a, npos, nneg)
                        rows.append({"model": model_id, "horizon": h, "direction": direction,
                                     "period": period, "maturity": mat, "auc": round(a, 4),
                                     "n_pos": npos, "n_neg": nneg, "verdict": verdict})
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("Walk-forward-pool probability outputs + AUC re-test (models 1-3 rebuilt)")
    print("=" * 70)

    factor_forecasts = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "factor_forecasts.parquet"))
    copper = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "curves.parquet"))
    walkfwd_resid = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "walkforward_residuals.parquet"))
    opt_path = os.path.join(PROCESSED_DATA_DIR, "lambda_ew_optimal.parquet")
    lambda_ew = float(pd.read_parquet(opt_path)["lambda_ew_optimal"].iloc[0]) if os.path.exists(opt_path) else LAMBDA_EW
    tau = np.array(MATURITIES_MONTHS, dtype=float)

    factor_forecasts_123 = factor_forecasts[factor_forecasts["model"].isin([1, 2, 3])]
    prob_123 = compute_probability_outputs(factor_forecasts_123, copper, walkfwd_resid, tau,
                                            LAMBDA_NS, MARGIN_THRESHOLDS, lambda_ew)

    prod_path = os.path.join(PROCESSED_DATA_DIR, "probability_outputs_bootstrap.parquet")
    prod = pd.read_parquet(prod_path)
    prob_0 = prod[prod["model"] == 0].copy()
    print(f"\nModel 0 (Random Walk): {len(prob_0)} rows carried over unchanged from production "
          f"(no fitted coefficient, never idealised).")

    prob_all = pd.concat([prob_0, prob_123], ignore_index=True)
    prob_all.to_parquet(os.path.join(PROCESSED_DATA_DIR, "probability_outputs_bootstrap_walkforward.parquet"))
    print(f"Saved: probability_outputs_bootstrap_walkforward.parquet ({len(prob_all)} rows)")

    brier_df = compute_brier_scores(prob_all)
    brier_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "brier_scores_bootstrap_walkforward.parquet"))
    print("Saved: brier_scores_bootstrap_walkforward.parquet")

    # ── BSS comparison: production vs walk-forward-pool, primary cells ─────
    prod_brier = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "brier_scores_bootstrap.parquet"))
    print("\n" + "-" * 100)
    print(f"RAW BSS COMPARISON - production (single-fit-per-window pool) vs walk-forward-pool, "
          f"{PRIMARY_MATURITY}M, {int(PRIMARY_THRESHOLD*100)}% threshold")
    print("-" * 100)
    for model_id in [1, 2, 3]:
        for h in [4, 13, 26]:
            for direction in ["up", "down"]:
                p_row = prod_brier[(prod_brier.model == model_id) & (prod_brier.horizon == h) &
                                    (prod_brier.maturity == PRIMARY_MATURITY) &
                                    (prod_brier.threshold == PRIMARY_THRESHOLD) &
                                    (prod_brier.direction == direction)]
                w_row = brier_df[(brier_df.model == model_id) & (brier_df.horizon == h) &
                                  (brier_df.maturity == PRIMARY_MATURITY) &
                                  (brier_df.threshold == PRIMARY_THRESHOLD) &
                                  (brier_df.direction == direction)]
                bss_p = float(p_row["brier_skill"].iloc[0]) if not p_row.empty else np.nan
                bss_w = float(w_row["brier_skill"].iloc[0]) if not w_row.empty else np.nan
                n_w = int(w_row["N"].iloc[0]) if not w_row.empty else 0
                print(f"{MODEL_LABELS[model_id]:<14} h={h:>2}W {direction:>4}: "
                      f"production BSS={bss_p:+.4f}   walk-forward BSS={bss_w:+.4f}  (N={n_w})")

    # ── AUC re-test ──────────────────────────────────────────────────────────
    print("\n" + "-" * 100)
    print("AUC re-test on walk-forward-pool probabilities (mirrors 15_calibration_diagnostics.py)")
    print("-" * 100)
    auc_df = run_auc_diagnostic(prob_all.copy())
    auc_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "calibration_diagnostics_walkforward.parquet"))

    prod_auc = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "calibration_diagnostics.parquet"))
    prod_grid = prod_auc[prod_auc["scope"] == "maturity_grid"] if "scope" in prod_auc.columns else prod_auc

    print(f"\n{'Model':<14} {'Period':<13} {'Mean AUC (production)':>24} {'Mean AUC (walk-forward)':>26}")
    print("-" * 82)
    for model_id in [0, 1, 2, 3]:
        for period in ["stable", "tariff_shock"]:
            p_sub = prod_grid[(prod_grid.model == model_id) & (prod_grid.period == period) &
                               (~prod_grid["verdict"].str.startswith(("UNDEFINED", "INSUFFICIENT")))]
            w_sub = auc_df[(auc_df.model == model_id) & (auc_df.period == period) &
                            (~auc_df["verdict"].str.startswith(("UNDEFINED", "INSUFFICIENT")))]
            mean_p = p_sub["auc"].mean() if not p_sub.empty else np.nan
            mean_w = w_sub["auc"].mean() if not w_sub.empty else np.nan
            print(f"{MODEL_LABELS[model_id]:<14} {period:<13} {mean_p:>24.4f} {mean_w:>26.4f}")

    print(f"\nSaved: calibration_diagnostics_walkforward.parquet")
    print("\nDone. 07/08/09/13/14/15 and their outputs were not modified.")


if __name__ == "__main__":
    main()
