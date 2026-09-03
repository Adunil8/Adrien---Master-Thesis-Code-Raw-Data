"""
09e_ch5_constant_maturity_trailing_vol.py - Rebuilds Chapter 5's "Probability
Calibration and Rank Discrimination" subsection on the SAME threshold
convention the deal itself uses (trailing realised volatility, scaled to the
deal's own 13-week tenor, evolving week by week) instead of the fixed k=8%
convention that subsection used previously.

WHY THIS SCRIPT: the fixed-k=8% design (13_bootstrap_probability_outputs.py
-> 15_calibration_diagnostics.py) answers "does this model rank weeks
correctly against an externally-asserted 8% bar." That is a controlled,
comparable-across-cells design, useful for the internal model comparison,
but it is not the bar a bank actually uses. The question this script answers
instead: holding the SAME real-world threshold the deal will use (trailing
vol, adjusted to the deal's own tenor) fixed, do these models still provide
genuine, exploitable probabilities on top of the trailing-vol reference
itself, more precision than the status quo, not a wholesale replacement
of it?

RELATIONSHIP TO 09c: 09c already runs this exact trailing-vol-threshold
design, but on the deal's ROLLED-DOWN price (tau_rem = 0 at the deal's own
maturity date, realised side = LME Cash). This script uses the SAME
threshold construction and the SAME residual pools, but evaluates the
CONSTANT-MATURITY 3M contract instead (tau fixed at 3 months, never
decremented; realised side = the 3M contract's own future price from
curves.parquet, not LME Cash). That is the same maturity convention already
used by every other main-text table in Chapter 5 (tbl-rmse-subperiod,
tbl-hitrate), so this brings the AUC/BSS subsection into line with its own
chapter rather than mixing two maturity conventions across one section.

MODEL 0 (RANDOM WALK): computed fresh from ns_factors.parquet's own realised
13-week factor differences, identical logic to 09c/13e (an actual observed
quantity, never idealised, no look-ahead to remove).
MODELS 1-3: looked up from walkforward_residuals.parquet (13e), horizon=13
only (the deal's own tenor), same point-in-time embargo (realized_date <= t)
used throughout the rest of the walk-forward-pool fix.

SCOPE: maturity=3M, horizon=13W only (the one cell that matches the deal's
own tenor, per the same single-cell design tbl-rmse-subperiod and
tbl-hitrate already use). This is deliberately narrower than the old
fixed-8% design's full 1-6M/multi-horizon grid -- that grid existed to
support the internal, threshold-controlled model comparison; a
trailing-vol threshold that itself only makes sense relative to a specific
tenor does not extend to that grid the same way.

STANDALONE / NON-DESTRUCTIVE: does not touch 07/08/09/09a/09c/13/14/15 or
their outputs.

Inputs : data/processed/factor_forecasts.parquet
         data/processed/ns_factors.parquet
         data/processed/walkforward_residuals.parquet   (13e)
         data/processed/curves.parquet
         data/processed/lme_copper_cash.parquet          (trailing-vol basis)
Outputs: data/processed/ch5_constant_maturity_trailing_vol_probs.parquet
         data/processed/ch5_constant_maturity_trailing_vol_auc.parquet
         data/processed/ch5_constant_maturity_trailing_vol_bss.parquet

Run: python code/09e_ch5_constant_maturity_trailing_vol.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config import PROCESSED_DATA_DIR, LAMBDA_NS

MODEL_NAMES = {0: "Random Walk", 1: "AR-Direct", 2: "LP+Macro", 3: "LP+Macro+EW"}
DEAL_MATURITY = 3          # 3M contract, matching tbl-rmse-subperiod / tbl-hitrate
DEAL_HORIZON = 13          # 13 weeks = 3M tenor, matching the stylised deal
VOL_LOOKBACK_WEEKS = 52
N_SAMPLES = 1000
TARIFF_SHOCK_START = pd.Timestamp("2025-04-11")
FACTOR_COLS = ["beta0", "beta1", "beta2"]
MIN_CLASS_COUNT = 5


def trailing_vol_reference(front_month, horizon_weeks, lookback_weeks=52):
    """Identical to utils.trailing_vol_reference -- reproduced here only to
    keep this script's dependency list minimal and auditable in isolation;
    same formula, same source function, no divergence in logic."""
    from utils import trailing_vol_reference as _tvr
    return _tvr(front_month, horizon_weeks, lookback_weeks)


def ns_loadings_vec(tau_scalar, lambda_ns):
    lt = tau_scalar / lambda_ns
    l_load = 1.0
    if lt < 1e-8:
        s_load, c_load = 1.0, 0.0
    else:
        s_load = (1 - np.exp(-lt)) / lt
        c_load = s_load - np.exp(-lt)
    return np.array([l_load, s_load, c_load])


def auc(p, y):
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan, int(n_pos), int(n_neg)
    ranks = rankdata(p)
    a = (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(a), int(n_pos), int(n_neg)


def classify_verdict(a, n_pos, n_neg):
    if np.isnan(a):
        return "UNDEFINED (degenerate class)"
    if min(n_pos, n_neg) < MIN_CLASS_COUNT:
        return f"INSUFFICIENT SAMPLE (min class n={min(n_pos, n_neg)})"
    if a > 0.58:
        return "GENUINE SKILL"
    if a >= 0.42:
        return "weak/none"
    return "INVERTED (worse than random)"


def build_model0_pool(factors):
    """Random Walk's own realised 13-week factor differences -- an actual
    observed quantity, not a fitted estimate. Identical to 09c."""
    combined = factors[FACTOR_COLS].dropna()
    diffs = combined.shift(-DEAL_HORIZON) - combined
    out = diffs.dropna().reset_index()
    out.columns = ["date", "resid_beta0", "resid_beta1", "resid_beta2"]
    out["realized_date"] = out["date"] + pd.to_timedelta(DEAL_HORIZON, unit="W")
    return out


def compute_for_model(model_id, factor_forecasts, resid_lookup, curves, sigma_series, loadings, rng):
    entry_col = f"m{DEAL_MATURITY:02d}"
    ff = factor_forecasts[(factor_forecasts.model == model_id) &
                           (factor_forecasts.horizon == DEAL_HORIZON)].copy()
    ff["date"] = pd.to_datetime(ff["date"])
    ff["t_plus_h"] = pd.to_datetime(ff["t_plus_h"])

    pool_all = resid_lookup[model_id]
    records = []
    for _, row in ff.iterrows():
        t, t_ph = row["date"], row["t_plus_h"]
        if t not in curves.index or entry_col not in curves.columns:
            continue
        p0 = curves.loc[t, entry_col]
        if pd.isna(p0) or p0 <= 0:
            continue
        sigma_t = sigma_series.get(t, np.nan)
        if pd.isna(sigma_t):
            continue

        mu_beta = row[["beta0_hat", "beta1_hat", "beta2_hat"]].values.astype(float)
        eligible = pool_all[pool_all["realized_date"] <= t]
        if len(eligible) < 5:
            continue
        resid_pool = eligible[["resid_beta0", "resid_beta1", "resid_beta2"]].values

        idx = rng.integers(0, len(resid_pool), size=N_SAMPLES)
        sim_betas = mu_beta[None, :] + resid_pool[idx]
        sim_prices = np.exp(sim_betas @ loadings)

        # Realised side: the SAME constant-maturity 3M contract's own future
        # price, from curves.parquet -- not LME Cash (that is 09c's, for its
        # rolled-down-to-zero-tau convention only).
        actual_price = curves.loc[t_ph, entry_col] if t_ph in curves.index else np.nan
        if pd.isna(actual_price) or actual_price <= 0:
            actual_price = np.nan

        up_barrier, down_barrier = p0 * (1 + sigma_t), p0 * (1 - sigma_t)
        p_up = float(np.mean(sim_prices > up_barrier))
        p_down = float(np.mean(sim_prices < down_barrier))
        if np.isnan(actual_price):
            act_up, act_down = np.nan, np.nan
        else:
            act_up = 1 if actual_price > up_barrier else 0
            act_down = 1 if actual_price < down_barrier else 0

        records.append({
            "model": model_id, "date": t, "sigma_threshold": round(float(sigma_t), 4),
            "entry_price": round(float(p0), 2),
            "p_up": round(p_up, 4), "p_down": round(p_down, 4),
            "actual_up": act_up, "actual_down": act_down,
            "n_pool": len(resid_pool),
        })
    return pd.DataFrame(records)


def main():
    print("=" * 78)
    print("Chapter 5 probability calibration + AUC -- constant maturity, evolving trailing-vol threshold")
    print(f"maturity={DEAL_MATURITY}M (constant, no roll-down), horizon={DEAL_HORIZON}W")
    print("=" * 78)

    factor_forecasts = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "factor_forecasts.parquet"))
    factors = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet"))[FACTOR_COLS]
    wf = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "walkforward_residuals.parquet"))
    wf13 = wf[wf.horizon == DEAL_HORIZON]

    resid_lookup = {0: build_model0_pool(factors)}
    for m in [1, 2, 3]:
        resid_lookup[m] = wf13[wf13.model == m][
            ["date", "resid_beta0", "resid_beta1", "resid_beta2", "realized_date"]].reset_index(drop=True)
    for m in [0, 1, 2, 3]:
        print(f"  Model {m} ({MODEL_NAMES[m]}): {len(resid_lookup[m])} pool entries available")

    curves = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "curves.parquet"))
    curves.index = pd.to_datetime(curves.index)
    cash = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "lme_copper_cash.parquet"))["cash"]
    cash.index = pd.to_datetime(cash.index)

    # Threshold basis: same trailing-vol construction as the deal (09c), same
    # tenor (13W = 3M). Cash is used only to ESTIMATE the trailing-vol level
    # itself (the market's own recent realised volatility) -- it plays no
    # role in defining the barrier's price level or the realised outcome,
    # both of which use the 3M contract throughout this script.
    sigma_series = trailing_vol_reference(cash, horizon_weeks=DEAL_HORIZON, lookback_weeks=VOL_LOOKBACK_WEEKS)
    loadings = ns_loadings_vec(float(DEAL_MATURITY), LAMBDA_NS)  # constant maturity: never decremented

    rngs = dict(zip(MODEL_NAMES.keys(), np.random.default_rng(42).spawn(len(MODEL_NAMES))))

    raw_df = pd.concat([
        compute_for_model(m, factor_forecasts, resid_lookup, curves, sigma_series, loadings, rngs[m])
        for m in MODEL_NAMES
    ], ignore_index=True)
    raw_path = os.path.join(PROCESSED_DATA_DIR, "ch5_constant_maturity_trailing_vol_probs.parquet")
    raw_df.to_parquet(raw_path)
    print(f"\nSaved: {raw_path}  ({len(raw_df)} rows)")

    raw_df["period"] = np.where(raw_df["date"] >= TARIFF_SHOCK_START, "shock", "stable")

    print("\n--- AUC, constant maturity, evolving trailing-vol threshold ---")
    auc_rows = []
    for m in MODEL_NAMES:
        sub = raw_df[raw_df.model == m]
        for direction in ["up", "down"]:
            for period in ["stable", "shock"]:
                g = sub[sub.period == period]
                valid = g[[f"p_{direction}", f"actual_{direction}"]].dropna()
                a, npos, nneg = auc(valid[f"p_{direction}"].values, valid[f"actual_{direction}"].values) \
                    if len(valid) >= 10 else (np.nan, len(valid[valid[f"actual_{direction}"] == 1]),
                                               len(valid[valid[f"actual_{direction}"] == 0]))
                verdict = classify_verdict(a, npos, nneg) if len(valid) >= 10 else "TOO FEW (N<10)"
                auc_rows.append({"model": MODEL_NAMES[m], "direction": direction, "period": period,
                                  "auc": round(a, 4) if not np.isnan(a) else np.nan,
                                  "n_pos": npos, "n_neg": nneg, "N": len(valid), "verdict": verdict})
                if not np.isnan(a):
                    print(f"  {MODEL_NAMES[m]:12s} {direction:5s} {period:6s}: "
                          f"AUC={a:.4f}  (n_pos={npos}, n_neg={nneg})  {verdict}")
                else:
                    print(f"  {MODEL_NAMES[m]:12s} {direction:5s} {period:6s}: N={len(valid)} (too few)")
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "ch5_constant_maturity_trailing_vol_auc.parquet"))

    print("\n--- Raw Brier Skill Score, constant maturity, evolving trailing-vol threshold ---")
    bss_rows = []
    for m in MODEL_NAMES:
        sub = raw_df[raw_df.model == m]
        for direction in ["up", "down"]:
            for period in ["stable", "shock"]:
                g = sub[sub.period == period]
                valid = g[[f"p_{direction}", f"actual_{direction}"]].dropna()
                if len(valid) < 10:
                    continue
                p, y = valid[f"p_{direction}"].values, valid[f"actual_{direction}"].values.astype(float)
                bs = float(np.mean((p - y) ** 2))
                base_rate = float(y.mean())
                bs_clim = base_rate * (1 - base_rate)
                bss = 1 - bs / bs_clim if bs_clim > 0 else np.nan
                bss_rows.append({"model": MODEL_NAMES[m], "direction": direction, "period": period,
                                  "brier_score": round(bs, 5), "base_rate": round(base_rate, 4),
                                  "brier_skill": round(bss, 4) if not np.isnan(bss) else np.nan, "N": len(valid)})
                print(f"  {MODEL_NAMES[m]:12s} {direction:5s} {period:6s}: "
                      f"BSS={bss:+.4f}  (base_rate={base_rate:.3f}, N={len(valid)})")
    pd.DataFrame(bss_rows).to_parquet(os.path.join(PROCESSED_DATA_DIR, "ch5_constant_maturity_trailing_vol_bss.parquet"))

    print("\nDone. 07/08/09/09a/09c/13/14/15 and their outputs were not modified.")


if __name__ == "__main__":
    main()
