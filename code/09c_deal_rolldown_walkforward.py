"""
09c_deal_rolldown_walkforward.py - Rebuilds 09a_deal_rolldown_probability.py's
EXACT methodology (same roll-down, same LME-Cash realized side, same
volatility threshold, same Platt fit) on the walk-forward, point-in-time-
honest residual pool (13e_walkforward_residual_pool.py) instead of the
idealised single-fit-per-window pool (rolling_lp_residuals.pkl).

WHY THIS SCRIPT: Chapter 6's own AUC table (0.93/0.77/0.84 in main.qmd) and
Platt fit rest on the same idealised residual pool already fixed for
Chapters 4-5 (13e/13f). This closes that gap for the deal-specific,
roll-down-adjusted analysis too, so Chapter 6 is built on the same honest
base as the rest of the thesis.

MODEL 0 (RANDOM WALK): computed fresh here directly from ns_factors.parquet
(an actual realised difference, not a fitted coefficient -- never idealised,
see 13e's docstring for the same point). MODELS 1-3: looked up from
walkforward_residuals.parquet (13e), horizon=13 only (the deal's own tenor),
with the same point-in-time embargo already used throughout this fix
(realized_date <= t -- a pool entry is eligible only once its own outcome
has actually happened as of the date being evaluated).

STANDALONE / NON-DESTRUCTIVE: does not touch 09/09a/13/14/15 or their
outputs.

Inputs : data/processed/factor_forecasts.parquet
         data/processed/walkforward_residuals.parquet   (13e)
         data/processed/ns_factors.parquet               (Model 0's own pool)
         data/processed/curves.parquet
         data/processed/lme_copper_cash.parquet
Outputs: data/processed/deal_rolldown_probabilities_walkforward.parquet
         data/processed/deal_rolldown_calibrated_walkforward.parquet
         report/figures/fig_model_alert_comparison_walkforward.png
         report/figures/fig_directional_accuracy_walkforward.png

Run: python code/09c_deal_rolldown_walkforward.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from scipy.stats import rankdata

from config import PROCESSED_DATA_DIR, FIGURES_DIR, LAMBDA_NS
from utils import trailing_vol_reference

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


# ── Identical to 09a_deal_rolldown_probability.py ───────────────────────────
def ns_loadings_vec(tau_scalar, lambda_ns):
    lt = tau_scalar / lambda_ns
    l_load = 1.0
    if lt < 1e-8:
        s_load, c_load = 1.0, 0.0
    else:
        s_load = (1 - np.exp(-lt)) / lt
        c_load = s_load - np.exp(-lt)
    return np.array([l_load, s_load, c_load])


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


def auc(p, y):
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan, int(n_pos), int(n_neg)
    ranks = rankdata(p)
    a = (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(a), int(n_pos), int(n_neg)


def build_model0_pool(factors):
    """Random Walk's own realised 13-week differences -- an actual observed
    quantity, not a fitted estimate, so there is no look-ahead to remove
    (see 13e's docstring for the same point). One row per origin."""
    combined = factors[FACTOR_COLS].dropna()
    diffs = combined.shift(-DEAL_HORIZON) - combined
    out = diffs.dropna().reset_index()
    out.columns = ["date", "resid_beta0", "resid_beta1", "resid_beta2"]
    out["realized_date"] = out["date"] + pd.to_timedelta(DEAL_HORIZON, unit="W")
    return out


def compute_for_model(model_id, factor_forecasts, resid_lookup, curves,
                       cash, sigma_series, loadings_rem, rng):
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
        sim_log_prices = sim_betas @ loadings_rem
        sim_prices = np.exp(sim_log_prices)

        idx_nearest = cash.index.get_indexer([t_ph], method="nearest",
                                              tolerance=pd.Timedelta(days=3))[0]
        actual_price = np.nan if idx_nearest == -1 else cash.loc[cash.index[idx_nearest]]
        if pd.isna(actual_price):
            actual_price = np.nan

        up_barrier, down_barrier = p0 * (1 + sigma_t), p0 * (1 - sigma_t)
        p_up = float(np.mean(sim_prices > up_barrier))
        p_down = float(np.mean(sim_prices < down_barrier))
        if np.isnan(actual_price):
            act_up, act_down = np.nan, np.nan
        else:
            act_up = 1 if actual_price > up_barrier else 0
            act_down = 1 if actual_price < down_barrier else 0

        sim_returns = sim_prices / p0 - 1.0
        model_expected_vol = float(np.std(sim_returns))
        if np.isnan(actual_price):
            realized_abs_move, naive_error, model_error = np.nan, np.nan, np.nan
        else:
            realized_abs_move = abs(actual_price / p0 - 1.0)
            naive_error = abs(float(sigma_t) - realized_abs_move)
            model_error = abs(model_expected_vol - realized_abs_move)

        records.append({
            "model": model_id, "date": t, "sigma_threshold": round(float(sigma_t), 4),
            "entry_price": round(float(p0), 2),
            "p_up": round(p_up, 4), "p_down": round(p_down, 4),
            "actual_up": act_up, "actual_down": act_down,
            "model_expected_vol": round(model_expected_vol, 4),
            "realized_abs_move": round(realized_abs_move, 4) if not np.isnan(realized_abs_move) else np.nan,
            "naive_error": round(naive_error, 4) if not np.isnan(naive_error) else np.nan,
            "model_error": round(model_error, 4) if not np.isnan(model_error) else np.nan,
            "n_pool": len(resid_pool),
        })
    return pd.DataFrame(records)


def plot_model_alert_comparison(calib_df, out_dir, suffix=""):
    """Plots the RAW probability, not the Platt-calibrated one. Section 3.8
    restricts calibration to AR-Direct, the only model whose ranking skill
    is validated as a curve-wide property, so fitting Platt to the other
    three would contradict that rule directly: LP+Macro+EW's upside slope
    collapses to a~0.05 if it is calibrated, flattening its whole line to
    the base rate, a symptom of calibrating a model with no real ranking
    skill there (AUC~0.43), not a plotting error. Chapter 6 uses the raw
    probability for the same reason. calib_df's *_cal columns and the
    Platt fit itself are kept upstream in main(), unused here, since 09d
    still depends on the saved calibrated parquet for its own
    out-of-sample comparison."""
    model_ids = sorted(MODEL_NAMES.keys())
    directions = ["up", "down"]
    fig, axes = plt.subplots(len(directions), len(model_ids), figsize=(16, 7), sharex=True, sharey=True)
    for i, direction in enumerate(directions):
        for j, m in enumerate(model_ids):
            ax = axes[i, j]
            sub = calib_df[calib_df.model == m].sort_values("date")
            p_col, act_col = f"p_{direction}", f"actual_{direction}"
            valid = sub.dropna(subset=[p_col, act_col])
            ax.plot(valid["date"], valid[p_col], color="lightgrey", linewidth=0.6, zorder=1)
            realized = valid[act_col] == 1
            ax.scatter(valid.loc[~realized, "date"], valid.loc[~realized, p_col],
                       color="firebrick", s=8, alpha=0.55, zorder=3, label="Not realized")
            ax.scatter(valid.loc[realized, "date"], valid.loc[realized, p_col],
                       color="forestgreen", s=22, zorder=5, label="Realized")
            ax.axvline(TARIFF_SHOCK_START, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.25)
            if i == 0:
                ax.set_title(MODEL_NAMES[m], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{'Up' if direction == 'up' else 'Down'}\nP(raw)")
            if i == len(directions) - 1:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.tick_params(axis="x", rotation=45)
    axes[0, 0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Raw probability vs realized outcome, by model and direction (walk-forward pool)\n"
                 "(green = threshold cleared that week, red = not; dashed line: tariff-shock boundary)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig_model_alert_comparison{suffix}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: fig_model_alert_comparison{suffix}.png")


def plot_directional_accuracy(calib_df, out_dir, suffix=""):
    """Raw probability, not calibrated -- see plot_model_alert_comparison's
    docstring for why."""
    model_ids = sorted(MODEL_NAMES.keys())
    fig, axes = plt.subplots(1, len(model_ids), figsize=(18, 4.5), sharex=True, sharey=True)
    summary_rows = []
    for j, m in enumerate(model_ids):
        ax = axes[j]
        sub = calib_df[calib_df.model == m].sort_values("date")
        valid = sub.dropna(subset=["p_up", "p_down"])
        ax.plot(valid["date"], valid["p_up"], color="firebrick", linewidth=0.7, alpha=0.6, label="P(up)")
        ax.plot(valid["date"], valid["p_down"], color="steelblue", linewidth=0.7, alpha=0.6, label="P(down)")
        favors_up = valid["p_up"] > valid["p_down"]
        up_events = valid[valid["actual_up"] == 1]
        up_correct = favors_up.loc[up_events.index]
        ax.scatter(up_events.loc[up_correct, "date"], up_events.loc[up_correct, "p_up"],
                   color="forestgreen", marker="^", s=50, zorder=5, label="Correct side favoured")
        ax.scatter(up_events.loc[~up_correct, "date"], up_events.loc[~up_correct, "p_up"],
                   color="black", marker="^", s=50, zorder=5, label="Wrong side favoured")
        down_events = valid[valid["actual_down"] == 1]
        down_correct = (~favors_up).loc[down_events.index]
        ax.scatter(down_events.loc[down_correct, "date"], down_events.loc[down_correct, "p_down"],
                   color="forestgreen", marker="v", s=50, zorder=5)
        ax.scatter(down_events.loc[~down_correct, "date"], down_events.loc[~down_correct, "p_down"],
                   color="black", marker="v", s=50, zorder=5)
        ax.axvline(TARIFF_SHOCK_START, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(MODEL_NAMES[m], fontsize=10)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45)
        n_events = len(up_events) + len(down_events)
        n_correct = int(up_correct.sum()) + int(down_correct.sum())
        summary_rows.append({"model": MODEL_NAMES[m], "N breach weeks": n_events, "N correct side": n_correct,
                              "directional accuracy": round(n_correct / n_events, 3) if n_events else np.nan})
    axes[0].set_ylabel("Raw probability")
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Directional accuracy (walk-forward pool): did the model favour the side that actually happened?")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig_directional_accuracy{suffix}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: fig_directional_accuracy{suffix}.png")
    print(pd.DataFrame(summary_rows).to_string(index=False))


def main():
    print("=" * 78)
    print("Roll-down probability construction, stylised deal -- WALK-FORWARD pool")
    print(f"tau={DEAL_MATURITY}M, h={DEAL_HORIZON}W, tau_rem={TAU_REM:.4f}M")
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
        print(f"  Model {m} ({MODEL_NAMES[m]}): {len(resid_lookup[m])} pool entries available "
              f"({resid_lookup[m]['date'].min().date()} -> {resid_lookup[m]['date'].max().date()})")

    curves = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "curves.parquet"))
    curves.index = pd.to_datetime(curves.index)
    cash = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "lme_copper_cash.parquet"))["cash"]
    cash.index = pd.to_datetime(cash.index)

    sigma_series = trailing_vol_reference(cash, horizon_weeks=DEAL_HORIZON, lookback_weeks=VOL_LOOKBACK_WEEKS)
    loadings_rem = ns_loadings_vec(TAU_REM, LAMBDA_NS)
    # Each model gets its own independent RNG stream, matching
    # 13_bootstrap_probability_outputs.py. All four models here draw via
    # the same rng.integers call, so a single shared generator advanced across
    # models in sequence was not silently miscalibrating anything -- but it
    # still meant a model's reported numbers were not, strictly, a function of
    # that model's own data and code alone (e.g. changing the horizon grid or
    # adding a fifth model would shift every existing model's draws too).
    # Spawning independent child generators removes that latent coupling.
    rngs = dict(zip(MODEL_NAMES.keys(), np.random.default_rng(42).spawn(len(MODEL_NAMES))))

    raw_df = pd.concat([
        compute_for_model(m, factor_forecasts, resid_lookup, curves, cash, sigma_series, loadings_rem, rngs[m])
        for m in MODEL_NAMES
    ], ignore_index=True)
    raw_path = os.path.join(PROCESSED_DATA_DIR, "deal_rolldown_probabilities_walkforward.parquet")
    raw_df.to_parquet(raw_path)
    print(f"\nSaved: {raw_path}  ({len(raw_df)} rows)")

    print("\n--- AUC comparison, walk-forward pool ---")
    raw_df["period"] = np.where(raw_df["date"] >= TARIFF_SHOCK_START, "shock", "stable")
    for m in MODEL_NAMES:
        sub = raw_df[raw_df.model == m]
        for direction in ["up", "down"]:
            for period in ["stable", "shock"]:
                g = sub[sub.period == period]
                valid = g[[f"p_{direction}", f"actual_{direction}"]].dropna()
                if len(valid) < 10:
                    print(f"  {MODEL_NAMES[m]:12s} {direction:5s} {period:8s}: N={len(valid)} (too few to test)")
                    continue
                a, npos, nneg = auc(valid[f"p_{direction}"].values, valid[f"actual_{direction}"].values)
                print(f"  {MODEL_NAMES[m]:12s} {direction:5s} {period:8s}: "
                      f"AUC={a:.4f}  (n_pos={npos}, n_neg={nneg}, N={len(valid)})")

    calib_df = raw_df.copy()
    calib_df["p_up_cal"] = np.nan
    calib_df["p_down_cal"] = np.nan
    calib_df["in_calibration_domain"] = calib_df["date"] < TARIFF_SHOCK_START
    coef_rows = []
    for m, grp_model in calib_df.groupby("model"):
        fit_grp = grp_model[grp_model["in_calibration_domain"]]
        for direction in ["up", "down"]:
            p_col, act_col, cal_col = f"p_{direction}", f"actual_{direction}", f"p_{direction}_cal"
            valid_fit = fit_grp[[p_col, act_col]].dropna()
            if len(valid_fit) < 20:
                continue
            a, b = fit_platt(valid_fit[p_col].values, valid_fit[act_col].values.astype(float))
            all_valid = grp_model[[p_col, act_col]].dropna()
            p_cal_all = sigmoid(a * logit(all_valid[p_col].values) + b)
            calib_df.loc[all_valid.index, cal_col] = p_cal_all
            base_rate = float(valid_fit[act_col].mean())
            bs_raw = float(np.mean((valid_fit[p_col].values - valid_fit[act_col].values) ** 2))
            bs_cal = float(np.mean((sigmoid(a * logit(valid_fit[p_col].values) + b) - valid_fit[act_col].values) ** 2))
            bs_clim = base_rate * (1 - base_rate)
            bss_raw = 1 - bs_raw / bs_clim if bs_clim > 0 else np.nan
            bss_cal = 1 - bs_cal / bs_clim if bs_clim > 0 else np.nan
            coef_rows.append({"model": MODEL_NAMES[m], "direction": direction, "a": round(a, 4), "b": round(b, 4),
                               "N_fit": len(valid_fit), "base_rate": round(base_rate, 4),
                               "bss_raw": round(bss_raw, 4), "bss_cal": round(bss_cal, 4)})
    coef_df = pd.DataFrame(coef_rows)
    print("\n--- Platt calibration, walk-forward pool ---")
    print(coef_df.to_string(index=False))

    calib_path = os.path.join(PROCESSED_DATA_DIR, "deal_rolldown_calibrated_walkforward.parquet")
    calib_df.to_parquet(calib_path)
    print(f"\nSaved: {calib_path}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_model_alert_comparison(calib_df, FIGURES_DIR, suffix="_walkforward")
    plot_directional_accuracy(calib_df, FIGURES_DIR, suffix="_walkforward")

    print("\nDone. 09/09a/13/14/15 and their outputs were not modified.")


if __name__ == "__main__":
    main()
