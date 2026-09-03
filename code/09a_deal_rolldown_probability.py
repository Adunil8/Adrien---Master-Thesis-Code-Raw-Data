"""
09a_deal_rolldown_probability.py - Roll-down-adjusted probability construction
for the stylised deal specifically (Chapter 6), as distinct from the
constant-maturity convention used for the curve-forecasting results in
Chapters 4-5 (13_bootstrap_probability_outputs.py).

WHY A SEPARATE SCRIPT, NOT A CHANGE TO 13:
Chapters 4-5 ask a curve-DYNAMICS question (how does the term structure
move), for which the constant-maturity comparison F(tau, t+h) vs F(tau, t)
is the standard, literature-consistent quantity (Diebold & Li 2006; Bianchi
et al. 2023) and is retained unchanged there. The stylised deal (Chapter 6)
asks a different, POSITION-specific question: what happens to the price of
the one dated contract hedging this deal as it ages toward its own prompt
date, evaluated at the deal's own closing date, h equal to the deal's tenor.

Two design choices follow from the same underlying idea: stop asserting
numbers this thesis cannot independently defend, and use directly observed
prices wherever they exist instead of a model-based stand-in.

  (1) THE THRESHOLD IS NOT A FIXED PERCENTAGE (5/8/10/15%). Those
      numbers describe a TPA credit line, a real but unverifiable,
      bilaterally-negotiated figure (Section 3.9's own discussion admits
      "no single such rate exists to cite"). There is no defensible reason
      to fix that particular number as the object this analysis tests
      against. Instead the threshold is the trailing realised volatility
      itself (`utils.trailing_vol_reference`), computed fresh at every
      date and scaled to the deal's own tenor: a self-referential question,
      "will the market move more than its own recent normal range implies
      over this deal's tenor," not a comparison against an asserted
      external number. A 2-month deal and a 3-month deal read different
      thresholds off the same underlying volatility (different horizon
      scaling), and the same deal signed on two different dates reads two
      different thresholds too, since sigma itself is time-varying.

  (2) THE REALIZED SIDE USES THE DIRECTLY OBSERVED LME CASH PRICE, not a
      Nelson-Siegel reconstruction and not LP1. At the deal's own tenor,
      remaining maturity hits exactly zero (see ns_loadings_vec below), and
      a contract at zero remaining maturity is, economically, a cash
      position. An earlier version of this script used LP1 (the front-month
      generic series) as a stand-in for that cash value, since the original
      Bloomberg extraction (Appendix II) pulled LP1-LP24 but not LME Cash
      (`LMCADY Comdty`). That was an approximation with a real flaw: LP1 is
      itself a rolling series, quoted at any date D as the price for a
      contract roughly 1 month out FROM D, so LP1 read exactly at t+h
      implies a delivery date about a month AFTER the deal's own fixed
      delivery date, not on it.

      This is now fixed properly rather than worked around. A public daily
      LME Copper Cash-Settlement series (westmetall.com, 2008-2026,
      cleaned by 01d_lme_cash_cleaning.py into
      data/processed/lme_copper_cash.parquet) is used instead: a genuine
      zero-maturity spot price with no rolling-maturity drift to correct
      for, read directly at t+h. The ENTRY price is still the deal's own
      tenor-matched contract (m03 for a 3M deal), since that is the price
      the deal is actually transacted against; only the EXIT/closing side
      uses Cash.

      This fix is scoped to the deal's own realized-price comparison only.
      It does not add Cash as a point in the core Nelson-Siegel
      cross-sectional fit (Section 3.2), which would ripple through the
      entire factor-extraction and forecasting pipeline (Chapters 3-6) and
      require re-verifying every downstream empirical result; that remains
      documented as future work (Section 3.9), not undertaken here. The
      MODEL'S OWN forecast already targets the right quantity regardless:
      loadings_rem (below) evaluates the NS loading vector at tau=0
      exactly, so the model's simulated distribution already represents
      what Cash would be, according to the model's forecasted curve. Only
      the observed/realized side, and the front-end anchor (or lack of
      one) in the NS fit itself, were ever the gap.

  (3) ALL FOUR MODELS ARE RUN, NOT ONLY AR-DIRECT. Chapter 5 established
      that AR-Direct has the only validated rank-discrimination skill
      under the fixed-threshold convention. Whether that ranking still
      holds under a self-referential, time-varying threshold is a
      separate empirical question, not an assumption carried over
      automatically, so it is checked here directly (see the AUC table
      printed below) rather than presumed.

Inputs  : data/processed/factor_forecasts.parquet   (each model's forecasted
                                                       beta at t+h, from 07)
          data/processed/rolling_lp_residuals.pkl    (each model's residual
                                                       pool per test date, from 13)
          data/processed/curves.parquet              (F(tau, t), the deal's
                                                       entry price, m03)
          data/processed/lme_copper_cash.parquet     (LME Cash, from 01b;
                                                       the realized exit
                                                       price and the
                                                       volatility reference,
                                                       both computed on Cash)
Outputs : data/processed/deal_rolldown_probabilities.parquet   (raw, all 4 models)
          data/processed/deal_rolldown_calibrated.parquet      (Platt-calibrated, all 4 models)
          report/figures/fig_model_alert_comparison.png        (per-model, per-direction
                                                                  calibrated probability vs
                                                                  realized outcome, full
                                                                  test period, Section 6.1)
          report/figures/fig_threshold_vs_realized.png         (market-data-only check:
                                                                  realized 13-week move vs
                                                                  the trailing-vol threshold,
                                                                  no model involved, Section 3.9)
          report/figures/fig_directional_accuracy.png          (per-model check: on weeks a
                                                                  breach happened, did the
                                                                  model favour the correct
                                                                  side, Section 6.1)

Run: python code/09a_deal_rolldown_probability.py   (after 05, 07, 13, 01d)
"""

import sys
import os
import pickle
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
DEAL_MATURITY = 3        # 3M contract -- the deal's tenor
DEAL_HORIZON = 13        # 13 weeks -- the deal's own 3M tenor (Section 3.9)
VOL_LOOKBACK_WEEKS = 52  # trailing 12-month window for the threshold itself
WEEKS_PER_MONTH = 52 / 12
TAU_REM = DEAL_MATURITY - DEAL_HORIZON / WEEKS_PER_MONTH   # = 0 exactly: the
                                                            # contract reaches
                                                            # its own closing
                                                            # date at h=13W
N_SAMPLES = 1000
EPS = 1e-4
TARIFF_SHOCK_START = pd.Timestamp("2025-04-11")


def ns_loadings_vec(tau_scalar, lambda_ns):
    """
    Loading row [L, S, C] at a single (possibly fractional) maturity.

    Handles tau -> 0 explicitly. The ratio form (1-exp(-lt))/lt is a 0/0
    indeterminate at lt=0; its analytic limit is 1 (L'Hopital), giving
    S(0)=1 and C(0)=S(0)-exp(0)=0. This matters here: at the deal's own
    3M tenor (h=13W), the contract's remaining maturity is exactly zero,
    and the fitted price there collapses to exp(beta0+beta1), the
    level-plus-slope, spot-like value NS implies at the very front of
    the curve, not an undefined quantity.
    """
    lt = tau_scalar / lambda_ns
    l_load = 1.0
    if lt < 1e-8:
        s_load = 1.0
        c_load = 0.0
    else:
        s_load = (1 - np.exp(-lt)) / lt
        c_load = s_load - np.exp(-lt)
    return np.array([l_load, s_load, c_load])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(p_raw, y):
    """Same constrained (a>=0) Platt fit as 14_platt_calibration.py -- see
    that file's docstring for the current, verified rationale (a structural
    safeguard across the full fitting grid, not an empirical fix for
    AR-Direct specifically -- an unconstrained fit does not actually invert
    AR-Direct's own ranking under the current bootstrap pipeline)."""
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


def plot_model_alert_comparison(calib_df: pd.DataFrame, out_dir: str) -> None:
    """
    8 panels: 4 models x 2 directions (up, down). Each panel plots the
    calibrated probability of clearing the volatility threshold for
    EVERY week of the test period (165 points), colored by what actually
    happened that week: green if the move cleared the threshold, red if
    it did not. A thin grey line connects consecutive weeks for visual
    continuity only; the colour, not the line, carries the information.

    Colouring every single week the same way, not just the weeks that
    cleared the threshold, makes hits and misses directly comparable: a
    model with real skill should show green points sitting higher on the
    page than red points, not scattered through them. Marking only the
    hits (e.g. with a triangle) would leave misses looking like unexplained
    points on the line, with nothing to compare them against.

    Rows = direction (up, down); columns = model, so a reader compares
    across models within the same row at a glance. The vertical dashed
    line marks the tariff-shock boundary (11 April 2025): the calibration
    is fit on dates before it only, so probabilities after it are shown
    but should be read as extrapolated, not validated (Section 4.5).

    No horizontal reference line is drawn (an empirical-base-rate line was
    tried and removed: it added clutter without adding information here).

    LP+MACRO'S DOWN PANEL SHOWS NARROW, NOT FLAT, VARIATION (see module
    docstring point 2 on the LME Cash realised/threshold side). Its raw
    down-direction probability has
    weak discriminating skill (AUC 0.57, modestly above chance rather than
    at it), so the constrained Platt fit (a >= 0, Section 4.5) finds a
    small positive slope (a=0.34) instead of collapsing fully to a=0. The
    calibrated output does vary (std 0.015, range 4.8%-14.0%), but the
    resulting Brier Skill Score is still essentially zero (0.0007): the
    small amount of raw ranking ability survives calibration, but
    contributes negligible practical skill over the historical base rate.
    A model with slightly more than nothing to say, not nothing at all,
    but still not enough to act on.
    """
    model_ids = sorted(MODEL_NAMES.keys())
    directions = ["up", "down"]
    fig, axes = plt.subplots(len(directions), len(model_ids), figsize=(16, 7),
                              sharex=True, sharey=True)

    for i, direction in enumerate(directions):
        for j, m in enumerate(model_ids):
            ax = axes[i, j]
            sub = calib_df[calib_df.model == m].sort_values("date")
            p_col, act_col = f"p_{direction}_cal", f"actual_{direction}"
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
                ax.set_ylabel(f"{'Up' if direction == 'up' else 'Down'}\nP(calibrated)")
            if i == len(directions) - 1:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.tick_params(axis="x", rotation=45)

    axes[0, 0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Calibrated probability vs realized outcome, by model and direction\n"
                 "(green = threshold cleared that week, red = not; dashed line: tariff-shock boundary)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_model_alert_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved: fig_model_alert_comparison.png")


def plot_directional_accuracy(calib_df: pd.DataFrame, out_dir: str) -> None:
    """
    A different, more basic question than the figure above: not "was the
    probability high enough," but "on weeks a breach actually happened,
    did the model favour the side that turned out to be right." For each
    model, both P(up) and P(down) are plotted together, and every week a
    breach actually occurred is marked: green if that model's own higher
    probability that week pointed toward the direction that happened,
    black if it pointed the wrong way.

    This is a directional hit-rate check, not a probability-calibration
    check: a model can pass this test by favouring "up" with 51% against
    49% and still be right, or fail it while favouring the correct side
    only weakly. It answers "which way did the model lean" not "how
    confident was it," which is exactly why it is kept as a separate
    figure from fig_model_alert_comparison rather than merged into it.
    """
    model_ids = sorted(MODEL_NAMES.keys())
    fig, axes = plt.subplots(1, len(model_ids), figsize=(18, 4.5), sharex=True, sharey=True)

    summary_rows = []
    for j, m in enumerate(model_ids):
        ax = axes[j]
        sub = calib_df[calib_df.model == m].sort_values("date")
        valid = sub.dropna(subset=["p_up_cal", "p_down_cal"])

        ax.plot(valid["date"], valid["p_up_cal"], color="firebrick", linewidth=0.7, alpha=0.6, label="P(up)")
        ax.plot(valid["date"], valid["p_down_cal"], color="steelblue", linewidth=0.7, alpha=0.6, label="P(down)")

        favors_up = valid["p_up_cal"] > valid["p_down_cal"]

        up_events = valid[valid["actual_up"] == 1]
        up_correct = favors_up.loc[up_events.index]
        ax.scatter(up_events.loc[up_correct, "date"], up_events.loc[up_correct, "p_up_cal"],
                   color="forestgreen", marker="^", s=50, zorder=5, label="Correct side favoured")
        ax.scatter(up_events.loc[~up_correct, "date"], up_events.loc[~up_correct, "p_up_cal"],
                   color="black", marker="^", s=50, zorder=5, label="Wrong side favoured")

        down_events = valid[valid["actual_down"] == 1]
        down_correct = (~favors_up).loc[down_events.index]
        ax.scatter(down_events.loc[down_correct, "date"], down_events.loc[down_correct, "p_down_cal"],
                   color="forestgreen", marker="v", s=50, zorder=5)
        ax.scatter(down_events.loc[~down_correct, "date"], down_events.loc[~down_correct, "p_down_cal"],
                   color="black", marker="v", s=50, zorder=5)

        ax.axvline(TARIFF_SHOCK_START, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(MODEL_NAMES[m], fontsize=10)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45)

        n_events = len(up_events) + len(down_events)
        n_correct = int(up_correct.sum()) + int(down_correct.sum())
        summary_rows.append({
            "model": MODEL_NAMES[m], "N breach weeks": n_events, "N correct side": n_correct,
            "directional accuracy": round(n_correct / n_events, 3) if n_events else np.nan,
        })

    axes[0].set_ylabel("Calibrated probability")
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Directional accuracy: did the model favour the side that actually happened?\n"
                 "(green = correct side favoured, black = wrong side favoured; shown only on weeks a breach occurred)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_directional_accuracy.png"), dpi=150)
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    print("  Saved: fig_directional_accuracy.png")
    print(summary_df.to_string(index=False))


def plot_threshold_vs_realized(curves: pd.DataFrame, cash: pd.Series,
                                sigma_series: pd.Series, out_dir: str) -> None:
    """
    For every week the trailing-volatility threshold can be computed
    (from ~2007, once a full 52-week lookback exists, through the last
    date with 13 weeks of forward data), plot the deal's own threshold
    (+/- sigma_13W,t, shaded band) against the realized move from LP3 at
    t (entry) to LME Cash at t+13W (closing) -- the same threshold and the
    same realized outcome the probability comparison is built on, shown
    directly, not reduced to a hit/miss indicator.

    Unlike the model comparison figure, this one is NOT restricted to the
    2023-2026 out-of-sample test period: no model or forecast is involved
    here, only observed prices and a trailing statistic, so the full
    available history is used to show how often, and by how much, the
    realized move has actually cleared its own trailing-volatility range
    -- including 2008, 2020 and 2022, the case-study episodes discussed
    in Chapter 4.
    """
    entry_col = f"m{DEAL_MATURITY:02d}"

    records = []
    for t in curves.index:
        sigma_t = sigma_series.get(t, np.nan)
        if pd.isna(sigma_t) or entry_col not in curves.columns:
            continue
        p0 = curves.loc[t, entry_col]
        if pd.isna(p0) or p0 <= 0:
            continue
        t_ph = t + pd.Timedelta(weeks=DEAL_HORIZON)
        idx = cash.index.get_indexer([t_ph], method="nearest", tolerance=pd.Timedelta(days=3))[0]
        if idx == -1:
            continue
        actual_cash = cash.loc[cash.index[idx]]
        if pd.isna(actual_cash):
            continue
        records.append({"date": t, "sigma_pct": sigma_t * 100,
                         "realized_pct": (actual_cash - p0) / p0 * 100})

    df = pd.DataFrame(records).sort_values("date")
    if df.empty:
        print("  No data for fig_threshold_vs_realized.png -- skipping")
        return

    breach_up = df["realized_pct"] > df["sigma_pct"]
    breach_down = df["realized_pct"] < -df["sigma_pct"]
    colors = np.where(breach_up, "firebrick", np.where(breach_down, "steelblue", "black"))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(df["date"], -df["sigma_pct"], df["sigma_pct"], color="grey", alpha=0.25,
                     label=r"$\pm 1$ trailing-vol threshold ($\hat\sigma_{13W,t}$)")
    ax.plot(df["date"], df["realized_pct"], color="grey", linewidth=0.5, alpha=0.6, zorder=2)
    ax.scatter(df["date"], df["realized_pct"], c=colors, s=14, zorder=5,
               label="Realized 13-week move (LME Cash at t+13W vs LP3 at t)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(TARIFF_SHOCK_START, color="black", linestyle="--", linewidth=0.7, alpha=0.5,
               label="Tariff-shock boundary")
    ax.set_ylabel("Percent move")
    ax.set_title("Realized move vs the deal's own trailing-volatility threshold, weekly, full sample\n"
                 "(red = cleared upside, blue = cleared downside, black = within band)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_threshold_vs_realized.png"), dpi=150)
    plt.close(fig)

    n_breach_up, n_breach_down = int(breach_up.sum()), int(breach_down.sum())
    print(f"  Saved: fig_threshold_vs_realized.png  "
          f"({n_breach_up} upside breaches, {n_breach_down} downside breaches, N={len(df)})")


def compute_for_model(model_id, factor_forecasts, rolling_resid, curves,
                       cash, sigma_series, loadings_rem, rng):
    """
    Run the simulate-vs-realize comparison for one model, over every test
    date it has a forecast for, using each date's own trailing-volatility
    threshold rather than a fixed percentage.
    """
    entry_col = f"m{DEAL_MATURITY:02d}"   # LPY at t -- the deal's actual entry price

    ff = factor_forecasts[(factor_forecasts.model == model_id) &
                           (factor_forecasts.horizon == DEAL_HORIZON)].copy()
    ff["date"] = pd.to_datetime(ff["date"])
    ff["t_plus_h"] = pd.to_datetime(ff["t_plus_h"])

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
            continue  # no full trailing 52-week window yet at this date

        mu_beta = row[["beta0_hat", "beta1_hat", "beta2_hat"]].values.astype(float)
        resid_pool = rolling_resid.get((model_id, DEAL_HORIZON, t))
        if resid_pool is None or len(resid_pool) < 5:
            continue

        idx = rng.integers(0, len(resid_pool), size=N_SAMPLES)
        sim_betas = mu_beta[None, :] + resid_pool[idx]              # (N,3)
        sim_log_prices = sim_betas @ loadings_rem                    # (N,)
        sim_prices = np.exp(sim_log_prices)

        # Realized/exit side: the directly observed LME Cash price at t+h,
        # standing in for the deal's own contract at its own zero-remaining-
        # maturity closing date (see module docstring, point 2). Cash is a
        # true spot series with no rolling-maturity drift, so it is read
        # directly at t+h, unlike the earlier LP1-based approximation this
        # replaces.
        idx_nearest = cash.index.get_indexer([t_ph], method="nearest",
                                              tolerance=pd.Timedelta(days=3))[0]
        if idx_nearest == -1:
            actual_price = np.nan
        else:
            t_ph_actual = cash.index[idx_nearest]
            actual_price = cash.loc[t_ph_actual]
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

        # Expected-variation comparison (Section 6.2 redesign): the naive
        # side (sigma_t) is a standard deviation of trailing LME Cash
        # returns. For a genuine like-for-like comparison, the model side
        # is also a standard deviation, of the simulated forecast
        # distribution's returns, not a probability-scaled dollar figure.
        # Both are then checked against what the realised move actually
        # was, over the same tenor, to see which one is closer to the
        # truth rather than which one is smaller.
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
        })

    return pd.DataFrame(records)


def main():
    print("=" * 78)
    print("Roll-down probability construction, stylised deal -- volatility-threshold design")
    print(f"tau={DEAL_MATURITY}M, h={DEAL_HORIZON}W, tau_rem={TAU_REM:.4f}M")
    print("Threshold: trailing realised volatility on LME Cash, scaled to the deal's own tenor")
    print("Realized side: directly observed LME Cash price at t+h (not a curve reconstruction)")
    print("=" * 78)

    factor_forecasts = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "factor_forecasts.parquet"))

    with open(os.path.join(PROCESSED_DATA_DIR, "rolling_lp_residuals.pkl"), "rb") as f:
        pkl = pickle.load(f)
    rolling_resid = pkl["resid"]

    curves = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "curves.parquet"))
    curves.index = pd.to_datetime(curves.index)

    cash_path = os.path.join(PROCESSED_DATA_DIR, "lme_copper_cash.parquet")
    cash = pd.read_parquet(cash_path)["cash"]
    cash.index = pd.to_datetime(cash.index)
    print(f"Loaded LME Cash series: {cash.notna().sum()} weekly observations "
          f"({cash.dropna().index.min().date()} to {cash.dropna().index.max().date()})")

    # Threshold volatility is now computed on Cash too, not LP1, so the
    # "will the market move more than its own recent normal range implies"
    # question (module docstring, point 1) is measured on the same series
    # as the realized outcome it is compared against.
    sigma_series = trailing_vol_reference(cash, horizon_weeks=DEAL_HORIZON,
                                           lookback_weeks=VOL_LOOKBACK_WEEKS)

    loadings_rem = ns_loadings_vec(TAU_REM, LAMBDA_NS)   # fixed, reused every date and model
    rng = np.random.default_rng(42)

    raw_df = pd.concat([
        compute_for_model(m, factor_forecasts, rolling_resid, curves, cash, sigma_series, loadings_rem, rng)
        for m in MODEL_NAMES
    ], ignore_index=True)

    raw_path = os.path.join(PROCESSED_DATA_DIR, "deal_rolldown_probabilities.parquet")
    raw_df.to_parquet(raw_path)
    print(f"\nSaved: {raw_path}  ({len(raw_df)} rows, {raw_df['model'].nunique()} models)")

    # ── AUC comparison across all 4 models: does AR-Direct's edge from the ──
    # fixed-threshold convention (Chapter 5) survive under this threshold too?
    print("\n--- AUC comparison across all 4 models (volatility-threshold design) ---")
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

    # ── Expected-variation accuracy: is the naive trailing-vol range, or ──
    # each model's own forecast-implied volatility, closer to what the
    # market actually did over the same tenor? Both are standard
    # deviations of returns, so this compares like with like, rather than
    # comparing a raw range against a probability-scaled dollar figure.
    print("\n--- Expected-variation accuracy: naive trailing-vol vs model forecast (all vs realised) ---")
    for m in MODEL_NAMES:
        sub = raw_df[raw_df.model == m]
        for period in ["stable", "shock"]:
            g = sub[sub.period == period].dropna(subset=["naive_error", "model_error"])
            if len(g) < 10:
                print(f"  {MODEL_NAMES[m]:12s} {period:8s}: N={len(g)} (too few to test)")
                continue
            print(f"  {MODEL_NAMES[m]:12s} {period:8s}: "
                  f"naive MAE={g['naive_error'].mean():.4f}  model MAE={g['model_error'].mean():.4f}  "
                  f"(model closer on {(g['model_error'] < g['naive_error']).mean()*100:.0f}% of dates, N={len(g)})")

    # ── Platt calibration, per model, same convention as 14_platt_calibration.py: ──
    # fit on stable period only, apply everywhere, flag domain.
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
            bs_cal = float(np.mean((sigmoid(a * logit(valid_fit[p_col].values) + b)
                                     - valid_fit[act_col].values) ** 2))
            bs_clim = base_rate * (1 - base_rate)
            bss_raw = 1 - bs_raw / bs_clim if bs_clim > 0 else np.nan
            bss_cal = 1 - bs_cal / bs_clim if bs_clim > 0 else np.nan
            coef_rows.append({"model": MODEL_NAMES[m], "direction": direction, "a": round(a, 4), "b": round(b, 4),
                               "N_fit": len(valid_fit), "base_rate": round(base_rate, 4),
                               "bss_raw": round(bss_raw, 4), "bss_cal": round(bss_cal, 4)})

    coef_df = pd.DataFrame(coef_rows)
    print("\n--- Platt calibration, per model (stable period fit, volatility-threshold design) ---")
    print(coef_df.to_string(index=False))

    calib_path = os.path.join(PROCESSED_DATA_DIR, "deal_rolldown_calibrated.parquet")
    calib_df.to_parquet(calib_path)
    print(f"\nSaved: {calib_path}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("\nGenerating figures...")
    plot_model_alert_comparison(calib_df, FIGURES_DIR)
    plot_directional_accuracy(calib_df, FIGURES_DIR)
    plot_threshold_vs_realized(curves, cash, sigma_series, FIGURES_DIR)

    print("\nNext: rerun code/09_stylised_deal.py, which reads DEAL_MODEL_ID's calibrated")
    print("columns from this file for the liquidity-buffer illustration.")


if __name__ == "__main__":
    main()
