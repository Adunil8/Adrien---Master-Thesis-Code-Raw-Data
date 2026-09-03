"""
08_probability_outputs.py - Phase 5: Probabilistic forecast outputs.

DESIGN
------
For each test date t and horizon h, the LP regression gives a point forecast
μ_h = [β̂₀, β̂₁, β̂₂].  Training residuals give Σ_h (3×3 factor covariance).
Sampling K=1000 vectors from N(μ_h, Σ_h) and passing each through the NS
formula reconstructs 1000 plausible copper curves at t+h.

Percentile bands across those 1000 curves at each maturity τ give:
  - The fan chart (visualisation)
  - P(ΔF(τ, t+h) > k%)  for each threshold k  (probability output)

The uncertainty varies naturally by maturity because the NS loading vector
w(τ) = [1, L(τ,λ), C(τ,λ)] weighs the three factors differently:
  τ=1M  is dominated by β₁ (slope) → high uncertainty when slope uncertain
  τ=6M  is dominated by β₀ (level) → lower uncertainty at long end

OUTPUTS
-------
  data/processed/lp_sigma.pkl              - Σ_h matrices per (model, horizon)
  data/processed/curve_bands.parquet       - pct bands per (date, model, horizon, maturity)
  data/processed/probability_outputs.parquet - P(ΔF > k%) per (date, model, horizon, maturity, threshold)
  report/figures/fig_fan_*.png             - fan chart figures

Run: .venv/bin/python code/08_probability_outputs.py
"""

import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from config import (
    MATURITIES_MONTHS, LAMBDA_NS, LAMBDA_EW,
    TRAIN_END, TEST_START,
    FORECAST_HORIZONS_WEEKS, HORIZON_LABELS,
    MARGIN_THRESHOLDS,
    PROCESSED_DATA_DIR, FIGURES_DIR,
    MACRO_CHANGE_COLS, MACRO_LEVEL_COLS, MACRO_VAR_SPEC,
)
from utils import ns_curve, build_lp_matrices, fit_lp

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "text.usetex":       False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

N_SAMPLES  = 1000   # Monte Carlo draws per forecast
PCTS       = [0.5, 2.5, 5, 15, 25, 50, 75, 85, 95, 97.5, 99.5]
# Probability interval bands shown in fan chart:
# (lower_pct, upper_pct, label, alpha)
BANDS = [
    (37.5, 62.5, "25%", 0.65),
    (25,   75,   "50%", 0.45),
    (12.5, 87.5, "75%", 0.28),
    (5,    95,   "90%", 0.14),
]
BAND_COLOR = "#1a4a8a"   # navy blue - all bands share this hue, alpha varies

MODEL_LABELS = {0: "Random Walk", 1: "AR-Direct", 2: "LP + Macro", 3: "LP + Macro + EW"}
MODEL_SHORT  = {0: "RW", 1: "AR", 2: "LP", 3: "LP+EW"}


# ── Step 1: Rolling Σ_h - expanding window consistent with 07's backtest ──────

def _sigma_from_window(factors_w, macro_w, macro_cols, p, q, lambda_ew,
                       horizons):
    """
    Compute {(model, h): Σ_h} for one training window.
    Called for each test date - factors_w / macro_w include all data up to t.
    """
    factor_cols = ["beta0", "beta1", "beta2"]
    sigma = {}

    for h in horizons:
        # Model 0: RW - covariance of actual h-step factor changes
        combined_f = factors_w[factor_cols].dropna()
        Y_rw = combined_f.shift(-h).dropna()
        X_rw = combined_f.loc[Y_rw.index]
        res0 = Y_rw.values - X_rw.values
        sigma[(0, h)] = np.cov(res0.T) if len(res0) > 3 else np.eye(3) * 1e-4

        # Models 1–3: LP residuals
        X_ar, Y_ar, _ = build_lp_matrices(factors_w, macro_w, p, 0, h,
                                           macro_cols=None)
        X_lp, Y_lp, _ = build_lp_matrices(factors_w, macro_w, p, q, h,
                                           macro_cols=macro_cols)

        for mid, X, Y in [(1, X_ar, Y_ar), (2, X_lp, Y_lp)]:
            if X is None or len(X) < 10:
                sigma[(mid, h)] = np.eye(3) * 1e-4
                continue
            coef = fit_lp(X, Y)
            res  = Y - X @ coef
            sigma[(mid, h)] = np.cov(res.T) if len(res) > 3 else np.eye(3) * 1e-4

        if X_lp is not None and len(X_lp) >= 10:
            T_w = len(X_lp)
            w   = lambda_ew ** np.arange(T_w - 1, -1, -1)
            coef_ew = fit_lp(X_lp, Y_lp, weights=w)
            res_ew  = Y_lp - X_lp @ coef_ew
            sigma[(3, h)] = (np.cov(res_ew.T) if len(res_ew) > 3
                             else np.eye(3) * 1e-4)
        else:
            sigma[(3, h)] = sigma.get((2, h), np.eye(3) * 1e-4)

    return sigma


def compute_rolling_sigma(factors, macro, macro_cols, p, q, lambda_ew,
                          horizons, test_dates):
    """
    Compute Σ_h for every test date using the expanding window [2006 → t].

    Returns dict keyed by (model_id, h, date) → np.ndarray (3, 3).
    This ensures the uncertainty bands at each date reflect the same
    training window used for the LP point forecasts in 07.
    """
    rolling_sigma = {}
    n = len(test_dates)
    print(f"Computing rolling Σ_h for {n} test dates × "
          f"{len(horizons)} horizons × 4 models...")

    for i, t in enumerate(test_dates):
        if i % 30 == 0:
            print(f"  {i}/{n}...", end=" ", flush=True)
        t = pd.Timestamp(t)
        sigma_t = _sigma_from_window(
            factors.loc[:t], macro.loc[:t],
            macro_cols, p, q, lambda_ew, horizons,
        )
        for (mid, h), mat in sigma_t.items():
            rolling_sigma[(mid, h, t)] = mat

    print("done.")
    return rolling_sigma


def get_sigma(rolling_sigma, model_id, h, date):
    """Look up Σ_h for a given (model, horizon, date). Fallback to identity."""
    key = (model_id, h, pd.Timestamp(date))
    return rolling_sigma.get(key, np.eye(3) * 0.01)


# ── Step 2: Sample curves and compute percentile bands ───────────────────────

def sample_curves(mu_beta, sigma_h, tau, lambda_ns, n_samples=N_SAMPLES,
                  rng=None):
    """
    Draw n_samples from N(mu_beta, sigma_h) and reconstruct NS curves.

    Returns:
        curves : (n_samples, len(tau)) array of prices in $/t
    """
    if rng is None:
        rng = np.random.default_rng(42)
    try:
        betas = rng.multivariate_normal(mu_beta, sigma_h, size=n_samples)
    except np.linalg.LinAlgError:
        # Sigma not PSD - use diagonal approximation
        betas = rng.multivariate_normal(mu_beta, np.diag(np.diag(sigma_h)),
                                        size=n_samples)
    curves = np.exp(np.array([ns_curve(b, tau, lambda_ns) for b in betas]))
    return curves   # (n_samples, len(tau))


def percentile_bands(curves, pcts=PCTS):
    """
    Compute percentile bands across Monte Carlo samples.
    Returns (len(pcts), len(tau)) array.
    """
    return np.percentile(curves, pcts, axis=0)


# ── Step 3: Fan chart figure ──────────────────────────────────────────────────

YMIN, YMAX = 6000, 12000   # fixed y-axis range - consistent across all panels


def plot_fan_chart(date_t, horizons_to_plot, tau, lambda_ns,
                   factors_df, copper_df, factor_forecasts_df, rolling_sigma,
                   model_id=2, figsize=(12, 5.2), save_path=None):
    """
    One figure with len(horizons_to_plot) panels side by side.
    Each panel: current curve (black solid) + forecasted bands at horizon h.
    Actual realised curve at t+h shown as thin grey line.
    Legend placed below the figure - never overlaps the bands.
    All panels share the same y-axis range (YMIN, YMAX).
    """
    rng  = np.random.default_rng(42)
    n_h  = len(horizons_to_plot)
    fig, axes = plt.subplots(1, n_h, figsize=figsize, sharey=True,
                             facecolor="white")
    if n_h == 1:
        axes = [axes]

    # Current observed curve at t
    if date_t in copper_df.index:
        current_prices = copper_df.loc[date_t].values[:len(tau)]
    else:
        current_prices = None

    for ax, h in zip(axes, horizons_to_plot):
        if date_t not in factors_df.index:
            ax.set_title(f"h={h}W\n(no data)")
            continue

        mask = ((factor_forecasts_df["date"]    == date_t) &
                (factor_forecasts_df["horizon"] == h) &
                (factor_forecasts_df["model"]   == model_id))
        sub = factor_forecasts_df[mask]
        if sub.empty:
            ax.set_title(f"h={h}W\n(no forecast)")
            continue

        mu_beta = sub[["beta0_hat", "beta1_hat", "beta2_hat"]].values[0]
        sig_h   = get_sigma(rolling_sigma, model_id, h, date_t)

        curves = sample_curves(mu_beta, sig_h, tau, lambda_ns, rng=rng)
        bands  = percentile_bands(curves, [b[0] for b in BANDS] +
                                          [b[1] for b in BANDS] + [50])
        n_b    = len(BANDS)
        lowers = bands[:n_b]
        uppers = bands[n_b:2*n_b]
        median = np.exp(ns_curve(mu_beta, tau, lambda_ns))

        # Draw bands widest-first so narrower bands sit on top
        for i, (lo_pct, hi_pct, lbl, alpha) in enumerate(reversed(BANDS)):
            idx = len(BANDS) - 1 - i
            ax.fill_between(tau, lowers[idx], uppers[idx],
                            color=BAND_COLOR, alpha=alpha,
                            linewidth=0, zorder=2)

        ax.plot(tau, median, color=BAND_COLOR, linewidth=1.6,
                linestyle="--", zorder=4)

        if current_prices is not None:
            ax.plot(tau, current_prices[:len(tau)], color="#111111",
                    linewidth=1.8, linestyle="-", zorder=5)

        t_ph_vals = sub["t_plus_h"].values
        if len(t_ph_vals) > 0:
            t_ph = pd.Timestamp(t_ph_vals[0])
            if t_ph in copper_df.index:
                realized = copper_df.loc[t_ph].values[:len(tau)]
                ax.plot(tau, realized[:len(tau)], color="#777777",
                        linewidth=1.0, linestyle=":", zorder=3)

        h_lbl = HORIZON_LABELS.get(h, f"{h}W")
        ax.set_title(f"Horizon: {h_lbl}", fontsize=9.5, fontweight="bold", pad=6)
        ax.set_xlabel("Time to maturity (months)", fontsize=8.0)
        ax.set_xticks(tau)
        ax.set_xticklabels([f"{int(m)}M" for m in tau], fontsize=7.5)
        ax.tick_params(labelsize=7.5)
        ax.set_ylim(YMIN, YMAX)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"))
        ax.grid(axis="y", linewidth=0.3, color="#dddddd", zorder=0)
        ax.set_axisbelow(True)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_linewidth(0.6)

    axes[0].set_ylabel("Price (USD/tonne)", fontsize=8.0)

    # Legend below the figure - never inside the panels
    legend_handles = [
        mlines.Line2D([], [], color="#111111", lw=1.8, label="Observed curve (t)"),
        mlines.Line2D([], [], color=BAND_COLOR, lw=1.6, ls="--",
                      label="Median forecast"),
        mlines.Line2D([], [], color="#777777", lw=1.0, ls=":",
                      label="Realised curve (t+h)"),
    ]
    for lo_pct, hi_pct, lbl, alpha in BANDS:
        legend_handles.append(matplotlib.patches.Patch(
            facecolor=BAND_COLOR, alpha=alpha, label=f"{lbl} interval"))
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=4,
               fontsize=7.5, frameon=False)

    date_str = pd.Timestamp(date_t).strftime("%d %b %Y")
    fig.suptitle(
        f"LME Copper Futures Curve - Probabilistic Forecast at {date_str}\n"
        f"Model: {MODEL_LABELS[model_id]}   "
        f"Bands: 25 / 50 / 75 / 90% prediction intervals   "
        f"(N = {N_SAMPLES} Monte Carlo draws)",
        fontsize=8.5, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {save_path}")
    return fig


# ── Step 4: Model comparison fan chart (one horizon, all models) ──────────────

def plot_model_comparison(date_t, horizon, tau, lambda_ns,
                          factors_df, copper_df, factor_forecasts_df, rolling_sigma,
                          figsize=(13, 6.5), save_path=None):
    """
    2×2 grid: one panel per model, same horizon h.
    Shows 50% and 90% bands + median. All panels share the same y-axis range.
    Legend placed below the figure - never overlaps the bands.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True,
                             facecolor="white")
    axes = axes.flatten()
    rng  = np.random.default_rng(42)

    if date_t in copper_df.index:
        current_prices = copper_df.loc[date_t].values[:len(tau)]
    else:
        current_prices = None

    mask_h  = ((factor_forecasts_df["horizon"] == horizon) &
               (factor_forecasts_df["date"]    == date_t))
    sub_all = factor_forecasts_df[mask_h]

    sub_any = sub_all[sub_all["model"] == 2]
    realized = None
    if not sub_any.empty:
        t_ph = pd.Timestamp(sub_any["t_plus_h"].values[0])
        if t_ph in copper_df.index:
            realized = copper_df.loc[t_ph].values[:len(tau)]

    for ax, model_id in zip(axes, [0, 1, 2, 3]):
        sub = sub_all[sub_all["model"] == model_id]
        if sub.empty:
            ax.set_title(MODEL_LABELS[model_id])
            continue

        mu_beta = sub[["beta0_hat", "beta1_hat", "beta2_hat"]].values[0]
        sig_h   = get_sigma(rolling_sigma, model_id, horizon, date_t)
        curves  = sample_curves(mu_beta, sig_h, tau, lambda_ns, rng=rng)

        bands = percentile_bands(curves, [b[0] for b in BANDS] +
                                         [b[1] for b in BANDS])
        n_b    = len(BANDS)
        lowers = bands[:n_b]
        uppers = bands[n_b:]
        med    = np.exp(ns_curve(mu_beta, tau, lambda_ns))

        for i, (lo_pct, hi_pct, lbl, alpha) in enumerate(reversed(BANDS)):
            idx = len(BANDS) - 1 - i
            ax.fill_between(tau, lowers[idx], uppers[idx],
                            color=BAND_COLOR, alpha=alpha, lw=0)
        ax.plot(tau, med, color=BAND_COLOR, lw=1.5, ls="--")
        if current_prices is not None:
            ax.plot(tau, current_prices[:len(tau)], color="#111111", lw=1.6)
        if realized is not None:
            ax.plot(tau, realized[:len(tau)], color="#777777", lw=1.0, ls=":")

        ax.set_title(MODEL_LABELS[model_id], fontsize=8.5, fontweight="bold")
        ax.set_xticks(tau)
        ax.set_xticklabels([f"{int(m)}M" for m in tau], fontsize=7.5)
        ax.tick_params(labelsize=7.5)
        ax.set_ylim(YMIN, YMAX)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"))
        ax.grid(axis="y", lw=0.3, color="#dddddd")
        ax.set_axisbelow(True)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_linewidth(0.6)
        if model_id in [0, 2]:
            ax.set_ylabel("Price (USD/tonne)", fontsize=8.0)
        ax.set_xlabel("Time to maturity (months)", fontsize=8.0)

    # Legend below the figure
    legend_handles = [
        mlines.Line2D([], [], color="#111111", lw=1.8, label="Observed curve (t)"),
        mlines.Line2D([], [], color=BAND_COLOR, lw=1.5, ls="--", label="Median forecast"),
        mlines.Line2D([], [], color="#777777", lw=1.0, ls=":", label="Realised curve (t+h)"),
    ]
    for lo_pct, hi_pct, lbl, alpha in BANDS:
        legend_handles.append(matplotlib.patches.Patch(
            facecolor=BAND_COLOR, alpha=alpha, label=f"{lbl} interval"))
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=4,
               fontsize=7.5, frameon=False)

    date_str = pd.Timestamp(date_t).strftime("%d %b %Y")
    h_lbl    = HORIZON_LABELS.get(horizon, f"{horizon}W")
    fig.suptitle(
        f"LME Copper - {h_lbl} Forecast at {date_str}",
        fontsize=9.0, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {save_path}")
    return fig


# ── Step 5: Rolling probability outputs ──────────────────────────────────────

def compute_probability_outputs(factor_forecasts_df, copper_df, rolling_sigma,
                                 tau, lambda_ns, thresholds, n_samples=N_SAMPLES):
    """
    For each (date, horizon, model, maturity, threshold):
      p_up   = P(F(τ,t+h) > F(τ,t) × (1+k%))
      p_down = P(F(τ,t+h) < F(τ,t) × (1-k%))
      actual_up   = 1 if realised price actually exceeded threshold
      actual_down = 1 if realised price actually fell below threshold

    Returns DataFrame for Brier score computation.
    """
    rng     = np.random.default_rng(42)
    records = []

    dates   = factor_forecasts_df["date"].unique()
    n_total = len(dates)
    print(f"\nComputing probability outputs for {n_total} test dates...")

    for i, date_t in enumerate(sorted(dates)):
        if i % 30 == 0:
            print(f"  {i}/{n_total}...", end=" ", flush=True)

        if date_t not in copper_df.index:
            continue
        current_prices = copper_df.loc[date_t].values

        for h in FORECAST_HORIZONS_WEEKS:
            mask = ((factor_forecasts_df["date"]    == date_t) &
                    (factor_forecasts_df["horizon"] == h))
            sub = factor_forecasts_df[mask]

            # Find realised prices at t+h
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
                sig_h   = get_sigma(rolling_sigma, model_id, h, date_t)

                # Sample curves at t+h
                curves = sample_curves(mu_beta, sig_h, tau, lambda_ns, rng=rng)
                # curves: (n_samples, len(tau))

                for j, mat in enumerate(MATURITIES_MONTHS):
                    if j >= len(current_prices) or np.isnan(current_prices[j]):
                        continue
                    p0 = current_prices[j]      # price at t
                    pr = (realized_prices[j]
                          if j < len(realized_prices) else np.nan)

                    for k in thresholds:
                        up_barrier   = p0 * (1 + k)
                        down_barrier = p0 * (1 - k)
                        sample_col   = curves[:, j]

                        p_up   = float(np.mean(sample_col > up_barrier))
                        p_down = float(np.mean(sample_col < down_barrier))

                        act_up   = (1 if (not np.isnan(pr) and pr > up_barrier)
                                    else 0) if not np.isnan(pr) else np.nan
                        act_down = (1 if (not np.isnan(pr) and pr < down_barrier)
                                    else 0) if not np.isnan(pr) else np.nan

                        records.append({
                            "date":      date_t,
                            "horizon":   h,
                            "model":     model_id,
                            "maturity":  mat,
                            "threshold": k,
                            "p_up":      round(p_up,   4),
                            "p_down":    round(p_down, 4),
                            "actual_up": act_up,
                            "actual_down": act_down,
                        })

    print("done.")
    return pd.DataFrame(records)


def compute_brier_scores(prob_df):
    """Brier score per (model, horizon, maturity, threshold, direction)."""
    rows = []
    for (model, h, mat, k), grp in prob_df.groupby(
            ["model", "horizon", "maturity", "threshold"]):
        for direction in ["up", "down"]:
            p_col   = f"p_{direction}"
            act_col = f"actual_{direction}"
            valid   = grp[[p_col, act_col]].dropna()
            if len(valid) < 5:
                continue
            bs = float(np.mean((valid[p_col] - valid[act_col]) ** 2))
            # Climatological Brier score (baseline: always predict base rate)
            base_rate = float(valid[act_col].mean())
            bs_clim   = float(base_rate * (1 - base_rate))
            bss = 1 - bs / bs_clim if bs_clim > 0 else np.nan
            rows.append({
                "model":     model,
                "horizon":   h,
                "maturity":  mat,
                "threshold": k,
                "direction": direction,
                "brier_score":  round(bs,   5),
                "brier_skill":  round(bss,  4),
                "base_rate":    round(base_rate, 4),
                "N":         len(valid),
            })
    return pd.DataFrame(rows)


# ── Step 6: Reliability diagram (calibration) ────────────────────────────────

def plot_reliability_diagram(prob_df, horizon, threshold, maturity,
                              direction="up", n_bins=10,
                              figsize=(9, 4.5), save_path=None):
    """
    Reliability (calibration) diagram for P(ΔF > +k%) or P(ΔF < -k%).

    Each model's predictions are binned into n_bins equal-width probability
    intervals. For each bin, plot mean(predicted p) vs mean(actual outcome).
    The diagonal = perfect calibration. Below diagonal = overconfident.
    Above diagonal = underconfident.

    A sharpness histogram (inset) shows whether predictions are concentrated
    near 0/1 or clustered around the base rate.

    Parameters
    ----------
    horizon     : int    - forecast horizon in weeks (e.g. 13 = 3M)
    threshold   : float  - e.g. 0.08 for 8%
    maturity    : int    - maturity in months (e.g. 3)
    direction   : 'up' or 'down'
    n_bins      : int    - number of probability bins (default 10)
    """
    p_col   = f"p_{direction}"
    act_col = f"actual_{direction}"
    dir_lbl = f"ΔF > +{int(threshold*100)}%" if direction == "up" else \
              f"ΔF < −{int(threshold*100)}%"

    sub = prob_df[
        (prob_df["horizon"]   == horizon) &
        (prob_df["threshold"] == threshold) &
        (prob_df["maturity"]  == maturity)
    ].dropna(subset=[p_col, act_col]).copy()

    if sub.empty:
        print(f"  [reliability] No data for h={horizon}, k={threshold}, mat={maturity}")
        return None

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_mids  = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    model_colors = {0: "#999999", 1: "#e07b39", 2: "#2a6099", 3: "#1a4a8a"}
    fig, (ax_main, ax_hist) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.08}
    )

    for model_id in [0, 1, 2, 3]:
        grp = sub[sub["model"] == model_id]
        if grp.empty:
            continue

        p_vals   = grp[p_col].values
        act_vals = grp[act_col].values.astype(float)

        # Bin by predicted probability
        bin_idx   = np.digitize(p_vals, bin_edges[1:-1])   # 0 … n_bins-1
        frac_obs  = []
        mean_pred = []
        counts    = []

        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() < 3:
                frac_obs.append(np.nan)
                mean_pred.append(bin_mids[b])
                counts.append(0)
            else:
                frac_obs.append(float(np.mean(act_vals[mask])))
                mean_pred.append(float(np.mean(p_vals[mask])))
                counts.append(int(mask.sum()))

        frac_obs  = np.array(frac_obs)
        mean_pred = np.array(mean_pred)
        ok        = ~np.isnan(frac_obs)

        ax_main.plot(
            mean_pred[ok], frac_obs[ok],
            color=model_colors[model_id], lw=1.6, marker="o",
            markersize=4.5, label=MODEL_LABELS[model_id],
            zorder=3,
        )

        # Sharpness histogram (one per model, stacked)
        ax_hist.hist(p_vals, bins=bin_edges, color=model_colors[model_id],
                     alpha=0.35, density=True, edgecolor="none")

    # Perfect calibration reference
    ax_main.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#aaaaaa",
                 label="Perfect calibration", zorder=1)

    # Base rate reference
    base_rate = float(sub[act_col].mean())
    ax_main.axhline(base_rate, ls=":", lw=0.8, color="#aaaaaa", zorder=1)
    ax_main.text(0.02, base_rate + 0.02, f"Base rate = {base_rate:.2f}",
                 fontsize=7.0, color="#888888", transform=ax_main.transData)

    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlabel("Predicted probability", fontsize=9)
    ax_main.set_ylabel("Observed frequency", fontsize=9)
    ax_main.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax_main.tick_params(labelsize=7.5)
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.grid(lw=0.3, color="#dddddd")
    ax_main.set_axisbelow(True)

    ax_hist.set_xlabel("Predicted prob.", fontsize=8)
    ax_hist.set_ylabel("Density", fontsize=8)
    ax_hist.set_xlim(0, 1)
    ax_hist.tick_params(labelsize=7)
    ax_hist.set_title("Sharpness", fontsize=8)
    ax_hist.grid(lw=0.3, color="#dddddd", axis="y")

    h_lbl = HORIZON_LABELS.get(horizon, f"{horizon}W")
    fig.suptitle(
        f"Reliability Diagram - P({dir_lbl}), {maturity}M maturity, {h_lbl} horizon",
        fontsize=9.5, fontweight="bold"
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {save_path}")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("Phase 5 - Probabilistic Forecast Outputs")
    print("=" * 60)

    # Load inputs
    factors = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet")
    )[["beta0", "beta1", "beta2"]]
    copper  = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "curves.parquet"))
    macro_changes = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "macro_changes.parquet"))
    macro_levels  = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "macro.parquet"))
    macro = pd.concat([
        macro_changes[MACRO_CHANGE_COLS],
        macro_levels[MACRO_LEVEL_COLS],
    ], axis=1)
    macro_cols = MACRO_VAR_SPEC

    factor_forecasts = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "factor_forecasts.parquet"))
    lag_sel = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "lag_selection.parquet"))

    opt_path = os.path.join(PROCESSED_DATA_DIR, "lambda_ew_optimal.parquet")
    lambda_ew = (float(pd.read_parquet(opt_path)["lambda_ew_optimal"].iloc[0])
                 if os.path.exists(opt_path) else LAMBDA_EW)

    p   = int(lag_sel["recommended"].iloc[0])
    q   = 1
    tau = np.array(MATURITIES_MONTHS, dtype=float)

    print(f"p={p}, q={q}, λ_EW={lambda_ew:.4f}")
    print(f"Maturities: {MATURITIES_MONTHS}  |  Horizons: {FORECAST_HORIZONS_WEEKS}W")

    # ── Step 1: Rolling Σ_h - expanding window consistent with 07 ───────────
    test_dates = sorted(factor_forecasts["date"].unique())
    sigma_path = os.path.join(PROCESSED_DATA_DIR, "rolling_lp_sigma.pkl")
    if os.path.exists(sigma_path):
        with open(sigma_path, "rb") as f:
            rolling_sigma = pickle.load(f)
        print(f"Loaded rolling Σ_h from {sigma_path}  ({len(rolling_sigma)} matrices)")
    else:
        rolling_sigma = compute_rolling_sigma(
            factors, macro, macro_cols, p, q, lambda_ew,
            FORECAST_HORIZONS_WEEKS, test_dates,
        )
        with open(sigma_path, "wb") as f:
            pickle.dump(rolling_sigma, f)
        print(f"Saved rolling Σ_h → {sigma_path}")

    # Print σ diagnostics at the tariff shock date (Apr 2025) for LP+Macro
    target_tariff = pd.Timestamp("2025-04-11")
    diag_date = pd.Timestamp(
        min(test_dates, key=lambda d: abs(pd.Timestamp(d) - target_tariff)))
    print(f"\nFactor forecast std √Σ_h at {diag_date.strftime('%d %b %Y')} "
          f"- LP+Macro (rolling):")
    print(f"  {'Horizon':8s} σ(β₀)    σ(β₁)    σ(β₂)")
    for h in FORECAST_HORIZONS_WEEKS:
        sig  = get_sigma(rolling_sigma, 2, h, diag_date)
        stds = np.sqrt(np.diag(sig))
        h_lbl = HORIZON_LABELS.get(h, f"{h}W")
        print(f"  {h_lbl:8s} {stds[0]:.5f}  {stds[1]:.5f}  {stds[2]:.5f}")

    # ── Step 2: Fan chart - case study dates ──────────────────────────────────
    n_td = len(test_dates)
    case_study_candidates = {
        "start":  test_dates[0],
        "mid":    test_dates[n_td // 2],
        "tariff": diag_date,
    }

    h_plot = [4, 13, 26]   # 1M, 3M, 6M

    for case_name, date_t in case_study_candidates.items():
        date_t = pd.Timestamp(date_t)
        print(f"\nFan chart: {case_name} - {date_t.strftime('%d %b %Y')}")

        fname = f"fig_fan_{case_name}_lp.png"
        fig = plot_fan_chart(
            date_t, h_plot, tau, LAMBDA_NS,
            factors, copper, factor_forecasts, rolling_sigma,
            model_id=2,
            figsize=(13, 4.5),
            save_path=os.path.join(FIGURES_DIR, fname),
        )
        plt.close(fig)

        fname2 = f"fig_fan_{case_name}_model_comp_3m.png"
        fig2 = plot_model_comparison(
            date_t, horizon=13, tau=tau, lambda_ns=LAMBDA_NS,
            factors_df=factors, copper_df=copper,
            factor_forecasts_df=factor_forecasts, rolling_sigma=rolling_sigma,
            figsize=(12, 5.5),
            save_path=os.path.join(FIGURES_DIR, fname2),
        )
        plt.close(fig2)

    # ── Step 3: Rolling probability outputs ───────────────────────────────────
    prob_path = os.path.join(PROCESSED_DATA_DIR, "probability_outputs.parquet")
    # Always recompute - rolling sigma has changed
    prob_df = compute_probability_outputs(
        factor_forecasts, copper, rolling_sigma, tau, LAMBDA_NS,
        thresholds=MARGIN_THRESHOLDS,
    )
    prob_df.to_parquet(prob_path)
    print(f"Saved probability outputs → {prob_path}")

    # ── Step 4: Brier scores ──────────────────────────────────────────────────
    brier_df = compute_brier_scores(prob_df)
    brier_path = os.path.join(PROCESSED_DATA_DIR, "brier_scores.parquet")
    brier_df.to_parquet(brier_path)
    print(f"Saved Brier scores → {brier_path}")

    # Print Brier skill score summary (3M maturity, 8% threshold, upside)
    print("\n" + "─" * 65)
    print("Brier Skill Score - P(ΔF > +8%), 3M maturity")
    print("(positive BSS = beats climatological baseline)")
    print("─" * 65)
    model_labels = {0: "RW", 1: "AR-Direct", 2: "LP+Macro", 3: "LP+Macro+EW"}
    for h in [4, 13, 26]:
        sub = brier_df[
            (brier_df["maturity"]  == 3) &
            (brier_df["threshold"] == 0.08) &
            (brier_df["direction"] == "up") &
            (brier_df["horizon"]   == h)
        ].sort_values("model")
        if sub.empty:
            continue
        h_lbl = HORIZON_LABELS.get(h, f"{h}W")
        print(f"\n{h_lbl} horizon:")
        for _, row in sub.iterrows():
            m = model_labels.get(row["model"], str(row["model"]))
            bss_str = (f"{row['brier_skill']:+.3f}"
                       if not np.isnan(row["brier_skill"]) else "  n/a")
            print(f"  {m:18s}  BS={row['brier_score']:.4f}  "
                  f"BSS={bss_str}  base={row['base_rate']:.2f}  N={row['N']}")

    # ── Step 5: Reliability diagrams (Test G2) ────────────────────────────────
    print("\nGenerating reliability diagrams...")
    rel_configs = [
        (4,  0.08, 3,  "up",   "fig_rel_1m_8pct_up.png"),
        (13, 0.08, 3,  "up",   "fig_rel_3m_8pct_up.png"),
        (13, 0.08, 3,  "down", "fig_rel_3m_8pct_down.png"),
        (26, 0.08, 3,  "up",   "fig_rel_6m_8pct_up.png"),
    ]
    for h, k, mat, dirn, fname in rel_configs:
        fig = plot_reliability_diagram(
            prob_df, horizon=h, threshold=k, maturity=mat,
            direction=dirn, n_bins=10,
            save_path=os.path.join(FIGURES_DIR, fname),
        )
        if fig is not None:
            plt.close(fig)

    print("\nPhase 5 complete.")
    print("Fan charts saved to report/figures/fig_fan_*.png")
    print("Reliability diagrams saved to report/figures/fig_rel_*.png")
    print("Next: run 09_stylised_deal.py")


if __name__ == "__main__":
    main()
