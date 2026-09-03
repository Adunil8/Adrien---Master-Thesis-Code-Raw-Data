"""
02b_ns_log.py - Nelson-Siegel fit on log-transformed prices.

Phase 2b, Configuration 2 of 4.
Fits NS model on log(price), following Bianchi et al. (2023).
Log transformation improves stability across price regimes and yields
dimensionless, percentage-scale factors. This is the configuration used
in the main analysis (02_nelson_siegel.py).

Outputs RMSE in BOTH log space (fit space) and original price space ($/t)
so results are directly comparable to the raw-price configurations.

Outputs:
  data/processed/comparison/ns_log_factors.parquet   - β₀, β₁, β₂, RMSE (both spaces), R²
  data/processed/comparison/ns_log_metrics.json      - summary statistics for comparison table
  report/figures/fig_nslog_fit_example.png
  report/figures/fig_nslog_rmse.png

Run: python code/02b_ns_log.py
     (from thesis/ root with .venv active)
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import config
from utils import ns_loadings

# ── Grid parameters ────────────────────────────────────────────────────────────
GRID_MIN = 1.0
GRID_MAX = 60.0
GRID_N   = 200


# ── Helpers ────────────────────────────────────────────────────────────────────

def tau_from_columns(curves: pd.DataFrame) -> np.ndarray:
    return np.array(
        [float(c[1:]) for c in curves.columns if c.startswith("m") and c[1:].isdigit()]
    )


def grid_search(train_log: np.ndarray, tau: np.ndarray) -> tuple[float, float]:
    """
    Find λ minimising mean cross-sectional RMSE over training dates.
    RMSE computed in log space (the space in which fitting is done).
    """
    grid = np.linspace(GRID_MIN, GRID_MAX, GRID_N)
    best_lam, best_rmse = grid[0], np.inf
    for lam in grid:
        X      = ns_loadings(tau, lam)
        pinv_X = np.linalg.pinv(X)
        betas  = train_log @ pinv_X.T
        fitted = betas @ X.T
        mean_rmse = float(np.sqrt(np.mean((train_log - fitted) ** 2, axis=1)).mean())
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_lam  = lam
    return best_lam, best_rmse


def fit_all_dates(
    log_arr: np.ndarray,
    raw_arr: np.ndarray,
    tau: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit NS(log) to every date simultaneously.

    Parameters
    ----------
    log_arr : (n_dates, n_tau) - log-transformed prices
    raw_arr : (n_dates, n_tau) - original prices ($/t), for price-space metrics

    Returns
    -------
    betas          : (n_dates, 3)    - β₀, β₁, β₂  (in log space)
    rmse_logspace  : (n_dates,)      - RMSE in log space (fit space)
    rmse_pricespace: (n_dates,)      - RMSE in $/t (back-transformed)
    r2             : (n_dates,)      - R² in price space
    fitted_log     : (n_dates, n_tau) - fitted values in log space
    """
    X      = ns_loadings(tau, lam)
    pinv_X = np.linalg.pinv(X)
    betas  = log_arr @ pinv_X.T          # (n_dates, 3)
    fitted_log = betas @ X.T             # (n_dates, n_tau)  - in log space

    resid_log      = log_arr - fitted_log
    rmse_logspace  = np.sqrt(np.mean(resid_log ** 2, axis=1))

    # Back-transform to price space for comparable RMSE
    fitted_price   = np.exp(fitted_log)   # (n_dates, n_tau) - in $/t
    resid_price    = raw_arr - fitted_price
    rmse_pricespace = np.sqrt(np.mean(resid_price ** 2, axis=1))

    # R² in price space
    row_mean = raw_arr.mean(axis=1, keepdims=True)
    ss_tot   = np.sum((raw_arr - row_mean) ** 2, axis=1)
    ss_res   = np.sum(resid_price ** 2, axis=1)
    r2       = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    return betas, rmse_logspace, rmse_pricespace, r2, fitted_log


# ── Figures ────────────────────────────────────────────────────────────────────

def plot_fit_example(
    curves: pd.DataFrame,
    betas_df: pd.DataFrame,
    tau: np.ndarray,
    lam: float,
    out_dir: str,
) -> None:
    """Observed vs NS-fitted curves (price space) for three representative dates.

    Dates are fixed, not picked by idxmax/idxmin on this fit's own beta1
    (see 02b_ns_raw.py's plot_fit_example for the full account of why: that
    selection landed on 2021-10-29 as "Contango", an ill-conditioned fit
    right next to this configuration's own worst-RMSE date, on a day the
    observed prices were actually in backwardation). Same three fixed
    dates as 02b_ns_raw.py, so the two figures are directly comparable on
    identical calendar dates: 2006-01-06 (most extreme backwardation),
    2021-08-20 (within 1 USD/t of flat), 2024-07-12 (deepest contango in
    the full sample, also cited in Section 3.1)."""
    date_back = pd.Timestamp("2006-01-06")
    date_flat = pd.Timestamp("2021-08-20")
    date_cont = pd.Timestamp("2024-07-12")

    tau_fine = np.linspace(tau.min(), tau.max(), 300)
    cases = [
        (date_back, "Backwardation", config.COLORS["backwardation"]),
        (date_flat, "Neutral / Flat",  config.COLORS["neutral"]),
        (date_cont, "Contango",         config.COLORS["contango"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (date, label, color) in zip(axes, cases):
        b = betas_df.loc[date, ["beta0", "beta1", "beta2"]].values
        X_fine      = ns_loadings(tau_fine, lam)
        fit_log     = X_fine @ b
        fit_price   = np.exp(fit_log)                  # back to price space for display
        actual_price = curves.loc[date].values

        ax.plot(tau_fine, fit_price, color=color, linewidth=2, label="NS fit (log→exp)")
        ax.scatter(tau, actual_price, color="black", s=25, zorder=5, label="Observed")
        if label == "Neutral / Flat":
            # Autoscaling on a ~$2/t range makes noise look like curvature.
            # Fixed range keeps this panel visually flat, matching what it is.
            ax.set_ylim(9020, 9080)
            ax.set_yticks([9030, 9040, 9050, 9060, 9070])
        ax.set_title(
            f"{label}\n{date.strftime('%Y-%m-%d')}\n"
            r"$\beta_0$" f"={b[0]:.3f}  " r"$\beta_1$" f"={b[1]:.3f}  " r"$\beta_2$" f"={b[2]:.3f}",
            fontsize=8,
        )
        ax.set_xlabel("Maturity (months)")
        ax.set_ylabel("Price ($/t)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "NS (log prices) - Cross-Sectional Fit Examples [displayed in price space]",
        fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_nslog_fit_example.png")
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def plot_rmse(
    rmse_log_ts: pd.Series,
    rmse_price_ts: pd.Series,
    out_dir: str,
) -> None:
    """Two-panel: RMSE in log space (top) and price space $/t (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for ax, ts, ylabel, color in [
        (ax1, rmse_log_ts,   "RMSE (log space)",  "steelblue"),
        (ax2, rmse_price_ts, "RMSE ($/t)",        "seagreen"),
    ]:
        mu, sig = float(ts.mean()), float(ts.std())
        ax.plot(ts.index, ts, linewidth=0.8, color=color, label="RMSE")
        ax.axhline(mu,         color="navy",   linestyle=":", linewidth=0.8,
                   label=f"Mean  {mu:.5f}")
        ax.axhline(mu + 2*sig, color="tomato", linestyle="--", linewidth=1.0,
                   label=f"Mean+2σ  {mu+2*sig:.5f}")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    ax1.set_title("NS (log) - Fit RMSE per Date", fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_nslog_rmse.png")
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    comp_dir = os.path.join(config.PROCESSED_DATA_DIR, "comparison")
    os.makedirs(comp_dir, exist_ok=True)
    plt.style.use(config.PLOT_STYLE)

    print("=" * 60)
    print("02b_ns_log.py - Nelson-Siegel on log prices")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    curves = pd.read_parquet(os.path.join(config.PROCESSED_DATA_DIR, "curves.parquet"))
    col_names = [f"m{m:02d}" for m in config.MATURITIES_MONTHS]
    curves = curves[col_names]
    tau    = tau_from_columns(curves)

    raw_arr = curves.values.astype(float)
    log_arr = np.log(raw_arr)              # NS fitted on log prices
    train_mask = curves.index <= pd.Timestamp(config.TRAIN_END)
    train_log  = log_arr[train_mask]

    print(f"Data:     {len(curves)} obs × {len(tau)} maturities")
    print(f"Maturities: {list(tau.astype(int))} months")
    print(f"Training: {train_mask.sum()} obs (≤ {config.TRAIN_END})")
    print(f"Test:     {(~train_mask).sum()} obs (≥ {config.TEST_START})")

    # ── Grid search for optimal λ ──────────────────────────────────────────────
    print(f"\nGrid search NS (log): λ ∈ [{GRID_MIN}, {GRID_MAX}], {GRID_N} points ...")
    optimal_lam, train_rmse_log = grid_search(train_log, tau)
    print(f"  λ*              = {optimal_lam:.4f}")
    print(f"  Train RMSE (log)= {train_rmse_log:.6f}")

    # ── Fit all dates ──────────────────────────────────────────────────────────
    print("\nFitting NS (log) for all dates ...")
    betas, rmse_log, rmse_price, r2, _fitted_log = fit_all_dates(
        log_arr, raw_arr, tau, optimal_lam
    )

    betas_df      = pd.DataFrame(betas, index=curves.index,
                                 columns=["beta0", "beta1", "beta2"])
    rmse_log_ts   = pd.Series(rmse_log,   index=curves.index, name="rmse_logspace")
    rmse_price_ts = pd.Series(rmse_price, index=curves.index, name="rmse_pricespace")
    r2_ts         = pd.Series(r2,         index=curves.index, name="r2")

    # ── Summary statistics ─────────────────────────────────────────────────────
    mean_price      = float(curves["m01"].loc[:config.TRAIN_END].mean())

    # Log-space metrics
    mean_rmse_log   = float(rmse_log_ts.mean())
    median_rmse_log = float(rmse_log_ts.median())
    max_rmse_log    = float(rmse_log_ts.max())

    # Price-space metrics (comparable to raw models)
    mean_rmse_p     = float(rmse_price_ts.mean())
    median_rmse_p   = float(rmse_price_ts.median())
    max_rmse_p      = float(rmse_price_ts.max())
    max_rmse_date   = rmse_price_ts.idxmax().strftime("%Y-%m-%d")
    mean_r2         = float(r2_ts.mean())
    rmse_pct        = mean_rmse_p / mean_price * 100
    pct_2sigma      = float(
        (rmse_price_ts > mean_rmse_p + 2 * float(rmse_price_ts.std())).mean() * 100
    )
    pct_above_2pct  = float((rmse_price_ts > mean_price * 0.02).mean() * 100)

    # Cross-sectional AIC in log space (for within-scale NS vs SV comparison)
    n_tau     = len(tau)
    k_ns      = 3
    rss_log   = (rmse_log ** 2) * n_tau
    aic_arr   = 2 * k_ns + n_tau * np.log(rss_log / n_tau)
    mean_aic  = float(aic_arr.mean())

    print(f"\n  ── Log-space metrics (fit space) ──")
    print(f"  Mean RMSE (log)      = {mean_rmse_log:.6f}")
    print(f"  Median RMSE (log)    = {median_rmse_log:.6f}")
    print(f"  Max RMSE (log)       = {max_rmse_log:.6f}")
    print(f"\n  ── Price-space metrics (comparable to raw models) ──")
    print(f"  Mean RMSE ($/t)      = {mean_rmse_p:.4f}")
    print(f"  Median RMSE ($/t)    = {median_rmse_p:.4f}")
    print(f"  Max RMSE ($/t)       = {max_rmse_p:.4f}  ({max_rmse_date})")
    print(f"  RMSE as % of price   = {rmse_pct:.3f}%  (ref: mean m01 = {mean_price:.0f} $/t)")
    print(f"  Mean R²              = {mean_r2:.6f}")
    print(f"  Mean AIC (log space) = {mean_aic:.2f}")
    print(f"  % dates > mean+2σ    = {pct_2sigma:.1f}%")
    print(f"  % dates > 2% price   = {pct_above_2pct:.1f}%")

    # ── Save factors parquet ───────────────────────────────────────────────────
    out_df = betas_df.copy()
    out_df["rmse_logspace"]   = rmse_log_ts
    out_df["rmse_pricespace"] = rmse_price_ts
    out_df["r2"]              = r2_ts
    out_df["model"]           = "NS"
    out_df["scale"]           = "log"
    parquet_path = os.path.join(comp_dir, "ns_log_factors.parquet")
    out_df.to_parquet(parquet_path)
    print(f"\nSaved factors: {parquet_path}")

    # ── Save metrics JSON ──────────────────────────────────────────────────────
    metrics = {
        "label":                   "NS (log)",
        "model":                   "NS",
        "scale":                   "log",
        "n_params":                k_ns,
        "lambda1":                 round(optimal_lam, 4),
        "lambda2":                 None,
        "train_rmse_fitspace":     round(float(train_rmse_log), 8),
        "mean_rmse_fitspace":      round(mean_rmse_log, 8),
        "mean_rmse_pricespace":    round(mean_rmse_p, 6),
        "median_rmse_pricespace":  round(median_rmse_p, 6),
        "max_rmse_pricespace":     round(max_rmse_p, 6),
        "max_rmse_date":           max_rmse_date,
        "mean_r2_pricespace":      round(mean_r2, 6),
        "mean_aic_fitspace":       round(mean_aic, 4),
        "pct_flagged_2sigma":      round(pct_2sigma, 2),
        "pct_above_2pct_price":    round(pct_above_2pct, 2),
        "mean_rmse_pct_of_price":  round(rmse_pct, 4),
        "mean_price_m01_train":    round(mean_price, 2),
        "n_obs":                   int(len(curves)),
        "n_maturities":            int(n_tau),
    }
    json_path = os.path.join(comp_dir, "ns_log_metrics.json")
    with open(json_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved metrics: {json_path}")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\nGenerating figures ...")
    plot_fit_example(curves, betas_df, tau, optimal_lam, config.FIGURES_DIR)
    plot_rmse(rmse_log_ts, rmse_price_ts, config.FIGURES_DIR)

    print("\n02b_ns_log.py complete.")
    print("Next: python code/02b_sv_raw.py")


if __name__ == "__main__":
    main()
