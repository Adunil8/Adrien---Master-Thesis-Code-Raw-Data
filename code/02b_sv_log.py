"""
02b_sv_log.py - Svensson (1994) fit on log-transformed prices.

Phase 2b, Configuration 4 of 4.
Combines the Svensson four-factor model with log-price transformation.
Provides the richest fit of the four configurations tested, at the cost
of two extra parameters relative to NS (log) and reduced interpretability of β₃.

Results are reported in BOTH log space (fit space) and original price space ($/t)
to enable direct comparison across all four configurations.

Outputs:
  data/processed/comparison/sv_log_factors.parquet   - β₀–β₃, RMSE (both spaces), R²
  data/processed/comparison/sv_log_metrics.json      - summary statistics for comparison table
  report/figures/fig_svlog_fit_example.png
  report/figures/fig_svlog_rmse.png

Run: python code/02b_sv_log.py
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
from utils import sv_loadings

# ── Grid parameters ────────────────────────────────────────────────────────────
GRID_MIN = 1.0
GRID_MAX = 60.0
GRID_N   = 30
MIN_SEP  = 1.5


# ── Helpers ────────────────────────────────────────────────────────────────────

def tau_from_columns(curves: pd.DataFrame) -> np.ndarray:
    return np.array(
        [float(c[1:]) for c in curves.columns if c.startswith("m") and c[1:].isdigit()]
    )


def grid_search_2d(
    train_log: np.ndarray,
    tau: np.ndarray,
) -> tuple[float, float, float]:
    """
    Find (λ₁, λ₂) minimising mean cross-sectional RMSE in log space.
    Skips pairs where |λ₁ − λ₂| < MIN_SEP or rank(X) < 4.
    """
    grid = np.linspace(GRID_MIN, GRID_MAX, GRID_N)
    best_lam1, best_lam2, best_rmse = grid[0], grid[-1], np.inf
    n_evaluated = 0

    for lam1 in grid:
        for lam2 in grid:
            if abs(lam1 - lam2) < MIN_SEP:
                continue
            X = sv_loadings(tau, lam1, lam2)
            if np.linalg.matrix_rank(X) < 4:
                continue
            pinv_X    = np.linalg.pinv(X)
            betas     = train_log @ pinv_X.T
            fitted    = betas @ X.T
            mean_rmse = float(np.sqrt(np.mean((train_log - fitted) ** 2, axis=1)).mean())
            n_evaluated += 1
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_lam1 = lam1
                best_lam2 = lam2

    print(f"    Evaluated {n_evaluated} / {GRID_N**2} (λ₁,λ₂) pairs")
    return best_lam1, best_lam2, best_rmse


def fit_all_dates(
    log_arr: np.ndarray,
    raw_arr: np.ndarray,
    tau: np.ndarray,
    lam1: float,
    lam2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit Svensson(log) to every date simultaneously.

    Parameters
    ----------
    log_arr : (n_dates, n_tau) - log-transformed prices
    raw_arr : (n_dates, n_tau) - original prices ($/t)

    Returns
    -------
    betas           : (n_dates, 4)    - β₀, β₁, β₂, β₃  (in log space)
    rmse_logspace   : (n_dates,)      - RMSE in log space (fit space)
    rmse_pricespace : (n_dates,)      - RMSE in $/t (back-transformed)
    r2              : (n_dates,)      - R² in price space
    fitted_log      : (n_dates, n_tau) - fitted values in log space
    """
    X          = sv_loadings(tau, lam1, lam2)   # (n_tau, 4)
    pinv_X     = np.linalg.pinv(X)
    betas      = log_arr @ pinv_X.T             # (n_dates, 4)
    fitted_log = betas @ X.T                    # (n_dates, n_tau)  log space

    resid_log      = log_arr - fitted_log
    rmse_logspace  = np.sqrt(np.mean(resid_log ** 2, axis=1))

    # Back-transform to price space
    fitted_price    = np.exp(fitted_log)
    resid_price     = raw_arr - fitted_price
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
    lam1: float,
    lam2: float,
    out_dir: str,
) -> None:
    """Observed vs Svensson-fitted curves (price space) for three representative dates."""
    train = betas_df.loc[:config.TRAIN_END].dropna()
    date_back = train["beta1"].idxmax()
    date_cont = train["beta1"].idxmin()
    remaining = train.drop([date_back, date_cont])
    date_flat = remaining["beta1"].abs().idxmin()

    tau_fine = np.linspace(tau.min(), tau.max(), 300)
    cases = [
        (date_back, "Backwardation", config.COLORS["backwardation"]),
        (date_flat, "Neutral / Flat",  config.COLORS["neutral"]),
        (date_cont, "Contango",         config.COLORS["contango"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (date, label, color) in zip(axes, cases):
        b = betas_df.loc[date, ["beta0", "beta1", "beta2", "beta3"]].values
        X_fine      = sv_loadings(tau_fine, lam1, lam2)
        fit_price   = np.exp(X_fine @ b)          # back to price space for display
        actual_price = curves.loc[date].values

        ax.plot(tau_fine, fit_price, color=color, linewidth=2, label="SV fit (log→exp)")
        ax.scatter(tau, actual_price, color="black", s=25, zorder=5, label="Observed")
        ax.set_title(
            f"{label}\n{date.strftime('%Y-%m-%d')}\n"
            f"β₀={b[0]:.3f}  β₁={b[1]:.3f}\nβ₂={b[2]:.3f}  β₃={b[3]:.3f}",
            fontsize=7.5,
        )
        ax.set_xlabel("Maturity (months)")
        ax.set_ylabel("Price ($/t)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Svensson (log) - Cross-Sectional Fit Examples  [λ₁*={lam1:.2f}, λ₂*={lam2:.2f}]",
        fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_svlog_fit_example.png")
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
        (ax1, rmse_log_ts,   "RMSE (log space)",  "darkorange"),
        (ax2, rmse_price_ts, "RMSE ($/t)",        "#7c3aed"),
    ]:
        mu, sig = float(ts.mean()), float(ts.std())
        ax.plot(ts.index, ts, linewidth=0.8, color=color, label="RMSE")
        ax.axhline(mu,         color="navy",   linestyle=":", linewidth=0.8,
                   label=f"Mean  {mu:.6f}")
        ax.axhline(mu + 2*sig, color="tomato", linestyle="--", linewidth=1.0,
                   label=f"Mean+2σ  {mu+2*sig:.6f}")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    ax1.set_title("Svensson (log) - Fit RMSE per Date", fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_svlog_rmse.png")
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
    print("02b_sv_log.py - Svensson on log prices")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    curves = pd.read_parquet(os.path.join(config.PROCESSED_DATA_DIR, "curves.parquet"))
    col_names = [f"m{m:02d}" for m in config.MATURITIES_MONTHS]
    curves = curves[col_names]
    tau    = tau_from_columns(curves)

    raw_arr = curves.values.astype(float)
    log_arr = np.log(raw_arr)
    train_mask = curves.index <= pd.Timestamp(config.TRAIN_END)
    train_log  = log_arr[train_mask]

    print(f"Data:     {len(curves)} obs × {len(tau)} maturities")
    print(f"Maturities: {list(tau.astype(int))} months")
    print(f"Training: {train_mask.sum()} obs (≤ {config.TRAIN_END})")
    print(f"Test:     {(~train_mask).sum()} obs (≥ {config.TEST_START})")

    # ── 2D grid search for optimal (λ₁, λ₂) ──────────────────────────────────
    print(f"\nGrid search SV (log): {GRID_N}×{GRID_N} pairs, λ ∈ [{GRID_MIN}, {GRID_MAX}]")
    print(f"  Min separation |λ₁ − λ₂| > {MIN_SEP} enforced ...")
    optimal_lam1, optimal_lam2, train_rmse_log = grid_search_2d(train_log, tau)
    print(f"  λ₁*             = {optimal_lam1:.4f}")
    print(f"  λ₂*             = {optimal_lam2:.4f}")
    print(f"  Train RMSE (log)= {train_rmse_log:.8f}")

    # ── Fit all dates ──────────────────────────────────────────────────────────
    print("\nFitting Svensson (log) for all dates ...")
    betas, rmse_log, rmse_price, r2, _fitted_log = fit_all_dates(
        log_arr, raw_arr, tau, optimal_lam1, optimal_lam2
    )

    betas_df = pd.DataFrame(
        betas, index=curves.index,
        columns=["beta0", "beta1", "beta2", "beta3"],
    )
    rmse_log_ts   = pd.Series(rmse_log,   index=curves.index, name="rmse_logspace")
    rmse_price_ts = pd.Series(rmse_price, index=curves.index, name="rmse_pricespace")
    r2_ts         = pd.Series(r2,         index=curves.index, name="r2")

    # ── Summary statistics ─────────────────────────────────────────────────────
    mean_price      = float(curves["m01"].loc[:config.TRAIN_END].mean())

    mean_rmse_log   = float(rmse_log_ts.mean())
    median_rmse_log = float(rmse_log_ts.median())
    max_rmse_log    = float(rmse_log_ts.max())

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

    n_tau    = len(tau)
    k_sv     = 4
    rss_log  = (rmse_log ** 2) * n_tau
    aic_arr  = 2 * k_sv + n_tau * np.log(rss_log / n_tau)
    mean_aic = float(aic_arr.mean())

    print(f"\n  ── Log-space metrics (fit space) ──")
    print(f"  Mean RMSE (log)      = {mean_rmse_log:.8f}")
    print(f"  Median RMSE (log)    = {median_rmse_log:.8f}")
    print(f"  Max RMSE (log)       = {max_rmse_log:.8f}")
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
    out_df["model"]           = "Svensson"
    out_df["scale"]           = "log"
    parquet_path = os.path.join(comp_dir, "sv_log_factors.parquet")
    out_df.to_parquet(parquet_path)
    print(f"\nSaved factors: {parquet_path}")

    # ── Save metrics JSON ──────────────────────────────────────────────────────
    metrics = {
        "label":                   "Svensson (log)",
        "model":                   "Svensson",
        "scale":                   "log",
        "n_params":                k_sv,
        "lambda1":                 round(optimal_lam1, 4),
        "lambda2":                 round(optimal_lam2, 4),
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
    json_path = os.path.join(comp_dir, "sv_log_metrics.json")
    with open(json_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved metrics: {json_path}")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\nGenerating figures ...")
    plot_fit_example(curves, betas_df, tau, optimal_lam1, optimal_lam2, config.FIGURES_DIR)
    plot_rmse(rmse_log_ts, rmse_price_ts, config.FIGURES_DIR)

    print("\n02b_sv_log.py complete.")
    print("Next: python code/02b_model_comparison.py  (generates comparison table)")


if __name__ == "__main__":
    main()
