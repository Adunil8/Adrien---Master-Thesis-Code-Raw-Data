"""
02b_sv_raw.py - Svensson (1994) fit on raw (level) prices.

Phase 2b, Configuration 3 of 4.
Extends Nelson-Siegel with a second curvature term (β₃) at a second decay rate λ₂,
allowing the model to capture a second hump in the futures curve at a different
maturity. This provides additional fit flexibility at the cost of one extra parameter.

2D grid search for optimal (λ₁, λ₂) on training data, enforcing minimum separation
|λ₁ − λ₂| > 1.5 to avoid near-rank-deficiency in the loading matrix.

Outputs:
  data/processed/comparison/sv_raw_factors.parquet   - β₀–β₃, RMSE ($/t), R² per date
  data/processed/comparison/sv_raw_metrics.json      - summary statistics for comparison table
  report/figures/fig_svraw_fit_example.png
  report/figures/fig_svraw_rmse.png

Run: python code/02b_sv_raw.py
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
GRID_MIN    = 1.0
GRID_MAX    = 60.0
GRID_N      = 30          # 30×30 = 900 pairs - fast with vectorised fit
MIN_SEP     = 1.5         # |λ₁ − λ₂| must exceed this to avoid near-singular loadings


# ── Helpers ────────────────────────────────────────────────────────────────────

def tau_from_columns(curves: pd.DataFrame) -> np.ndarray:
    return np.array(
        [float(c[1:]) for c in curves.columns if c.startswith("m") and c[1:].isdigit()]
    )


def grid_search_2d(
    train_arr: np.ndarray,
    tau: np.ndarray,
) -> tuple[float, float, float]:
    """
    Find (λ₁, λ₂) minimising mean cross-sectional RMSE over training dates.
    Skips pairs where |λ₁ − λ₂| < MIN_SEP or rank(X) < 4.

    Returns (optimal_lam1, optimal_lam2, optimal_mean_rmse)
    """
    grid = np.linspace(GRID_MIN, GRID_MAX, GRID_N)
    best_lam1, best_lam2, best_rmse = grid[0], grid[-1], np.inf
    n_evaluated = 0

    for lam1 in grid:
        for lam2 in grid:
            if abs(lam1 - lam2) < MIN_SEP:
                continue
            X = sv_loadings(tau, lam1, lam2)     # (n_tau, 4)
            if np.linalg.matrix_rank(X) < 4:
                continue
            pinv_X    = np.linalg.pinv(X)
            betas     = train_arr @ pinv_X.T
            fitted    = betas @ X.T
            mean_rmse = float(np.sqrt(np.mean((train_arr - fitted) ** 2, axis=1)).mean())
            n_evaluated += 1
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_lam1 = lam1
                best_lam2 = lam2

    print(f"    Evaluated {n_evaluated} / {GRID_N**2} (λ₁,λ₂) pairs")
    return best_lam1, best_lam2, best_rmse


def fit_all_dates(
    curves_arr: np.ndarray,
    tau: np.ndarray,
    lam1: float,
    lam2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit Svensson(raw) to every date simultaneously.

    Returns
    -------
    betas      : (n_dates, 4)    - β₀, β₁, β₂, β₃
    fitted_arr : (n_dates, n_tau)
    rmse       : (n_dates,)     - RMSE in price space ($/t)
    r2         : (n_dates,)     - R² in price space
    """
    X      = sv_loadings(tau, lam1, lam2)   # (n_tau, 4)
    pinv_X = np.linalg.pinv(X)             # (4, n_tau)
    betas  = curves_arr @ pinv_X.T         # (n_dates, 4)
    fitted = betas @ X.T                   # (n_dates, n_tau)
    resid  = curves_arr - fitted

    rmse = np.sqrt(np.mean(resid ** 2, axis=1))

    row_mean = curves_arr.mean(axis=1, keepdims=True)
    ss_tot   = np.sum((curves_arr - row_mean) ** 2, axis=1)
    ss_res   = np.sum(resid ** 2, axis=1)
    r2       = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    return betas, fitted, rmse, r2


# ── Figures ────────────────────────────────────────────────────────────────────

def plot_fit_example(
    curves: pd.DataFrame,
    betas_df: pd.DataFrame,
    tau: np.ndarray,
    lam1: float,
    lam2: float,
    out_dir: str,
) -> None:
    """Observed vs Svensson-fitted curves for three representative dates."""
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
        X_fine   = sv_loadings(tau_fine, lam1, lam2)
        fit_line = X_fine @ b
        ax.plot(tau_fine, fit_line, color=color, linewidth=2, label="SV fit")
        ax.scatter(tau, curves.loc[date].values, color="black", s=25, zorder=5,
                   label="Observed")
        ax.set_title(
            f"{label}\n{date.strftime('%Y-%m-%d')}\n"
            f"β₀={b[0]:.0f}  β₁={b[1]:.1f}  β₂={b[2]:.1f}  β₃={b[3]:.1f}",
            fontsize=7.5,
        )
        ax.set_xlabel("Maturity (months)")
        ax.set_ylabel("Price ($/t)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Svensson (raw) - Cross-Sectional Fit Examples  [λ₁*={lam1:.2f}, λ₂*={lam2:.2f}]",
        fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_svraw_fit_example.png")
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def plot_rmse(rmse_ts: pd.Series, out_dir: str) -> None:
    mu  = float(rmse_ts.mean())
    sig = float(rmse_ts.std())
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(rmse_ts.index, rmse_ts, linewidth=0.8, color="darkorange", label="RMSE ($/t)")
    ax.axhline(mu,         color="navy",   linestyle=":", linewidth=0.8,
               label=f"Mean  {mu:.2f} $/t")
    ax.axhline(mu + 2*sig, color="tomato", linestyle="--", linewidth=1.0,
               label=f"Mean+2σ  {mu+2*sig:.2f} $/t")
    ax.set_ylabel("RMSE ($/t)")
    ax.set_title("Svensson (raw) - Fit RMSE per Date", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_svraw_rmse.png")
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
    print("02b_sv_raw.py - Svensson on raw prices")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    curves = pd.read_parquet(os.path.join(config.PROCESSED_DATA_DIR, "curves.parquet"))
    col_names = [f"m{m:02d}" for m in config.MATURITIES_MONTHS]
    curves = curves[col_names]
    tau    = tau_from_columns(curves)

    curves_arr = curves.values.astype(float)
    train_mask = curves.index <= pd.Timestamp(config.TRAIN_END)
    train_arr  = curves_arr[train_mask]

    print(f"Data:     {len(curves)} obs × {len(tau)} maturities")
    print(f"Maturities: {list(tau.astype(int))} months")
    print(f"Training: {train_mask.sum()} obs (≤ {config.TRAIN_END})")
    print(f"Test:     {(~train_mask).sum()} obs (≥ {config.TEST_START})")

    # ── 2D grid search for optimal (λ₁, λ₂) ──────────────────────────────────
    print(f"\nGrid search SV (raw): {GRID_N}×{GRID_N} pairs, λ ∈ [{GRID_MIN}, {GRID_MAX}]")
    print(f"  Min separation |λ₁ − λ₂| > {MIN_SEP} enforced ...")
    optimal_lam1, optimal_lam2, train_rmse = grid_search_2d(train_arr, tau)
    print(f"  λ₁*        = {optimal_lam1:.4f}")
    print(f"  λ₂*        = {optimal_lam2:.4f}")
    print(f"  Train RMSE = {train_rmse:.4f} $/t")

    # ── Fit all dates ──────────────────────────────────────────────────────────
    print("\nFitting Svensson (raw) for all dates ...")
    betas, _fitted, rmse_arr, r2_arr = fit_all_dates(
        curves_arr, tau, optimal_lam1, optimal_lam2
    )

    betas_df = pd.DataFrame(
        betas, index=curves.index,
        columns=["beta0", "beta1", "beta2", "beta3"],
    )
    rmse_ts = pd.Series(rmse_arr, index=curves.index, name="rmse_pricespace")
    r2_ts   = pd.Series(r2_arr,   index=curves.index, name="r2")

    # ── Summary statistics ─────────────────────────────────────────────────────
    mean_price     = float(curves["m01"].loc[:config.TRAIN_END].mean())
    mean_rmse      = float(rmse_ts.mean())
    median_rmse    = float(rmse_ts.median())
    max_rmse       = float(rmse_ts.max())
    max_rmse_date  = rmse_ts.idxmax().strftime("%Y-%m-%d")
    mean_r2        = float(r2_ts.mean())
    rmse_pct       = mean_rmse / mean_price * 100
    pct_2sigma     = float((rmse_ts > mean_rmse + 2 * float(rmse_ts.std())).mean() * 100)
    pct_above_2pct = float((rmse_ts > mean_price * 0.02).mean() * 100)

    # Cross-sectional AIC in price space (for within-scale NS vs SV comparison)
    n_tau    = len(tau)
    k_sv     = 4
    rss_arr  = (rmse_arr ** 2) * n_tau
    aic_arr  = 2 * k_sv + n_tau * np.log(rss_arr / n_tau)
    mean_aic = float(aic_arr.mean())

    print(f"\n  Mean RMSE ($/t)      = {mean_rmse:.4f}")
    print(f"  Median RMSE ($/t)    = {median_rmse:.4f}")
    print(f"  Max RMSE ($/t)       = {max_rmse:.4f}  ({max_rmse_date})")
    print(f"  RMSE as % of price   = {rmse_pct:.3f}%  (ref: mean m01 = {mean_price:.0f} $/t)")
    print(f"  Mean R²              = {mean_r2:.6f}")
    print(f"  Mean AIC (per date)  = {mean_aic:.2f}")
    print(f"  % dates > mean+2σ    = {pct_2sigma:.1f}%")
    print(f"  % dates > 2% price   = {pct_above_2pct:.1f}%")

    # ── Save factors parquet ───────────────────────────────────────────────────
    out_df = betas_df.copy()
    out_df["rmse_pricespace"] = rmse_ts
    out_df["r2"]              = r2_ts
    out_df["model"]           = "Svensson"
    out_df["scale"]           = "raw"
    parquet_path = os.path.join(comp_dir, "sv_raw_factors.parquet")
    out_df.to_parquet(parquet_path)
    print(f"\nSaved factors: {parquet_path}")

    # ── Save metrics JSON ──────────────────────────────────────────────────────
    metrics = {
        "label":                  "Svensson (raw)",
        "model":                  "Svensson",
        "scale":                  "raw",
        "n_params":               k_sv,
        "lambda1":                round(optimal_lam1, 4),
        "lambda2":                round(optimal_lam2, 4),
        "train_rmse_fitspace":    round(float(train_rmse), 6),
        "mean_rmse_fitspace":     round(mean_rmse, 6),
        "mean_rmse_pricespace":   round(mean_rmse, 6),
        "median_rmse_pricespace": round(median_rmse, 6),
        "max_rmse_pricespace":    round(max_rmse, 6),
        "max_rmse_date":          max_rmse_date,
        "mean_r2_pricespace":     round(mean_r2, 6),
        "mean_aic_fitspace":      round(mean_aic, 4),
        "pct_flagged_2sigma":     round(pct_2sigma, 2),
        "pct_above_2pct_price":   round(pct_above_2pct, 2),
        "mean_rmse_pct_of_price": round(rmse_pct, 4),
        "mean_price_m01_train":   round(mean_price, 2),
        "n_obs":                  int(len(curves)),
        "n_maturities":           int(n_tau),
    }
    json_path = os.path.join(comp_dir, "sv_raw_metrics.json")
    with open(json_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved metrics: {json_path}")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\nGenerating figures ...")
    plot_fit_example(curves, betas_df, tau, optimal_lam1, optimal_lam2, config.FIGURES_DIR)
    plot_rmse(rmse_ts, config.FIGURES_DIR)

    print("\n02b_sv_raw.py complete.")
    print("Next: python code/02b_sv_log.py")


if __name__ == "__main__":
    main()
