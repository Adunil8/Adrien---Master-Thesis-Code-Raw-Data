"""
02b_ns_raw.py - Nelson-Siegel fit on raw (level) prices.

Phase 2b, Configuration 1 of 4.
Fits NS model directly on copper price levels ($/t).
Grid-searches optimal decay λ on training data, then fits all 1065 weekly dates.
Computes RMSE and R² in price space ($/t) and saves factors for downstream comparison.

Outputs:
  data/processed/comparison/ns_raw_factors.parquet   - β₀, β₁, β₂, RMSE, R² per date
  data/processed/comparison/ns_raw_metrics.json      - summary statistics for comparison table
  report/figures/fig_nsraw_fit_example.png
  report/figures/fig_nsraw_rmse.png

Run: python code/02b_ns_raw.py
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


def grid_search(train_arr: np.ndarray, tau: np.ndarray) -> tuple[float, float]:
    """
    Find λ minimising mean cross-sectional RMSE over training dates.
    Vectorised: solves all dates simultaneously for each candidate λ.
    """
    grid = np.linspace(GRID_MIN, GRID_MAX, GRID_N)
    best_lam, best_rmse = grid[0], np.inf
    for lam in grid:
        X = ns_loadings(tau, lam)          # (n_tau, 3)
        pinv_X = np.linalg.pinv(X)        # (3, n_tau)
        betas = train_arr @ pinv_X.T      # (n_train, 3)
        fitted = betas @ X.T              # (n_train, n_tau)
        mean_rmse = float(np.sqrt(np.mean((train_arr - fitted) ** 2, axis=1)).mean())
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_lam  = lam
    return best_lam, best_rmse


def fit_all_dates(
    curves_arr: np.ndarray,
    tau: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit NS(raw) to every date simultaneously via vectorised OLS.

    Returns
    -------
    betas      : (n_dates, 3)  - β₀, β₁, β₂
    fitted_arr : (n_dates, n_tau) - reconstructed curves
    rmse       : (n_dates,)   - RMSE in price space ($/t)
    r2         : (n_dates,)   - R² in price space
    """
    X      = ns_loadings(tau, lam)          # (n_tau, 3)
    pinv_X = np.linalg.pinv(X)             # (3, n_tau)
    betas  = curves_arr @ pinv_X.T         # (n_dates, 3)
    fitted = betas @ X.T                   # (n_dates, n_tau)
    resid  = curves_arr - fitted

    rmse = np.sqrt(np.mean(resid ** 2, axis=1))

    # R² per date
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
    lam: float,
    out_dir: str,
) -> None:
    """Observed vs NS-fitted curves for three representative dates.

    Dates are fixed, not picked by idxmax/idxmin on this fit's own raw-
    price beta1. That selection was tried first and picked 2021-10-29 as
    the "Contango" example, right next to this configuration's own single
    worst-RMSE date (2021-10-15, ns_raw_metrics.json), during the October
    2021 LME copper backwardation squeeze. The high grid-searched lambda
    this configuration uses concentrates curvature very close to the front
    end, and produced an unstable, ill-conditioned fit exactly there
    (beta1=-4528, an order of magnitude beyond any other date): the actual
    observed prices that day were declining with maturity (m01=9728.5 >
    m06=9470.5, near-far spread +258), genuine backwardation, the opposite
    of the fitted label. The three dates below are picked instead directly
    from the observed near-minus-far price spread (model-free) and cross-
    checked against the production log-price beta1 (Section 3.2), both of
    which agree unambiguously: 2006-01-06 is the sample's most extreme
    backwardation, 2021-08-20 is within 1 USD/t of perfectly flat, and
    2024-07-12 is the sample's single deepest contango (also cited in
    Section 3.1's curve description). This also sidesteps a structural
    problem with idxmax/idxmin on a dollar-scale beta1 across a sample
    spanning $3,000-$13,000/t: it selects for whichever era combines
    extreme slope with a high price level, not the most extreme curve
    shape in relative terms, the same scale problem the thesis's own
    choice of log prices for the production model addresses directly
    (Section 3.2)."""
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
        X_fine   = ns_loadings(tau_fine, lam)
        fit_line = X_fine @ b
        ax.plot(tau_fine, fit_line, color=color, linewidth=2, label="NS fit")
        ax.scatter(tau, curves.loc[date].values, color="black", s=25, zorder=5,
                   label="Observed")
        if label == "Neutral / Flat":
            # Autoscaling on a ~$2/t range makes noise look like curvature.
            # Fixed range keeps this panel visually flat, matching what it is.
            ax.set_ylim(9020, 9080)
            ax.set_yticks([9030, 9040, 9050, 9060, 9070])
        ax.set_title(
            f"{label}\n{date.strftime('%Y-%m-%d')}\n"
            r"$\beta_0$" f"={b[0]:.0f}  " r"$\beta_1$" f"={b[1]:.1f}  " r"$\beta_2$" f"={b[2]:.1f}",
            fontsize=8,
        )
        ax.set_xlabel("Maturity (months)")
        ax.set_ylabel("Price ($/t)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "NS (raw prices) - Cross-Sectional Fit Examples",
        fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_nsraw_fit_example.png")
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def plot_rmse(rmse_ts: pd.Series, out_dir: str) -> None:
    """RMSE per date with mean, mean+2σ, and train/test split."""
    mu  = float(rmse_ts.mean())
    sig = float(rmse_ts.std())
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(rmse_ts.index, rmse_ts, linewidth=0.8, color="steelblue", label="RMSE ($/t)")
    ax.axhline(mu,         color="navy",   linestyle=":", linewidth=0.8,
               label=f"Mean  {mu:.2f} $/t")
    ax.axhline(mu + 2*sig, color="tomato", linestyle="--", linewidth=1.0,
               label=f"Mean+2σ  {mu+2*sig:.2f} $/t")
    ax.set_ylabel("RMSE ($/t)")
    ax.set_title("NS (raw) - Fit RMSE per Date", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_nsraw_rmse.png")
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
    print("02b_ns_raw.py - Nelson-Siegel on raw prices")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    curves = pd.read_parquet(os.path.join(config.PROCESSED_DATA_DIR, "curves.parquet"))
    col_names = [f"m{m:02d}" for m in config.MATURITIES_MONTHS]
    curves = curves[col_names]
    tau    = tau_from_columns(curves)

    curves_arr = curves.values.astype(float)      # raw prices ($/t)
    train_mask = curves.index <= pd.Timestamp(config.TRAIN_END)
    train_arr  = curves_arr[train_mask]

    print(f"Data:     {len(curves)} obs × {len(tau)} maturities")
    print(f"Maturities: {list(tau.astype(int))} months")
    print(f"Training: {train_mask.sum()} obs (≤ {config.TRAIN_END})")
    print(f"Test:     {(~train_mask).sum()} obs (≥ {config.TEST_START})")

    # ── Grid search for optimal λ ──────────────────────────────────────────────
    print(f"\nGrid search NS (raw): λ ∈ [{GRID_MIN}, {GRID_MAX}], {GRID_N} points ...")
    optimal_lam, train_rmse = grid_search(train_arr, tau)
    print(f"  λ*         = {optimal_lam:.4f}")
    print(f"  Train RMSE = {train_rmse:.4f} $/t")

    # ── Fit all dates ──────────────────────────────────────────────────────────
    print("\nFitting NS (raw) for all dates ...")
    betas, _fitted, rmse_arr, r2_arr = fit_all_dates(curves_arr, tau, optimal_lam)

    betas_df = pd.DataFrame(betas, index=curves.index,
                            columns=["beta0", "beta1", "beta2"])
    rmse_ts  = pd.Series(rmse_arr, index=curves.index, name="rmse_pricespace")
    r2_ts    = pd.Series(r2_arr,   index=curves.index, name="r2")

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

    # Cross-sectional AIC (fit space, per-date average)
    n_tau   = len(tau)
    k_ns    = 3
    rss_arr = (rmse_arr ** 2) * n_tau
    aic_arr = 2 * k_ns + n_tau * np.log(rss_arr / n_tau)
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
    out_df["model"]           = "NS"
    out_df["scale"]           = "raw"
    parquet_path = os.path.join(comp_dir, "ns_raw_factors.parquet")
    out_df.to_parquet(parquet_path)
    print(f"\nSaved factors: {parquet_path}")

    # ── Save metrics JSON ──────────────────────────────────────────────────────
    metrics = {
        "label":                  "NS (raw)",
        "model":                  "NS",
        "scale":                  "raw",
        "n_params":               k_ns,
        "lambda1":                round(optimal_lam, 4),
        "lambda2":                None,
        "train_rmse_fitspace":    round(train_rmse, 6),
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
    json_path = os.path.join(comp_dir, "ns_raw_metrics.json")
    with open(json_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved metrics: {json_path}")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\nGenerating figures ...")
    plot_fit_example(curves, betas_df, tau, optimal_lam, config.FIGURES_DIR)
    plot_rmse(rmse_ts, config.FIGURES_DIR)

    print("\n02b_ns_raw.py complete.")
    print("Next: python code/02b_ns_log.py")


if __name__ == "__main__":
    main()
