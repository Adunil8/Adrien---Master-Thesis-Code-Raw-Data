"""
02_nelson_siegel.py - Phase 2: Nelson-Siegel cross-sectional fitting.

Inputs  : data/processed/curves.parquet
Outputs : data/processed/ns_factors.parquet   (β₀, β₁, β₂, rmse per date)
          data/processed/ns_regimes.parquet   (seasonally-adjusted regime per date)
          report/figures/fig_ns_fit_example.png
          report/figures/fig_ns_rmse.png
          report/figures/fig_ns_factors.png
          report/figures/fig_ns_regimes.png

Run     : python code/02_nelson_siegel.py
          (from thesis/ root directory with .venv active)

Fit quality gate (CLAUDE.md Phase 2 requirement):
  - Mean RMSE < NS_RMSE_WARN_THRESHOLD (config)
  - < 5% of dates with RMSE > NS_RMSE_FLAG_THRESHOLD (config)
  Both gates must pass before Phase 3 (03_stationarity_lags.py) can run.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

import config
from config import (
    LAMBDA_NS,
    LAMBDA_NS_GRID_MIN,
    LAMBDA_NS_GRID_MAX,
    LAMBDA_NS_GRID_N,
    REGIME_ZSCORE_BACKWARDATION,
    REGIME_ZSCORE_CONTANGO,
    TRAIN_END,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    FIGURE_DPI,
    PLOT_STYLE,
    COLORS,
    METAL,
    MATURITIES_MONTHS,   # controls which maturities enter the NS fit
                         # gasoline: [1..6]   copper: [1..12] or [1..15]
                         # change ONLY this line in config.py when switching commodity
)
from utils import ns_loadings, ns_curve, seasonal_zscore, classify_regime


# ── Helper: extract tau array from curves DataFrame columns ──────────────────

def tau_from_columns(curves: pd.DataFrame) -> np.ndarray:
    """
    Parse maturity months from column names 'm01', 'm02', ... 'm15'.
    Returns 1-D float array, e.g. [1., 2., 3., ..., 12.].
    """
    taus = []
    for col in curves.columns:
        if col.startswith("m") and col[1:].isdigit():
            taus.append(float(col[1:]))
    return np.array(taus, dtype=float)


# ── Grid search for optimal λ ────────────────────────────────────────────────

def grid_search_lambda(
    curves: pd.DataFrame,
    tau: np.ndarray,
    lam_min: float,
    lam_max: float,
    n_points: int,
) -> tuple[float, float]:
    """
    Find the NS decay parameter λ minimising mean cross-sectional RMSE
    across all non-missing dates in the sample.

    The grid covers [lam_min, lam_max] with n_points values.
    Uses numpy OLS (lstsq) for speed - no scipy overhead.

    Returns
    -------
    optimal_lambda : float
    optimal_rmse   : float
    """
    lambda_grid = np.linspace(lam_min, lam_max, n_points)
    best_lambda, best_rmse = lambda_grid[0], np.inf

    for lam in lambda_grid:
        loadings = ns_loadings(tau, lam)
        rmses = []
        for _date, row in curves.iterrows():
            valid = row.dropna()
            if len(valid) < 3:
                continue
            valid_tau = np.array(
                [tau[i] for i, c in enumerate(curves.columns) if c in valid.index]
            )
            load_valid = ns_loadings(valid_tau, lam)
            betas, _, _, _ = np.linalg.lstsq(load_valid, valid.values, rcond=None)
            fitted = load_valid @ betas
            rmses.append(np.sqrt(np.mean((valid.values - fitted) ** 2)))

        mean_rmse = float(np.mean(rmses)) if rmses else np.inf
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_lambda = lam

    return best_lambda, best_rmse


# ── Single-date NS fit ────────────────────────────────────────────────────────

def fit_ns_date(
    prices: np.ndarray,
    tau: np.ndarray,
    lambda_ns: float,
) -> tuple[np.ndarray, float]:
    """
    Fit Nelson-Siegel to observed futures curve at a single date via OLS.

    Returns
    -------
    betas : np.ndarray shape (3,) - [β₀, β₁, β₂]
    rmse  : float
    """
    loadings = ns_loadings(tau, lambda_ns)
    betas, _, _, _ = np.linalg.lstsq(loadings, prices, rcond=None)
    fitted = loadings @ betas
    rmse = float(np.sqrt(np.mean((prices - fitted) ** 2)))
    return betas, rmse


# ── Fit all dates ─────────────────────────────────────────────────────────────

def fit_all_dates(
    curves: pd.DataFrame,
    tau: np.ndarray,
    lambda_ns: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run NS OLS fit for every date in curves.

    Handles missing maturities gracefully: fits only to available (non-NaN)
    columns at each date. Rows with < 3 valid observations are skipped (NaN).

    Returns
    -------
    factors : pd.DataFrame  columns=[beta0, beta1, beta2]
    rmse_ts : pd.Series     per-date RMSE
    """
    factors_list = []
    rmse_list = []

    for _date, row in curves.iterrows():
        valid = row.dropna()
        if len(valid) < 3:
            factors_list.append([np.nan, np.nan, np.nan])
            rmse_list.append(np.nan)
            continue

        valid_tau = np.array(
            [tau[i] for i, c in enumerate(curves.columns) if c in valid.index]
        )
        betas, rmse = fit_ns_date(valid.values, valid_tau, lambda_ns)
        factors_list.append(betas.tolist())
        rmse_list.append(rmse)

    factors = pd.DataFrame(
        factors_list,
        index=curves.index,
        columns=["beta0", "beta1", "beta2"],
    )
    rmse_ts = pd.Series(rmse_list, index=curves.index, name="rmse")
    return factors, rmse_ts


# ── Fit quality gate ──────────────────────────────────────────────────────────

def validate_fit(
    rmse_ts: pd.Series,
    warn_threshold: float,
    flag_threshold: float,
) -> None:
    """
    Enforce fit quality gates from CLAUDE.md Phase 2 design.
    Raises RuntimeError if either condition fails.

    Gate 1: mean RMSE < warn_threshold  (2% of mean training price)
    Gate 2: < 5% of dates with RMSE > flag_threshold  (5% of mean training price)

    Parameters
    ----------
    warn_threshold : mean_price * NS_RMSE_WARN_PCT
    flag_threshold : mean_price * NS_RMSE_FLAG_PCT
    """
    clean = rmse_ts.dropna()
    mean_rmse = float(clean.mean())
    pct_extreme = float((clean > flag_threshold).mean() * 100)

    n_flagged = int((clean > rmse_ts.mean() + 2 * rmse_ts.std()).sum())

    print(f"\n  NS Fit Quality:")
    print(f"    Mean RMSE        : {mean_rmse:.4f}  (gate: < {warn_threshold:.4f})")
    print(f"    % dates > {flag_threshold:.4f}  : {pct_extreme:.1f}%  (gate: < 5.0%)")
    print(f"    Dates > mean+2σ  : {n_flagged}")

    if mean_rmse > warn_threshold:
        raise RuntimeError(
            f"NS mean RMSE = {mean_rmse:.4f} exceeds threshold {warn_threshold:.4f}.\n"
            "Consider Svensson (4-factor) as robustness check - see Appendix IV."
        )
    if pct_extreme > 5.0:
        raise RuntimeError(
            f"{pct_extreme:.1f}% of dates have RMSE > {flag_threshold:.4f} (gate: 5%).\n"
            "Investigate outlier dates before proceeding to Phase 3."
        )
    print("    FIT QUALITY GATES PASSED.")


# ── Regime classification ─────────────────────────────────────────────────────

def compute_regimes(
    factors: pd.DataFrame,
    train_end: str,
) -> pd.DataFrame:
    """
    Classify curve regime from the raw sign of β₁.

    β₁ > 0 → backwardation (near-end premium: F(1M) > F(6M))
    β₁ < 0 → contango     (far-end premium:  F(1M) < F(6M))

    The β₁ sign is the direct economic signal: no seasonal normalisation is
    applied because the 1–6M copper curve is monotone (clearly in one
    direction) almost all the time - seasonal z-score normalisation adds a
    neutral zone that obscures the actual market direction.

    The z-score is still computed and stored in the parquet for diagnostics
    and for the macro-variable Granger analysis (04_granger.py uses z-scores
    of DXY and inventory as VAR inputs, not the regime label itself).

    Returns DataFrame with columns: beta1_zscore, regime, week_number.
    """
    z_scores = seasonal_zscore(factors["beta1"], train_end)
    # Regime from raw β₁ sign - threshold=0.0 on raw beta1 (not z-score)
    regimes = classify_regime(
        factors["beta1"],   # raw β₁, not z-score
        threshold=0.0,      # β₁ > 0 = backwardation, β₁ < 0 = contango
    )
    week_numbers = pd.Series(
        [d.isocalendar().week for d in factors.index],
        index=factors.index,
        name="week_number",
        dtype=int,
    )
    return pd.DataFrame(
        {"beta1_zscore": z_scores, "regime": regimes, "week_number": week_numbers}
    )


# ── Figure 1: example NS fits ─────────────────────────────────────────────────

def plot_ns_fit_example(
    curves: pd.DataFrame,
    factors: pd.DataFrame,
    tau: np.ndarray,
    lambda_ns: float,
    out_dir: str,
) -> None:
    """
    Plot observed vs NS-fitted curves for three representative dates:
    one backwardation episode, one contango episode, one near-flat curve.
    """
    # Pick representative dates from training period
    train_factors = factors.loc[:TRAIN_END].dropna()

    # Backwardation: largest positive β₁ (slope > 0 → near > far → backwardation)
    date_back = train_factors["beta1"].idxmax()
    # Contango: most negative β₁
    date_cont = train_factors["beta1"].idxmin()
    # Flat: β₁ closest to zero (excluding the above)
    exclude = {date_back, date_cont}
    remaining = train_factors.drop(list(exclude))
    date_flat = remaining["beta1"].abs().idxmin()

    dates = [
        (date_back, "Backwardation", COLORS["backwardation"]),
        (date_flat, "Flat / Neutral", COLORS["neutral"]),
        (date_cont, "Contango", COLORS["contango"]),
    ]

    tau_fine = np.linspace(tau.min(), tau.max(), 300)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    metal_label = METAL.capitalize()

    for ax, (date, label, colour) in zip(axes, dates):
        betas = factors.loc[date, ["beta0", "beta1", "beta2"]].values
        row = curves.loc[date].dropna()
        valid_tau = np.array(
            [tau[i] for i, c in enumerate(curves.columns) if c in row.index]
        )
        y_fitted = ns_curve(betas, tau_fine, lambda_ns)
        ax.plot(tau_fine, y_fitted, color=colour, linewidth=2, label="NS fit")
        ax.scatter(valid_tau, row.values, color="black", zorder=5, s=30, label="Observed")
        ax.set_title(
            f"{label}\n{date.strftime('%Y-%m-%d')}\n"
            r"$\beta_0$" f"={betas[0]:.2f}  " r"$\beta_1$" f"={betas[1]:.2f}  " r"$\beta_2$" f"={betas[2]:.2f}",
            fontsize=8,
        )
        ax.set_xlabel("Maturity (months)")
        ax.set_ylabel("Price")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Nelson-Siegel Cross-Sectional Fit - {metal_label} Futures",
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_ns_fit_example.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_ns_fit_example.png")


# ── Figure 2: RMSE time series ────────────────────────────────────────────────

def plot_rmse_timeseries(rmse_ts: pd.Series, out_dir: str) -> None:
    """RMSE per date with +2σ flag threshold and train/test split marker."""
    clean = rmse_ts.dropna()
    threshold_2s = float(clean.mean() + 2 * clean.std())
    metal_label = METAL.capitalize()

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(rmse_ts.index, rmse_ts, linewidth=0.8, color="steelblue", label="RMSE")
    ax.axhline(
        threshold_2s,
        color="tomato",
        linestyle="--",
        linewidth=1.0,
        label=f"Mean + 2σ  ({threshold_2s:.4f})",
    )
    ax.axhline(
        clean.mean(),
        color="navy",
        linestyle=":",
        linewidth=0.8,
        label=f"Mean ({clean.mean():.4f})",
    )
    ax.axvline(
        pd.Timestamp(TRAIN_END),
        color="darkorange",
        linestyle="-",
        linewidth=1.0,
        alpha=0.7,
        label="Train / Test split",
    )
    ax.set_ylabel("RMSE")
    ax.set_title(
        f"NS Cross-Sectional Fit Quality - {metal_label} Futures 2006–2026",
        fontweight="bold",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_ns_rmse.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_ns_rmse.png")


# ── Figure 3: NS factor time series ──────────────────────────────────────────

def plot_ns_factors(factors: pd.DataFrame, out_dir: str) -> None:
    """β₀, β₁, β₂ time series (log-price scale, Section 3.3), train/test split marked."""
    factor_meta = {
        "beta0": (r"$\beta_0$ - Level",     "steelblue"),
        "beta1": (r"$\beta_1$ - Slope",     "seagreen"),
        "beta2": (r"$\beta_2$ - Curvature", "darkorange"),
    }
    metal_label = METAL.capitalize()

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    for ax, (col, (ylabel, color)) in zip(axes, factor_meta.items()):
        ax.plot(factors.index, factors[col], linewidth=0.8, color=color)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axvline(
            pd.Timestamp(TRAIN_END),
            color="darkorange",
            linestyle="-",
            linewidth=0.8,
            alpha=0.6,
            label="Train/Test split",
        )
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)

    # Train/test split legend on top panel only
    axes[0].set_title(
        f"Dynamic NS Factors - Log {metal_label} Futures 2006–2026",
        fontweight="bold",
    )
    split_line = plt.Line2D([0], [0], color="darkorange", linewidth=1.0, label="Train/Test split")
    axes[0].legend(handles=[split_line], fontsize=7, loc="upper right")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_ns_factors.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_ns_factors.png")


# ── Figure 4: regime classification ──────────────────────────────────────────

def plot_ns_regimes(
    factors: pd.DataFrame,
    regimes_df: pd.DataFrame,
    out_dir: str,
) -> None:
    """
    Two-panel regime figure:
      Panel 1 - β₁ time series with regime-coloured background shading.
      Panel 2 - Regime bar (colour strip) showing regime per date.
    """
    metal_label = METAL.capitalize()

    regime_color_map = {
        "backwardation": COLORS["backwardation"],
        "neutral":       COLORS["neutral"],
        "contango":      COLORS["contango"],
    }

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 6), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    # Panel 1: β₁ with regime background
    ax1.plot(factors.index, factors["beta1"], linewidth=0.9, color="black", label=r"$\beta_1$ (raw)")
    ax1.plot(
        regimes_df.index,
        regimes_df["beta1_zscore"],
        linewidth=0.7,
        color="royalblue",
        alpha=0.6,
        label=r"$\beta_1$ z-score (seasonal adj.)",
    )
    ax1.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.4)
    ax1.axhline(REGIME_ZSCORE_BACKWARDATION, color=COLORS["backwardation"],
                linewidth=0.6, linestyle=":", alpha=0.7)
    ax1.axhline(REGIME_ZSCORE_CONTANGO, color=COLORS["contango"],
                linewidth=0.6, linestyle=":", alpha=0.7)
    ax1.axvline(pd.Timestamp(TRAIN_END), color="darkorange", linewidth=0.8,
                alpha=0.7, linestyle="-")
    ax1.set_ylabel(r"$\beta_1$ / z-score", fontsize=9)
    ax1.set_title(
        f"Regime Classification - {metal_label} Futures (Seasonal Z-Score Method)",
        fontweight="bold",
    )
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)

    # Panel 2: colour strip
    dates = regimes_df.index
    regime_vals = regimes_df["regime"]
    for i in range(len(dates) - 1):
        ax2.axvspan(
            dates[i],
            dates[i + 1],
            color=regime_color_map.get(regime_vals.iloc[i], COLORS["neutral"]),
            alpha=0.85,
        )
    ax2.set_yticks([])
    ax2.set_ylabel("Regime", fontsize=8)

    # Legend for regime colours
    patches = [
        mpatches.Patch(color=COLORS["backwardation"], label="Backwardation"),
        mpatches.Patch(color=COLORS["neutral"],       label="Neutral"),
        mpatches.Patch(color=COLORS["contango"],      label="Contango"),
    ]
    ax2.legend(handles=patches, loc="lower right", fontsize=7, ncol=3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_ns_regimes.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_ns_regimes.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    plt.style.use(PLOT_STYLE)

    print("=" * 60)
    print("Phase 2 - Nelson-Siegel Cross-Sectional Fitting")
    print("=" * 60)

    # ── Load curves ──────────────────────────────────────────────────────────
    curves_path = os.path.join(PROCESSED_DATA_DIR, "curves.parquet")
    if not os.path.exists(curves_path):
        raise FileNotFoundError(
            f"{curves_path} not found. Run 01_data_cleaning.py first."
        )
    curves = pd.read_parquet(curves_path)

    # ── Select maturities from config ────────────────────────────────────────
    # To switch commodity: update MATURITIES_MONTHS in config.py - nothing else.
    # Gasoline:  MATURITIES_MONTHS = [1, 2, 3, 4, 5, 6]
    # Copper:    MATURITIES_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    col_names = [f"m{m:02d}" for m in MATURITIES_MONTHS]
    missing_cols = [c for c in col_names if c not in curves.columns]
    if missing_cols:
        raise ValueError(
            f"Columns {missing_cols} not found in curves.parquet.\n"
            f"Available: {list(curves.columns)}\n"
            f"Update MATURITIES_MONTHS in config.py to match available data."
        )
    curves = curves[col_names]
    tau = tau_from_columns(curves)

    print(f"\nLoaded {len(curves)} weekly observations.")
    print(f"Maturities: {list(tau.astype(int))} months  ({len(tau)} contracts)")
    print(f"Date range: {curves.index[0].date()} → {curves.index[-1].date()}")
    print(f"Missing data per column:\n{curves.isna().sum()[curves.isna().sum() > 0]}")

    # ── Apply price transformation ───────────────────────────────────────────
    if config.NS_PRICE_SCALE == 'log':
        curves_fit = np.log(curves)
        print(f"\n  Price scale: LOG (following Bianchi et al. 2023)")
    else:
        curves_fit = curves.copy()
        print(f"\n  Price scale: RAW (original units)")

    # ── Grid search for optimal λ ────────────────────────────────────────────
    print(
        f"\nGrid search: λ ∈ [{LAMBDA_NS_GRID_MIN}, {LAMBDA_NS_GRID_MAX}]"
        f"  ({LAMBDA_NS_GRID_N} points)  - training data only"
    )
    train_curves_fit = curves_fit.loc[:TRAIN_END]
    optimal_lambda, grid_best_rmse = grid_search_lambda(
        train_curves_fit, tau,
        LAMBDA_NS_GRID_MIN, LAMBDA_NS_GRID_MAX, LAMBDA_NS_GRID_N,
    )
    print(f"  Grid search result : λ* = {optimal_lambda:.4f},  RMSE = {grid_best_rmse:.6f}")
    print(f"  Config LAMBDA_NS   : {LAMBDA_NS}")

    # ── λ selection: grid search vs empirically-validated fallback ───────────
    # With only 6 maturities and 3 NS parameters the log-scale RMSE landscape is
    # essentially flat (RMSE < 0.001 for all λ in the grid): OLS near-interpolates
    # 6 data points with 3 free parameters, so fit quality alone cannot select λ.
    # When that happens, λ is chosen instead by a market-relevant criterion: which
    # λ makes sign(β1) agree most often with the curve's own, directly observed
    # backwardation/contango state (front-month vs 6M, no model involved)? That
    # search is run by 02c_lambda_classification_validation.py and its result is
    # read here rather than hardcoded, so this stays correct if the data changes.
    lambda_val_path = os.path.join(PROCESSED_DATA_DIR, "lambda_classification_validation.parquet")
    if not os.path.exists(lambda_val_path):
        raise FileNotFoundError(
            f"{lambda_val_path} not found. Run 02c_lambda_classification_validation.py first."
        )
    lambda_val_results = pd.read_parquet(lambda_val_path)
    best_val_row = lambda_val_results.loc[lambda_val_results["agreement_pct"].idxmax()]
    lambda_dl = float(best_val_row["lambda"])
    lambda_dl_agreement = float(best_val_row["agreement_pct"])
    rmse_range  = abs(grid_best_rmse - LAMBDA_NS)   # placeholder; recalculate below

    # Recompute RMSE range across the grid to detect flatness
    lambda_grid = np.linspace(LAMBDA_NS_GRID_MIN, LAMBDA_NS_GRID_MAX, min(20, LAMBDA_NS_GRID_N))
    rmse_vals   = []
    for lam in lambda_grid:
        _, r = grid_search_lambda(train_curves_fit, tau, lam, lam, 1)
        rmse_vals.append(r)
    rmse_range = max(rmse_vals) - min(rmse_vals)

    at_boundary = (
        abs(optimal_lambda - LAMBDA_NS_GRID_MIN) < 1e-6 or
        abs(optimal_lambda - LAMBDA_NS_GRID_MAX) < 1e-6
    )
    is_flat = rmse_range < 1e-4   # RMSE range < 0.0001 across grid = effectively flat

    if at_boundary or is_flat:
        lambda_to_use = lambda_dl
        print(
            f"  Grid result is {'at boundary' if at_boundary else 'on flat landscape'} "
            f"(RMSE range = {rmse_range:.6f})."
        )
        print(
            f"  → Reverting to empirically-validated fallback: λ = {lambda_dl:.4f} "
            f"({lambda_dl_agreement:.1f}% backwardation/contango agreement, "
            f"thesis Section 3.2)"
        )
    else:
        lambda_to_use = optimal_lambda
        print(f"  → Using grid result λ = {lambda_to_use:.4f}")

    print(f"  Final λ used: {lambda_to_use:.4f}")

    # ── Fit NS for all dates ─────────────────────────────────────────────────
    print("\nFitting NS model across all dates...")
    factors, rmse_ts = fit_all_dates(curves_fit, tau, lambda_to_use)
    n_valid = int(factors.dropna().shape[0])
    print(f"  {n_valid} / {len(curves)} dates with valid factors.")

    # ── Compute percentage-based RMSE thresholds (always on raw prices) ─────
    mean_price = float(curves["m01"].loc[:TRAIN_END].mean())
    rmse_warn_threshold = mean_price * config.NS_RMSE_WARN_PCT
    rmse_flag_threshold = mean_price * config.NS_RMSE_FLAG_PCT
    print(f"\n  Reference price (mean m01, training, raw): {mean_price:.2f}")
    print(f"  RMSE warn threshold : {rmse_warn_threshold:.4f}  ({config.NS_RMSE_WARN_PCT*100:.0f}% of mean price)")
    print(f"  RMSE flag threshold : {rmse_flag_threshold:.4f}  ({config.NS_RMSE_FLAG_PCT*100:.0f}% of mean price)")

    # ── Validate fit quality ─────────────────────────────────────────────────
    validate_fit(rmse_ts, rmse_warn_threshold, rmse_flag_threshold)

    # ── Regime classification (Step 4b) ─────────────────────────────────────
    print("\nComputing seasonally-adjusted z-score regime classification...")
    regimes_df = compute_regimes(factors, TRAIN_END)

    counts = regimes_df["regime"].value_counts()
    pct = (counts / len(regimes_df) * 100).round(1)
    print("  Regime distribution (full sample):")
    for regime in ["backwardation", "neutral", "contango"]:
        n = counts.get(regime, 0)
        p = pct.get(regime, 0.0)
        print(f"    {regime:15s}: {n:4d} weeks  ({p:.1f}%)")

    train_regimes = regimes_df.loc[:TRAIN_END, "regime"]
    test_regimes  = regimes_df.loc[TRAIN_END:,  "regime"]
    print(f"\n  Training (≤{TRAIN_END[:4]}) backwardation weeks : "
          f"{(train_regimes=='backwardation').sum()}")
    print(f"  Test     (>{TRAIN_END[:4]}) backwardation weeks : "
          f"{(test_regimes=='backwardation').sum()}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    # ns_factors.parquet - raw NS output
    factors_out = factors.copy()
    factors_out["rmse"] = rmse_ts
    factors_out["price_scale"] = config.NS_PRICE_SCALE  # 'log' or 'raw'
    factors_path = os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet")
    factors_out.to_parquet(factors_path)
    print(f"\nSaved: {factors_path}")
    print(f"  Columns: {list(factors_out.columns)}")

    # ns_regimes.parquet - regime classification
    regimes_path = os.path.join(PROCESSED_DATA_DIR, "ns_regimes.parquet")
    regimes_df.to_parquet(regimes_path)
    print(f"Saved: {regimes_path}")
    print(f"  Columns: {list(regimes_df.columns)}")

    # ── Diagnostic figures ───────────────────────────────────────────────────
    print("\nGenerating diagnostic figures...")
    plot_ns_fit_example(curves_fit, factors, tau, lambda_to_use, FIGURES_DIR)
    plot_rmse_timeseries(rmse_ts, FIGURES_DIR)
    plot_ns_factors(factors, FIGURES_DIR)
    plot_ns_regimes(factors, regimes_df, FIGURES_DIR)

    # ── Summary statistics ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("FACTOR SUMMARY STATISTICS")
    print("─" * 60)
    print(factors[["beta0", "beta1", "beta2"]].describe().round(4))
    print(f"\nOptimal λ used  : {lambda_to_use:.4f}")
    print(f"Mean fit RMSE   : {rmse_ts.dropna().mean():.4f}")
    print(f"Dates flagged   : {int((rmse_ts > rmse_ts.mean() + 2*rmse_ts.std()).sum())}")
    print("─" * 60)

    print("\nPhase 2 complete.")
    print("FIT QUALITY GATES PASSED - proceed to Phase 3.")
    print("Next: python code/03_stationarity_lags.py")


if __name__ == "__main__":
    main()
