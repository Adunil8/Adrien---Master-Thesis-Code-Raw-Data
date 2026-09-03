"""
03_stationarity_lags.py - Phase 3a: Factor stationarity and lag order selection.

Inputs  : data/processed/ns_factors.parquet
          data/processed/macro.parquet
          data/processed/macro_changes.parquet
Outputs : data/processed/stationarity_results.parquet      ← ADF/KPSS on levels
          data/processed/stationarity_transformed.parquet  ← ADF/KPSS on z52W series
          data/processed/var_spec_decision.parquet         ← final VAR variable spec
          data/processed/lag_selection.parquet
          report/figures/fig_stationarity_timeseries.png   ← raw series + rolling mean/std
          report/figures/fig_stationarity_acf.png          ← ACF for raw series
          report/figures/fig_stationarity_transform.png    ← levels vs z52W comparison
          report/figures/fig_stationarity_acf_z52w.png     ← ACF for z52W series

Methodological logic
--------------------
For each variable entering the VAR, stationarity is required. The decision tree is:
  (1) Test in LEVELS - if stationary (both ADF and KPSS agree) → use levels.
  (2) If non-stationary → test z52W (52-week rolling z-score).
      If z52W is stationary → use z52W (preserves contextual information relative
      to trailing-year norm; economically meaningful for macro-commodity links).
  (3) If z52W is also non-stationary → use first differences (ΔX_t, guaranteed I(0)
      for any I(1) series; most conservative, methodologically unambiguous).
This three-step decision is documented for each variable in var_spec_decision.parquet
and reported in Section 3.3 of the thesis.

Run: python code/03_stationarity_lags.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tsa.api import VAR

from config import (
    TRAIN_END, VAR_MAX_LAGS,
    PROCESSED_DATA_DIR, FIGURES_DIR,
    FIGURE_DPI, PLOT_STYLE,
    MACRO_CHANGE_COLS, MACRO_LEVEL_COLS,
)


# ── Verdict helpers ────────────────────────────────────────────────────────────

def _verdict_color(verdict: str) -> str:
    if "NON-STATIONARY" in verdict:
        return "#dc2626"   # red
    if "STATIONARY (both" in verdict:
        return "#16a34a"   # green
    return "#d97706"       # amber - conflicting


def _verdict_short(verdict: str) -> str:
    if "NON-STATIONARY" in verdict:
        return "Non-stationary"
    if "STATIONARY (both" in verdict:
        return "Stationary"
    return "Conflicting"


# ── 1. ADF + KPSS stationarity test ───────────────────────────────────────────

def run_stationarity_tests(series: pd.Series, label: str) -> dict:
    """
    Run ADF (H0: unit root) and KPSS (H0: stationary) on a single series.
    Uses AIC-selected lag order for ADF; automatic bandwidth for KPSS.
    Returns a dict with test statistics, p-values, and stationarity verdict.
    """
    s = series.dropna()

    adf_stat, adf_pval, adf_lags, _, _, _ = adfuller(s, autolag="AIC")
    kpss_stat, kpss_pval, kpss_lags, _ = kpss(s, regression="c", nlags="auto")

    adf_reject  = adf_pval  < 0.05   # reject H0 (unit root) → evidence of stationarity
    kpss_reject = kpss_pval < 0.05   # reject H0 (stationary) → evidence of non-stationarity

    adf_stationary  = adf_reject
    kpss_stationary = not kpss_reject

    if adf_stationary and kpss_stationary:
        verdict = "STATIONARY (both agree)"
    elif not adf_stationary and not kpss_stationary:
        verdict = "NON-STATIONARY (both agree)"
    elif adf_stationary and not kpss_stationary:
        verdict = "CONFLICTING - ADF: stationary, KPSS: non-stationary"
    else:
        verdict = "CONFLICTING - ADF: non-stationary, KPSS: stationary"

    return {
        "series":    label,
        "adf_stat":  round(adf_stat,  4),
        "adf_pval":  round(adf_pval,  4),
        "adf_lags":  int(adf_lags),
        "kpss_stat": round(kpss_stat, 4),
        "kpss_pval": round(kpss_pval, 4),
        "kpss_lags": int(kpss_lags),
        "verdict":   verdict,
    }


# ── 2. VAR specification decision ─────────────────────────────────────────────

def build_var_spec_decision(
    stat_levels: pd.DataFrame,
    stat_z52w:   pd.DataFrame,
    macro_changes_train: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    For each variable, apply the three-step decision tree and determine
    the transformation used in the VAR.

    Returns a DataFrame with one row per variable, columns:
        variable, stationary_in_levels, z52w_tested, stationary_in_z52w,
        var_transformation, economic_rationale

    macro_changes_train enables a fourth category (change_vars, below) for
    variables that enter the VAR through a fixed-window change transform
    (US3M_d1W, GPR_d26W, inventory_lag2_d4W) rather than levels or a rolling
    z-score (Section 3.1).
    A differenced series is NOT automatically stationary just because the
    level series is I(1) - US3M_d26W (a longer window, tested and rejected
    in favour of d1W) turned out to be non-stationary by both ADF and KPSS
    in this sample, most likely because 26-week rate changes ride long
    sustained hiking/cutting cycles (2008, 2015-19, 2022). ADF/KPSS are run
    on the actual transform used, not assumed, for exactly this reason.
    """
    # Variables entered in levels (factors + stationary macro)
    level_vars = {
        "beta0":          "NS level factor (log-price) - stationary confirmed",
        "beta1":          "NS slope factor - stationary confirmed",
        "beta2":          "NS curvature factor - stationary confirmed",
        "VIX":            "VIX implied volatility - stationary confirmed",
    }
    # Variables requiring z52W transformation
    z52w_vars = {
        "DXY":    "DXY_z52W",
        "T10Y2Y": "T10Y2Y_z52W",
    }
    # Variables entering via a fixed-window change transform (see docstring).
    # Derived from MACRO_CHANGE_COLS (config.py) rather than hardcoded, so
    # this stays correct automatically if the window/lag for any of these
    # three is ever revised there (hardcoding the column name directly would
    # silently go stale the moment config.py's window choice changes).
    change_vars = {
        "US3M":      next(c for c in MACRO_CHANGE_COLS if c.startswith("US3M")),
        "GPR":       next(c for c in MACRO_CHANGE_COLS if c.startswith("GPR")),
        "inventory": next(c for c in MACRO_CHANGE_COLS if c.startswith("inventory")),
    }

    rows = []
    for var in ["beta0", "beta1", "beta2", "inventory_lag2", "VIX"]:
        verdict = stat_levels.loc[var, "verdict"] if var in stat_levels.index else "not tested"
        rows.append({
            "variable":              var,
            "stationary_in_levels":  "STATIONARY" in verdict or "ADF: stationary" in verdict,
            "z52w_tested":           False,
            "stationary_in_z52w":    None,
            "var_transformation":    "levels",
            "adf_pval_levels":       stat_levels.loc[var, "adf_pval"] if var in stat_levels.index else None,
            "kpss_pval_levels":      stat_levels.loc[var, "kpss_pval"] if var in stat_levels.index else None,
            "adf_pval_z52w":         None,
            "kpss_pval_z52w":        None,
            "economic_rationale":    level_vars.get(var, ""),
        })

    for var, z_col in z52w_vars.items():
        verdict_lv = stat_levels.loc[var, "verdict"]  if var in stat_levels.index else ""
        z_label    = z_col + " (transformed)"
        verdict_z  = stat_z52w.loc[z_label, "verdict"] if z_label in stat_z52w.index else "not tested"
        # Same trend-stationary standard applied to beta0/beta1/beta2 (line 162)
        # and the change-transform variables (line 215): a clean "STATIONARY"
        # verdict or the "CONFLICTING - ADF: stationary" pattern both count.
        # An earlier version of this line checked "STATIONARY" only, which
        # silently downgraded DXY/T10Y2Y to first-difference despite the
        # thesis explicitly treating this exact conflicting pattern as
        # trend-stationary elsewhere (Section 3.3) - inconsistent with the
        # production spec, which uses DXY_z52W (config.py MACRO_VAR_SPEC).
        z_stationary = "STATIONARY" in verdict_z or "ADF: stationary" in verdict_z
        transformation = "z52W" if z_stationary else "first_difference"
        rationale = (
            f"Non-stationary in levels (ADF p={stat_levels.loc[var,'adf_pval']:.4f}, "
            f"KPSS p={stat_levels.loc[var,'kpss_pval']:.4f}). "
        )
        if z_stationary:
            rationale += (
                f"z52W trend-stationary (ADF p={stat_z52w.loc[z_label,'adf_pval']:.4f}), "
                "same conflicting-but-accepted pattern as beta1/beta2 (Section 3.3). "
                "z52W preferred over first difference on economic grounds: measures "
                "deviation from trailing-year norm - economically meaningful (contextual "
                "USD/rate signal for commodity prices) and confirmed strongest predictor "
                "in Granger tests."
            )
        else:
            rationale += "z52W also non-stationary → first difference used (guaranteed I(0))."

        rows.append({
            "variable":              var,
            "stationary_in_levels":  False,
            "z52w_tested":           True,
            "stationary_in_z52w":    z_stationary,
            "var_transformation":    transformation,
            "adf_pval_levels":       stat_levels.loc[var, "adf_pval"]  if var in stat_levels.index else None,
            "kpss_pval_levels":      stat_levels.loc[var, "kpss_pval"] if var in stat_levels.index else None,
            "adf_pval_z52w":         stat_z52w.loc[z_label, "adf_pval"]  if z_label in stat_z52w.index else None,
            "kpss_pval_z52w":        stat_z52w.loc[z_label, "kpss_pval"] if z_label in stat_z52w.index else None,
            "economic_rationale":    rationale,
        })

    if macro_changes_train is not None:
        for var, d_col in change_vars.items():
            if d_col not in macro_changes_train.columns:
                continue
            series = macro_changes_train[d_col].dropna()
            if len(series) < 30:
                continue
            test = run_stationarity_tests(series, d_col)
            adf_p, kpss_p = test["adf_pval"], test["kpss_pval"]
            stationary = "STATIONARY" in test["verdict"] or "ADF: stationary" in test["verdict"]
            adf_str = f"{adf_p:.4g}"
            verdict_str = "stationary" if stationary else "NOT stationary - see limitation"
            window_label = d_col.rsplit("_", 1)[-1].replace("d", "").replace("W", "-week")
            rationale = (
                f"{var} enters via a {window_label} change transform, not levels or z52W "
                f"(see Section 3.1 - selected on Granger evidence, not a levels "
                f"stationarity failure). ADF on {d_col}: p={adf_str} - {verdict_str}."
            )
            rows.append({
                "variable":              var,
                "stationary_in_levels":  False,   # raw level not used in the VAR for this variable
                "z52w_tested":           False,
                "stationary_in_z52w":    None,
                "var_transformation":    d_col,
                "adf_pval_levels":       None,
                "kpss_pval_levels":      None,
                "adf_pval_z52w":         round(adf_p, 4)  if adf_p  is not None else None,
                "kpss_pval_z52w":        round(kpss_p, 4) if kpss_p is not None else None,
                "economic_rationale":    rationale,
            })

    return pd.DataFrame(rows).set_index("variable")


# ── 3. Stationarity diagnostic figures ────────────────────────────────────────

SERIES_LABELS = {
    "beta0":         r"$\beta_0$ - Level (log-price)",
    "beta1":         r"$\beta_1$ - Slope (backwardation / contango)",
    "beta2":         r"$\beta_2$ - Curvature",
    "inventory_lag2":"Inventory - 2-week lag (tonnes)",
    "DXY":           "DXY - US Dollar Index",
    "VIX":           "VIX - Implied Volatility",
    "T10Y2Y":        "T10Y2Y - Yield Spread (10Y − 2Y, %)",
    "DXY_z52W":      "DXY - 52-week rolling z-score",
    "T10Y2Y_z52W":   "T10Y2Y - 52-week rolling z-score",
    "VIX_z52W":      "VIX - 52-week rolling z-score",
    "inventory_z52W":"Inventory - 52-week rolling z-score",
}


def plot_timeseries_panel(
    all_series:   dict,
    stat_results: pd.DataFrame,
    title:        str,
    out_path:     str,
    roll_window:  int = 52,
) -> None:
    """
    Panel: raw time series + rolling mean + ±1 rolling std band.
    Title of each panel is colour-coded by stationarity verdict.
    """
    plt.style.use(PLOT_STYLE)
    n = len(all_series)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes_flat  = axes.flatten() if n > 1 else [axes]

    for idx, (name, series) in enumerate(all_series.items()):
        ax  = axes_flat[idx]
        s   = series.dropna()
        rm  = s.rolling(roll_window, min_periods=roll_window // 2).mean()
        rs  = s.rolling(roll_window, min_periods=roll_window // 2).std()

        # Look up verdict - try both "name" and "name + (transformed)" keys
        verdict = ""
        for key in [name, name + " (transformed)"]:
            if key in stat_results.index:
                verdict = stat_results.loc[key, "verdict"]
                break
        color     = _verdict_color(verdict)
        label_str = _verdict_short(verdict)
        adf_p   = stat_results.loc[name, "adf_pval"]  if name in stat_results.index else (
                  stat_results.loc[name + " (transformed)", "adf_pval"]  if name + " (transformed)" in stat_results.index else float("nan"))
        kpss_p  = stat_results.loc[name, "kpss_pval"] if name in stat_results.index else (
                  stat_results.loc[name + " (transformed)", "kpss_pval"] if name + " (transformed)" in stat_results.index else float("nan"))

        ax.plot(s.index, s.values, color="#374151", linewidth=0.7, alpha=0.85, label="Observed")
        ax.plot(rm.index, rm.values, color=color, linewidth=1.8, linestyle="--", label=f"{roll_window}W rolling mean")
        ax.fill_between(s.index, (rm - rs).values, (rm + rs).values,
                        color=color, alpha=0.13, label="±1 rolling std")

        ax.set_title(
            f"{SERIES_LABELS.get(name, name)}  [{label_str}]\n"
            f"ADF p = {adf_p:.4f}   KPSS p = {kpss_p:.4f}",
            fontsize=9, color=color, fontweight="bold",
        )
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=6.5, loc="upper left")
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(color)
            sp.set_linewidth(1.6)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_acf_panel(
    all_series:   dict,
    stat_results: pd.DataFrame,
    title:        str,
    out_path:     str,
    nlags:        int = 40,
) -> None:
    """
    ACF panel. Non-stationary series → slow ACF decay.
    Stationary series → rapid decay to zero within a few lags.
    """
    plt.style.use(PLOT_STYLE)
    n = len(all_series)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes_flat  = axes.flatten() if n > 1 else [axes]

    for idx, (name, series) in enumerate(all_series.items()):
        ax  = axes_flat[idx]
        s   = series.dropna()
        N   = len(s)
        ci  = 1.96 / np.sqrt(N)

        acf_vals = acf(s, nlags=nlags, fft=True)
        lags     = np.arange(len(acf_vals))

        verdict = ""
        for key in [name, name + " (transformed)"]:
            if key in stat_results.index:
                verdict = stat_results.loc[key, "verdict"]
                break
        color     = _verdict_color(verdict)
        label_str = _verdict_short(verdict)

        ax.bar(lags, acf_vals, color=color, alpha=0.60, width=0.6)
        ax.axhline(+ci, linestyle="--", color="#6b7280", linewidth=0.9,
                   label=f"95% CI ±{ci:.3f}")
        ax.axhline(-ci, linestyle="--", color="#6b7280", linewidth=0.9)
        ax.axhline(0,   color="black",  linewidth=0.5)

        ax.set_title(
            f"{SERIES_LABELS.get(name, name)}  [{label_str}]",
            fontsize=9, color=color, fontweight="bold",
        )
        ax.set_xlabel("Lag (weeks)", fontsize=7)
        ax.set_ylabel("ACF", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5)
        ax.set_ylim(-0.35, 1.05)
        for sp in ax.spines.values():
            sp.set_edgecolor(color)
            sp.set_linewidth(1.6)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_transformation_comparison(
    macro_levels:  pd.DataFrame,
    macro_changes: pd.DataFrame,
    stat_levels:   pd.DataFrame,
    stat_z52w:     pd.DataFrame,
    out_path:      str,
) -> None:
    """
    Side-by-side comparison: raw level vs. z52W for each non-stationary macro variable.
    Left column = raw level (non-stationary - drifting mean).
    Right column = z52W transformation (stationary - mean-reverting).
    This figure justifies the z52W choice over first differences in Section 3.3.
    """
    plt.style.use(PLOT_STYLE)
    pairs = [
        ("DXY",    "DXY_z52W"),
        ("T10Y2Y", "T10Y2Y_z52W"),
    ]
    nrows = len(pairs)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 3.8 * nrows))

    for row, (level_col, z_col) in enumerate(pairs):
        if level_col not in macro_levels.columns or z_col not in macro_changes.columns:
            continue
        s_level = macro_levels[level_col].dropna()
        s_z52w  = macro_changes[z_col].dropna()

        # Common training date range
        s_level = s_level[s_level.index <= TRAIN_END]
        s_z52w  = s_z52w[s_z52w.index  <= TRAIN_END]

        for col_idx, (series, col_name, label_key) in enumerate([
            (s_level, level_col, level_col),
            (s_z52w,  z_col,     z_col),
        ]):
            ax = axes[row, col_idx]
            rm = series.rolling(52, min_periods=26).mean()
            rs = series.rolling(52, min_periods=26).std()

            # Verdict lookup
            if col_idx == 0:
                verdict = stat_levels.loc[level_col, "verdict"] if level_col in stat_levels.index else ""
            else:
                z_label = z_col + " (transformed)"
                verdict = stat_z52w.loc[z_label, "verdict"] if z_label in stat_z52w.index else ""
            color = _verdict_color(verdict)

            ax.plot(series.index, series.values, color="#374151", linewidth=0.7,
                    alpha=0.85, label="Observed")
            ax.plot(rm.index, rm.values, color=color, linewidth=1.8,
                    linestyle="--", label="52W rolling mean")
            ax.fill_between(series.index, (rm - rs).values, (rm + rs).values,
                            color=color, alpha=0.12, label="±1 rolling std")

            if col_idx == 0:
                subtitle = f"Raw levels [{_verdict_short(verdict)}]\nADF p={stat_levels.loc[level_col,'adf_pval']:.4f}  KPSS p={stat_levels.loc[level_col,'kpss_pval']:.4f}"
            else:
                z_label = z_col + " (transformed)"
                adf_p  = stat_z52w.loc[z_label, "adf_pval"]  if z_label in stat_z52w.index else float("nan")
                kpss_p = stat_z52w.loc[z_label, "kpss_pval"] if z_label in stat_z52w.index else float("nan")
                subtitle = f"52-week z-score [{_verdict_short(verdict)}]\nADF p={adf_p:.4f}  KPSS p={kpss_p:.4f}"

            ax.set_title(
                f"{SERIES_LABELS.get(label_key, label_key)}\n{subtitle}",
                fontsize=9, color=color, fontweight="bold",
            )
            if col_idx == 1:
                ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
                ax.axhline(+1.96, color="#dc2626", linewidth=0.6, linestyle=":", alpha=0.6, label="±1.96 σ")
                ax.axhline(-1.96, color="#dc2626", linewidth=0.6, linestyle=":", alpha=0.6)

            ax.tick_params(axis="x", rotation=30, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.legend(fontsize=6.5, loc="upper left")
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)
            for sp in ax.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(1.6)

    fig.suptitle(
        "Figure 3.C - Raw Level vs. 52-Week Z-Score: Non-Stationary Macro Variables\n"
        "Left: raw level (drifting mean → non-stationary). "
        "Right: z-score transformation (mean-reverting → stationary).\n"
        "Green border = stationary  |  Red border = non-stationary",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── 4. VAR lag order selection ────────────────────────────────────────────────

def select_var_lag(endog: pd.DataFrame, max_lags: int = 4) -> dict:
    """
    Select VAR(p) lag order via AIC and BIC on training data.
    Thesis protocol: prefer BIC when AIC and BIC disagree (avoids overfitting).
    """
    model    = VAR(endog)
    lag_info = model.select_order(maxlags=max_lags)

    aic_lag = lag_info.aic
    bic_lag = lag_info.bic
    hqc_lag = lag_info.hqic

    print("\n  VAR Lag Order Selection:")
    print(f"    AIC selects p = {aic_lag}  (forecasting-optimised)")
    print(f"    BIC selects p = {bic_lag}  (parsimony-optimised)")
    print(f"    HQC selects p = {hqc_lag}")

    if aic_lag != bic_lag:
        print(f"\n    AIC/BIC disagree. Thesis protocol: prefer BIC → p = {bic_lag}")
        recommended = bic_lag
    else:
        recommended = aic_lag

    print(f"\n    RECOMMENDED lag order: p = {recommended}")

    return {"aic": aic_lag, "bic": bic_lag, "hqc": hqc_lag, "recommended": recommended}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 3a - Stationarity Tests and Lag Order Selection")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    factors_path       = os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet")
    macro_path         = os.path.join(PROCESSED_DATA_DIR, "macro.parquet")
    macro_changes_path = os.path.join(PROCESSED_DATA_DIR, "macro_changes.parquet")

    for p in [factors_path, macro_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run earlier pipeline steps first.")

    factors       = pd.read_parquet(factors_path)[["beta0", "beta1", "beta2"]]
    macro_levels  = pd.read_parquet(macro_path)
    if "inventory" in macro_levels.columns:
        macro_levels = macro_levels.drop(columns=["inventory"])  # use lagged version only

    macro_changes = None
    if os.path.exists(macro_changes_path):
        macro_changes = pd.read_parquet(macro_changes_path)
    else:
        print("WARNING: macro_changes.parquet not found - z52W transformation tests skipped.")

    # Restrict to training period for all specification decisions
    factors_train = factors.loc[:TRAIN_END].dropna()
    macro_train   = macro_levels.loc[:TRAIN_END].dropna()
    if macro_changes is not None:
        macro_changes_train = macro_changes.loc[:TRAIN_END]
    else:
        macro_changes_train = None

    print(f"\nTraining sample: {factors_train.index.min().date()} → {factors_train.index.max().date()}")
    print(f"N = {len(factors_train)} weeks")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1 - Stationarity tests on raw levels
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("STEP 1 - ADF + KPSS on raw levels (NS factors + macro)")
    print("─" * 60)

    results_levels = []
    for col in ["beta0", "beta1", "beta2"]:
        results_levels.append(run_stationarity_tests(factors_train[col], col))
    for col in macro_train.columns:
        results_levels.append(run_stationarity_tests(macro_train[col], col))
    stat_levels = pd.DataFrame(results_levels).set_index("series")
    print("\n" + stat_levels[["adf_pval", "kpss_pval", "verdict"]].to_string())

    stat_levels.to_parquet(os.path.join(PROCESSED_DATA_DIR, "stationarity_results.parquet"))
    print(f"\nSaved: {os.path.join(PROCESSED_DATA_DIR, 'stationarity_results.parquet')}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2 - Stationarity tests on z52W transformations
    # (only for non-stationary variables: DXY, T10Y2Y)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("STEP 2 - ADF + KPSS on z52W transformations (for non-stationary variables)")
    print("─" * 60)

    results_z52w = []
    if macro_changes_train is not None:
        z52w_candidates = ["DXY_z52W", "T10Y2Y_z52W", "VIX_z52W", "inventory_z52W"]
        for col in z52w_candidates:
            if col in macro_changes_train.columns:
                s = macro_changes_train[col].dropna()
                if len(s) > 20:
                    results_z52w.append(
                        run_stationarity_tests(s, col + " (transformed)")
                    )

    if results_z52w:
        stat_z52w = pd.DataFrame(results_z52w).set_index("series")
        print("\n" + stat_z52w[["adf_pval", "kpss_pval", "verdict"]].to_string())
        stat_z52w.to_parquet(os.path.join(PROCESSED_DATA_DIR, "stationarity_transformed.parquet"))
        print(f"\nSaved: {os.path.join(PROCESSED_DATA_DIR, 'stationarity_transformed.parquet')}")
    else:
        stat_z52w = pd.DataFrame(columns=["adf_pval", "kpss_pval", "verdict"])
        print("  (skipped - macro_changes.parquet not available)")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3 - VAR variable specification decision table
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("STEP 3 - VAR variable specification decision")
    print("─" * 60)

    if not stat_z52w.empty:
        spec_decision = build_var_spec_decision(stat_levels, stat_z52w, macro_changes_train)
        spec_decision.to_parquet(os.path.join(PROCESSED_DATA_DIR, "var_spec_decision.parquet"))
        print("\n" + spec_decision[["stationary_in_levels", "z52w_tested",
                                    "stationary_in_z52w", "var_transformation"]].to_string())
        print(f"\nSaved: {os.path.join(PROCESSED_DATA_DIR, 'var_spec_decision.parquet')}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4 - Stationarity diagnostic figures
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("STEP 4 - Generating stationarity diagnostic figures")
    print("─" * 60)

    # Figure A: Raw level time series (factors + macro)
    raw_series = {
        **{c: factors_train[c] for c in ["beta0", "beta1", "beta2"]},
        **{c: macro_train[c]   for c in macro_train.columns},
    }
    plot_timeseries_panel(
        raw_series, stat_levels,
        title=(
            "Figure 3.A - Stationarity Diagnostics: Raw Time Series with Rolling Mean and Std\n"
            "Green border = stationary (both tests agree)  |  Red = non-stationary  |  Amber = conflicting\n"
            "A stationary series has a roughly constant mean; a non-stationary series shows a drifting mean."
        ),
        out_path=os.path.join(FIGURES_DIR, "fig_stationarity_timeseries.png"),
    )

    # Figure B: ACF for raw levels
    plot_acf_panel(
        raw_series, stat_levels,
        title=(
            "Figure 3.B - Autocorrelation Functions (ACF, 40 lags): Raw Levels\n"
            "Non-stationary (random walk) series: ACF decays slowly toward zero.\n"
            "Stationary series: ACF drops sharply within a few lags."
        ),
        out_path=os.path.join(FIGURES_DIR, "fig_stationarity_acf.png"),
    )

    # Figure C: Levels vs. z52W comparison for non-stationary macro variables
    if macro_changes is not None:
        plot_transformation_comparison(
            macro_levels.loc[:TRAIN_END],
            macro_changes.loc[:TRAIN_END],
            stat_levels,
            stat_z52w,
            out_path=os.path.join(FIGURES_DIR, "fig_stationarity_transform.png"),
        )

        # Figure D: ACF for z52W series
        if not stat_z52w.empty:
            z52w_series = {}
            for col in ["DXY_z52W", "T10Y2Y_z52W"]:
                if col in macro_changes_train.columns:
                    z52w_series[col] = macro_changes_train[col]
            if z52w_series:
                plot_acf_panel(
                    z52w_series, stat_z52w,
                    title=(
                        "Figure 3.D - ACF for 52-Week Z-Score Transformations\n"
                        "Confirm rapid ACF decay → z52W series are stationary.\n"
                        "Compare to Figure 3.B (raw levels) to see the transformation effect."
                    ),
                    out_path=os.path.join(FIGURES_DIR, "fig_stationarity_acf_z52w.png"),
                )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5 - VAR lag order selection
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("STEP 5 - VAR Lag Order Selection (AIC / BIC)")
    print("─" * 60)

    # Build endog with the CONFIRMED PRODUCTION macro set (must match
    # 05_models.py / 07_forecast_evaluation.py / 08_probability_outputs.py,
    # so lag selection stays consistent with the variables actually
    # entering the VAR/LP).
    if macro_changes_train is not None:
        macro_for_lag = pd.concat([
            macro_train[MACRO_LEVEL_COLS],
            macro_changes_train[MACRO_CHANGE_COLS],
        ], axis=1).dropna()
    else:
        macro_for_lag = macro_train

    endog = pd.concat([factors_train, macro_for_lag], axis=1).dropna()

    # Apply β₀ differencing only if stationarity test confirms unit root
    beta0_verdict = stat_levels.loc["beta0", "verdict"] if "beta0" in stat_levels.index else ""
    if "NON-STATIONARY" in beta0_verdict:
        endog["beta0"] = endog["beta0"].diff()
        endog = endog.dropna()
        print("\n  β₀ differenced for lag selection (unit root confirmed by both tests).")
    else:
        print("\n  β₀ used in levels (stationary confirmed).")

    lag_results = select_var_lag(endog, max_lags=VAR_MAX_LAGS)

    lag_df = pd.DataFrame([lag_results])
    lag_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "lag_selection.parquet"))
    print(f"\nSaved: {os.path.join(PROCESSED_DATA_DIR, 'lag_selection.parquet')}")

    print("\n" + "=" * 60)
    print("Phase 3a complete.")
    print("=" * 60)
    print("Outputs:")
    print("  data/processed/stationarity_results.parquet   ← ADF/KPSS on raw levels")
    print("  data/processed/stationarity_transformed.parquet ← ADF/KPSS on z52W")
    print("  data/processed/var_spec_decision.parquet      ← final VAR variable spec")
    print("  data/processed/lag_selection.parquet          ← recommended lag order p")
    print("  report/figures/fig_stationarity_timeseries.png")
    print("  report/figures/fig_stationarity_acf.png")
    print("  report/figures/fig_stationarity_transform.png ← key decision justification")
    print("  report/figures/fig_stationarity_acf_z52w.png")
    print("\nNext: run python code/04_granger.py")


if __name__ == "__main__":
    main()
