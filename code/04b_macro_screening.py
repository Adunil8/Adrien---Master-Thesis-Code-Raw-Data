"""
04b_macro_screening.py - Comprehensive macro variable screening.

PURPOSE:
    Systematic audit of ALL available macro data against NS factors (β₀, β₁, β₂).
    Tests every candidate variable with every plausible transformation at every lag.
    Produces a complete evidence table demonstrating that variable selection was
    data-driven, not arbitrary - required for thesis defence.

CANDIDATES TESTED:
    DXY, VIX, Inventory, COT_net_spec_pct_oi, CNY/USD, AUD/USD,
    Brent crude, US2Y, US3M, T10Y2Y, GPR

TRANSFORMATIONS:
    level    - raw value (only for variables that pass ADF stationarity)
    d1W      - 1-week absolute change
    d4W      - 4-week absolute change (~1 month)
    d13W     - 13-week absolute change (~3 months, one deal tenor)
    d26W     - 26-week absolute change (~6 months)
    z52W     - 52-week rolling z-score (deviation from trailing-year norm)

GRANGER CAUSALITY:
    Each (variable × transformation) → each NS factor (β₀, β₁, β₂)
    Tested at max_lags = 1, 2, 3, 4 weeks.
    Reports: best p-value across all tested lag orders (for screening purposes).
    Conservative selection: min p-value across lag orders.

OUTPUTS:
    data/processed/macro_screening_full.parquet   - full results grid
    data/processed/macro_all_transformed.parquet  - all transformed variables
    report/figures/fig_macro_screening_heatmap.png - p-value heatmap
    report/figures/fig_macro_screening_timeseries.png - candidate time series
    (tables for thesis are generated inline - copy to main.qmd)

Run: python code/04b_macro_screening.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from config import PROCESSED_DATA_DIR, FIGURES_DIR, TRAIN_END

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 - Load and resample all raw macro data to weekly Friday close
# ─────────────────────────────────────────────────────────────────────────────

def load_fred_csv(path: str, name: str) -> pd.Series:
    """Load a FRED CSV (date index, one value column) → weekly Friday close."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [name]
    df = df.replace(".", np.nan).astype(float)
    # Resample: keep only weekdays, then take last Friday value per week
    daily = df[df.index.dayofweek < 5]
    weekly = daily.resample("W-FRI").last()
    return weekly[name]


def load_all_macro() -> pd.DataFrame:
    """Load all candidate macro variables and align to weekly Friday index."""
    raw_dir = "data/raw"
    series = {}

    # ── FRED variables ──────────────────────────────────────────────────────
    fred_map = {
        "DXY":    "fred_dxy.csv",
        "VIX":    "fred_vix.csv",
        "US10Y":  "fred_us10y.csv",
        "US2Y":   "fred_us2y.csv",
        "US3M":   "fred_us3m.csv",
        "T10Y2Y": "fred_t10y2y.csv",
        "AUDUSD": "fred_audusd.csv",
        "CNYUSD": "fred_cnyusd.csv",
        "BRENT":  "fred_brent.csv",
        # DFII10 - 10Y TIPS yield (real interest rate). Coverage 2006–2026.
        # Tested as cost-of-carry proxy; DFII10_z52W → β₁ p=0.012 ** but
        # superseded by US3M_d26W (p=0.009 ***). Excluded on parsimony grounds.
        "DFII10": "fred_dfii10.csv",
        # TEDRATE (TED spread) - unavailable post-June 2023 (LIBOR discontinued).
        # Cannot be used in test period (2023–2026) for real-time forecasting.
        # Excluded on operational grounds; omitted from screening loop below.
        # "TEDRATE": "fred_tedrate.csv",   # file not present
    }
    for name, fname in fred_map.items():
        path = os.path.join(raw_dir, fname)
        if os.path.exists(path):
            series[name] = load_fred_csv(path, name)
        else:
            print(f"  WARNING: {path} not found - skipping {name}")

    # ── LME Inventory - load from processed macro.parquet (full 2006–2026 history)
    # Note: raw LME xlsx only contains partial 2026 extract from Bloomberg.
    # Full weekly history was built in 01_data_cleaning.py and stored in macro.parquet.
    macro_proc = os.path.join(PROCESSED_DATA_DIR, "macro.parquet")
    if os.path.exists(macro_proc):
        macro_existing = pd.read_parquet(macro_proc)
        if "inventory" in macro_existing.columns:
            series["Inventory"] = macro_existing["inventory"]

    # ── CFTC COT - net speculative position as % of open interest ───────────
    cot_path = os.path.join(PROCESSED_DATA_DIR, "cot_copper_weekly.parquet")
    if os.path.exists(cot_path):
        cot = pd.read_parquet(cot_path)
        series["COT_net_pct"] = cot["net_spec_pct_oi"]   # % of open interest
        series["COT_net_abs"] = cot["net_speculative"]    # raw contracts

    # ── GPR (Caldara & Iacoviello 2022) ─────────────────────────────────────
    gpr_path = os.path.join(raw_dir, "data_gpr_daily_recent.xlsx")
    if os.path.exists(gpr_path):
        gpr_raw = pd.read_excel(gpr_path)
        # The file has a transposed layout - find the GPRD row
        gpr_row = gpr_raw[gpr_raw.iloc[:, 9] == "GPRD"]
        if len(gpr_row) > 0:
            # The 'date' column is col 5 (index 5)
            dates = pd.to_datetime(gpr_raw["date"], errors="coerce")
            vals  = pd.to_numeric(gpr_raw["GPRD"], errors="coerce")
            gpr_s = pd.Series(vals.values, index=dates).dropna().sort_index()
        else:
            # Alternative: use GPRD column directly if present
            gpr_raw2 = pd.read_excel(gpr_path)
            dates = pd.to_datetime(gpr_raw2.iloc[:, 5], errors="coerce")
            gprd_col = gpr_raw2.columns.get_loc("GPRD") if "GPRD" in gpr_raw2.columns else None
            if gprd_col is not None:
                vals = pd.to_numeric(gpr_raw2.iloc[:, gprd_col], errors="coerce")
                gpr_s = pd.Series(vals.values, index=dates).dropna().sort_index()
            else:
                gpr_s = pd.Series(dtype=float)
        if len(gpr_s) > 0:
            gpr_daily = gpr_s[gpr_s.index.dayofweek < 5]
            series["GPR"] = gpr_daily.resample("W-FRI").last()

    # ── Combine ──────────────────────────────────────────────────────────────
    macro = pd.DataFrame(series)
    macro.index.name = "date"
    macro = macro.sort_index()
    return macro


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 - ADF stationarity test (quick screen for level use)
# ─────────────────────────────────────────────────────────────────────────────

def quick_adf(s: pd.Series) -> float:
    """Return ADF p-value on non-NaN values. Lower = more stationary."""
    clean = s.dropna()
    if len(clean) < 30:
        return np.nan
    try:
        return adfuller(clean, autolag="AIC")[1]
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 - Compute all transformations
# ─────────────────────────────────────────────────────────────────────────────

WINDOWS = {
    "d1W":  1,
    "d4W":  4,
    "d13W": 13,
    "d26W": 26,
}

def compute_transforms(macro: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each variable: compute level (if ADF-stationary) + all change windows + z52W.
    Returns:
        transformed   - DataFrame with all (var, transform) columns
        stationarity  - DataFrame with ADF p-values per variable and transform
    """
    records   = {}
    stat_rows = []

    for col in macro.columns:
        s = macro[col].dropna()
        adf_level = quick_adf(s)
        is_stationary_level = (adf_level < 0.10) if not np.isnan(adf_level) else False

        # Level - only if stationary
        if is_stationary_level:
            key = f"{col}_level"
            records[key] = macro[col]
            stat_rows.append({"variable": col, "transform": "level",
                               "adf_p": round(adf_level, 4), "use_level": True})
        else:
            stat_rows.append({"variable": col, "transform": "level",
                               "adf_p": round(adf_level, 4) if not np.isnan(adf_level) else 99,
                               "use_level": False})

        # Absolute changes - always stationary (or near-so) for I(1) series
        for tname, w in WINDOWS.items():
            delta = macro[col].diff(w)
            key   = f"{col}_{tname}"
            records[key] = delta
            adf_d = quick_adf(delta)
            stat_rows.append({"variable": col, "transform": tname,
                               "adf_p": round(adf_d, 4) if not np.isnan(adf_d) else 99,
                               "use_level": True})

        # 52-week rolling z-score (deviation from trailing-year norm)
        roll_mean = macro[col].rolling(52, min_periods=26).mean()
        roll_std  = macro[col].rolling(52, min_periods=26).std()
        z52w      = (macro[col] - roll_mean) / roll_std.replace(0, np.nan)
        key_z     = f"{col}_z52W"
        records[key_z] = z52w
        adf_z = quick_adf(z52w)
        stat_rows.append({"variable": col, "transform": "z52W",
                           "adf_p": round(adf_z, 4) if not np.isnan(adf_z) else 99,
                           "use_level": True})

    transformed = pd.DataFrame(records)
    transformed.index.name = "date"
    stationarity = pd.DataFrame(stat_rows)
    return transformed, stationarity


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 - Granger causality screening
# ─────────────────────────────────────────────────────────────────────────────

MAX_LAGS_TEST = 4   # test lags 1–4; report best (minimum) p-value

def granger_min_pval(cause: pd.Series, effect: pd.Series,
                     max_lags: int = MAX_LAGS_TEST) -> tuple[float, int]:
    """
    Run Granger causality test at lags 1…max_lags.
    Returns (min_pval, best_lag) across all tested lag orders.
    Uses SSR chi2 test p-value.
    """
    df = pd.DataFrame({"effect": effect, "cause": cause}).dropna()
    if len(df) < max_lags + 30:
        return np.nan, -1
    try:
        res = grangercausalitytests(df[["effect", "cause"]], maxlag=max_lags,
                                    verbose=False)
        pvals = [(lag, res[lag][0]["ssr_chi2test"][1]) for lag in range(1, max_lags + 1)]
        best_lag, best_p = min(pvals, key=lambda x: x[1])
        return round(best_p, 6), best_lag
    except Exception:
        return np.nan, -1


def run_full_screening(transformed: pd.DataFrame, factors: pd.DataFrame,
                       train_end: str) -> pd.DataFrame:
    """
    For each column in `transformed`, run Granger → each of β₀, β₁, β₂.
    Uses TRAINING DATA ONLY to avoid look-ahead.
    Returns DataFrame with one row per (macro_col, factor) combination.
    """
    # Restrict to training sample only
    tr = transformed.loc[:train_end]
    fa = factors.loc[:train_end, ["beta0", "beta1", "beta2"]]

    results = []
    n_cols = len(transformed.columns)
    for i, col in enumerate(transformed.columns):
        if i % 15 == 0:
            print(f"  Screening {i}/{n_cols}...", end="\r", flush=True)
        for fac in ["beta0", "beta1", "beta2"]:
            pval, best_lag = granger_min_pval(tr[col], fa[fac])
            # Parse variable name and transform from column name
            # Convention: {variable}_{transform}
            parts = col.rsplit("_", 1)
            var_name  = parts[0] if len(parts) == 2 else col
            transform = parts[1] if len(parts) == 2 else "?"
            results.append({
                "macro_col":  col,
                "variable":   var_name,
                "transform":  transform,
                "factor":     fac,
                "granger_p":  pval,
                "best_lag":   best_lag,
            })
    print(f"  Screening {n_cols}/{n_cols}... done.")
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 - Build summary table (pivot: variable × factor, best p-value across transforms)
# ─────────────────────────────────────────────────────────────────────────────

def significance_stars(p: float) -> str:
    if np.isnan(p):      return "-"
    if p < 0.01:         return "***"
    if p < 0.05:         return "**"
    if p < 0.10:         return "*"
    return "n.s."


def build_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """
    For each variable: find the BEST (minimum) p-value across all transforms,
    and record which transform achieved it.
    """
    rows = []
    variables = results["variable"].unique()
    for var in sorted(variables):
        sub = results[results["variable"] == var]
        row = {"Variable": var}
        for fac in ["beta0", "beta1", "beta2"]:
            fsub = sub[sub["factor"] == fac].dropna(subset=["granger_p"])
            if fsub.empty:
                row[f"{fac}_best_p"] = np.nan
                row[f"{fac}_best_transform"] = "-"
                row[f"{fac}_stars"] = "-"
            else:
                idx = fsub["granger_p"].idxmin()
                best = fsub.loc[idx]
                row[f"{fac}_best_p"] = best["granger_p"]
                row[f"{fac}_best_transform"] = best["transform"]
                row[f"{fac}_stars"] = significance_stars(best["granger_p"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_full_pivot(results: pd.DataFrame) -> pd.DataFrame:
    """
    Full pivot: rows = (variable, transform), cols = (factor, p-value).
    """
    pivot = results.pivot_table(
        index=["variable", "transform"],
        columns="factor",
        values="granger_p",
        aggfunc="min",
    ).round(4)
    return pivot


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 - Figures
# ─────────────────────────────────────────────────────────────────────────────

FACTOR_LABELS = {"beta0": "β₀ (Level)", "beta1": "β₁ (Slope)", "beta2": "β₂ (Curvature)"}
PALETTE = ["#003366", "#779ecb", "#aec6cf", "#cccccc"]


def plot_heatmap(results: pd.DataFrame, out_dir: str) -> None:
    """
    Heatmap: rows = (variable, transform), columns = NS factor.
    Cell colour = -log10(p-value): darker blue = more significant.
    """
    pivot = results.pivot_table(
        index=["variable", "transform"],
        columns="factor",
        values="granger_p",
        aggfunc="min",
    )
    # Reorder columns
    pivot = pivot.reindex(columns=["beta0", "beta1", "beta2"])
    pivot.columns = ["β₀ (Level)", "β₁ (Slope)", "β₂ (Curvature)"]

    # Sort rows by minimum p-value across factors
    pivot["_min"] = pivot.min(axis=1)
    pivot = pivot.sort_values("_min").drop(columns="_min")

    nlog = -np.log10(pivot.clip(lower=1e-6))  # transform for colour scaling

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.32)))

    cmap = plt.cm.Blues
    im = ax.imshow(nlog.values, cmap=cmap, aspect="auto",
                   vmin=0, vmax=max(3, nlog.values[~np.isnan(nlog.values)].max() + 0.5))

    # Annotate cells with stars
    for r in range(len(pivot)):
        for c in range(3):
            p = pivot.iloc[r, c]
            stars = significance_stars(p)
            p_str = f"{p:.3f}\n{stars}" if not np.isnan(p) else "-"
            color = "white" if (not np.isnan(p) and p < 0.05) else "black"
            ax.text(c, r, p_str, ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="bold" if stars != "n.s." else "normal")

    ax.set_xticks(range(3))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(
        [f"{v}  [{t}]" for v, t in pivot.index],
        fontsize=8, fontfamily="monospace"
    )
    ax.set_title("Granger Causality Screening - All Macro Candidates\n"
                 "p-value (best lag 1–4), training sample 2006–2022\n"
                 "Darker = more significant | *** p<0.01 | ** p<0.05 | * p<0.10",
                 fontsize=10, pad=12)

    # Significance threshold lines (visual guides)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("−log₁₀(p-value)", fontsize=9)
    cbar.set_ticks([1, 1.301, 2, 3])
    cbar.set_ticklabels(["p=0.10 (*)", "p=0.05 (**)", "p=0.01 (***)", "p=0.001"])

    plt.tight_layout()
    path = os.path.join(out_dir, "fig_macro_screening_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_macro_screening_heatmap.png")


def plot_candidate_timeseries(macro: pd.DataFrame, factors: pd.DataFrame,
                              out_dir: str) -> None:
    """
    Time series of all macro candidates (normalised) alongside NS factors.
    Helps visually verify direction of relationships.
    """
    candidates_level = ["DXY", "VIX", "Inventory", "COT_net_pct",
                        "CNYUSD", "AUDUSD", "BRENT", "US2Y", "US3M",
                        "T10Y2Y", "GPR"]
    present = [c for c in candidates_level if c in macro.columns]

    n = len(present)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 2.8), sharex=True)
    axes = axes.flatten()

    fac_normalised = (factors[["beta0", "beta1", "beta2"]] - factors[["beta0", "beta1", "beta2"]].mean()) \
                    / factors[["beta0", "beta1", "beta2"]].std()

    factor_colors = {"beta0": "#003366", "beta1": "#779ecb", "beta2": "#aec6cf"}

    for ax, col in zip(axes, present):
        s = macro[col].dropna()
        s_norm = (s - s.mean()) / s.std()
        ax.plot(s_norm.index, s_norm.values, color="#cc3300", lw=1.2,
                label=col, alpha=0.85)
        # Overlay NS factors (light, secondary)
        for fac, fc in factor_colors.items():
            ax.plot(fac_normalised.index, fac_normalised[fac].values,
                    color=fc, lw=0.7, alpha=0.35)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.set_ylim(-4, 4)
        ax.grid(alpha=0.2)

    # Hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    # Legend (factor colours)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#cc3300", lw=1.5, label="Macro variable (z-normalised)"),
        Line2D([0], [0], color="#003366", lw=0.9, label="β₀ (Level)"),
        Line2D([0], [0], color="#779ecb", lw=0.9, label="β₁ (Slope)"),
        Line2D([0], [0], color="#aec6cf", lw=0.9, label="β₂ (Curvature)"),
    ]
    fig.legend(handles=legend_elements, loc="lower right", ncol=2, fontsize=8,
               bbox_to_anchor=(0.98, 0.01))
    fig.suptitle("Macro Candidate Time Series vs NS Factors (z-normalised)\n"
                 "Red line = macro variable; blue shades = β₀, β₁, β₂",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_macro_screening_timeseries.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_macro_screening_timeseries.png")


def plot_correlation_bar(results: pd.DataFrame, out_dir: str) -> None:
    """
    Bar chart: for each variable, best Granger p-value across all transforms and factors.
    Annotates with which (transform, factor) combination was best.
    """
    best = (results.groupby("variable")["granger_p"]
                   .min()
                   .sort_values()
                   .dropna())

    # idxmin per variable - skip groups where all values are NaN
    idx = results.groupby("variable")["granger_p"].apply(
        lambda s: s.idxmin() if not s.isna().all() else None
    ).dropna()
    best_meta = results.loc[idx.values, ["variable", "transform", "factor", "granger_p"]]
    best_meta = best_meta.set_index("variable")
    # Restrict to variables that appear in `best`
    best_meta = best_meta.loc[best_meta.index.isin(best.index)]
    best_meta = best_meta.loc[best.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    neg_log = -np.log10(best.clip(lower=1e-6))
    colors = ["#003366" if p < 0.01 else "#779ecb" if p < 0.05
              else "#aec6cf" if p < 0.10 else "#dddddd" for p in best.values]
    bars = ax.barh(range(len(best)), neg_log.values, color=colors)
    ax.set_yticks(range(len(best)))
    ax.set_yticklabels(best.index, fontsize=9)
    ax.axvline(-np.log10(0.10), color="#cc8800", ls="--", lw=1.2, label="p=0.10 (*)")
    ax.axvline(-np.log10(0.05), color="#cc3300", ls="--", lw=1.2, label="p=0.05 (**)")
    ax.axvline(-np.log10(0.01), color="#800000", ls="--", lw=1.2, label="p=0.01 (***)")

    for i, (var, row) in enumerate(best_meta.iterrows()):
        label = f"[{row['transform']} → {row['factor']}, p={row['granger_p']:.3f}]"
        ax.text(neg_log[var] + 0.05, i, label, va="center", fontsize=7.5, color="#333333")

    ax.set_xlabel("−log₁₀(best Granger p-value)", fontsize=10)
    ax.set_title("Macro Variable Screening - Best Granger p-Value per Variable\n"
                 "Best transformation and target factor annotated | Training 2006–2022",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, neg_log.max() + 1.5)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_macro_screening_ranking.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_macro_screening_ranking.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 - Print thesis-ready table
# ─────────────────────────────────────────────────────────────────────────────

def print_thesis_table(summary: pd.DataFrame) -> None:
    """Print a clean table suitable for direct inclusion in main.qmd."""
    hdr = (f"\n{'Variable':<16} | {'β₀ (Level)':<22} | {'β₁ (Slope)':<22} | {'β₂ (Curvature)':<22}")
    print("─" * 90)
    print("MACRO SCREENING SUMMARY - Best Granger p-value per variable across all transforms")
    print("Training sample 2006–2022 | Best lag order 1–4 | *** p<0.01 ** p<0.05 * p<0.10")
    print("─" * 90)
    print(hdr)
    print("─" * 90)
    for _, row in summary.iterrows():
        var = row["Variable"]
        def fmt(fac):
            p    = row[f"{fac}_best_p"]
            t    = row[f"{fac}_best_transform"]
            s    = row[f"{fac}_stars"]
            if np.isnan(p): return f"{'-':<22}"
            return f"{s:>4}  p={p:.4f} [{t}]"[:22].ljust(22)
        print(f"{var:<16} | {fmt('beta0')} | {fmt('beta1')} | {fmt('beta2')}")
    print("─" * 90)
    print("Legend: [level]=raw level [d1W/4W/13W/26W]=N-week change [z52W]=52W z-score")


def print_detailed_table(results: pd.DataFrame) -> None:
    """Print full detail table per (variable, transform) for appendix."""
    print("\n" + "═" * 90)
    print("DETAILED GRANGER RESULTS - All Variables × All Transforms × All NS Factors")
    print("Training sample 2006–2022 | Best lag 1–4 reported")
    print("═" * 90)
    hdr = f"{'Variable':<14} {'Transform':<8} | {'→β₀':>12} {'lag':>4} | {'→β₁':>12} {'lag':>4} | {'→β₂':>12} {'lag':>4}"
    print(hdr)
    print("─" * 90)

    for var in sorted(results["variable"].unique()):
        sub_var = results[results["variable"] == var]
        for transform in ["level", "d1W", "d4W", "d13W", "d26W", "z52W"]:
            sub_t = sub_var[sub_var["transform"] == transform]
            if sub_t.empty:
                continue
            cells = {}
            for _, r in sub_t.iterrows():
                cells[r["factor"]] = (r["granger_p"], r["best_lag"])
            def fmt_cell(fac):
                if fac not in cells: return f"{'-':>12} {'-':>4}"
                p, lag = cells[fac]
                if np.isnan(p): return f"{'-':>12} {'-':>4}"
                stars = significance_stars(p)
                return f"{p:>8.4f} {stars:>3} {int(lag):>4}"
            print(f"{var:<14} {transform:<8} | {fmt_cell('beta0')} | "
                  f"{fmt_cell('beta1')} | {fmt_cell('beta2')}")
        print("─" * 90)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("=" * 60)
    print("Macro Variable Comprehensive Screening")
    print("=" * 60)

    # Step 1 - Load
    print("\nStep 1: Loading all macro candidates...")
    macro = load_all_macro()
    print(f"  Loaded {len(macro.columns)} variables: {list(macro.columns)}")
    print(f"  Date range: {macro.index.min().date()} → {macro.index.max().date()}")

    # Step 2 - Load NS factors
    factors = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet"))
    factors = factors[["beta0", "beta1", "beta2"]]

    # Step 3 - Compute all transformations
    print("\nStep 2: Computing transformations (level, d1W, d4W, d13W, d26W, z52W)...")
    transformed, stationarity = compute_transforms(macro)
    print(f"  Generated {len(transformed.columns)} (variable × transform) columns")

    # ADF summary
    stationary_levels = stationarity[stationarity["use_level"] & (stationarity["transform"] == "level")]
    print(f"\n  Variables stationary at level (ADF p<0.10): "
          f"{list(stationary_levels[stationary_levels['adf_p'] < 0.10]['variable'])}")

    # Save transformed
    out_tr = os.path.join(PROCESSED_DATA_DIR, "macro_all_transformed.parquet")
    transformed.to_parquet(out_tr)
    print(f"  Saved: {out_tr}")

    # Step 4 - Granger screening
    print("\nStep 3: Running Granger causality screening (all combos × 3 NS factors)...")
    print(f"  Total tests: {len(transformed.columns)} × 3 = {len(transformed.columns)*3}")
    results = run_full_screening(transformed, factors, TRAIN_END)
    print(f"  Completed {len(results)} Granger tests")

    # Save full results
    out_res = os.path.join(PROCESSED_DATA_DIR, "macro_screening_full.parquet")
    results.to_parquet(out_res)
    print(f"  Saved: {out_res}")

    # Step 5 - Summary tables
    summary = build_summary_table(results)
    print_thesis_table(summary)

    print_detailed_table(results)

    # Save to Excel (for thesis appendix)
    full_pivot = build_full_pivot(results)
    excel_path = os.path.join(PROCESSED_DATA_DIR, "macro_screening_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Best_per_Variable", index=False)
        full_pivot.to_excel(writer, sheet_name="Full_Grid")
        stationarity.to_excel(writer, sheet_name="ADF_Stationarity", index=False)
        results.to_excel(writer, sheet_name="All_Results", index=False)
    print(f"\n  Saved Excel: {excel_path}")

    # Step 6 - Figures
    print("\nStep 4: Generating figures...")
    plot_heatmap(results, FIGURES_DIR)
    plot_candidate_timeseries(macro, factors, FIGURES_DIR)
    plot_correlation_bar(results, FIGURES_DIR)

    # Step 7 - Selection recommendation
    print("\n" + "═" * 60)
    print("SELECTION RECOMMENDATION")
    print("═" * 60)
    sig = results[results["granger_p"] < 0.10].copy()
    sig_sorted = (sig.sort_values("granger_p")
                     .groupby(["variable", "factor"])
                     .first()
                     .reset_index())
    print("\nAll (variable, factor) pairs with any significant Granger result (p<0.10):")
    print(f"{'Variable':<16} {'Transform':<10} {'→ Factor':<10} {'p-value':>10} {'Lag':>5} {'Stars'}")
    print("─" * 60)
    for _, row in sig_sorted.sort_values("granger_p").iterrows():
        stars = significance_stars(row["granger_p"])
        print(f"{row['variable']:<16} {row['transform']:<10} {row['factor']:<10} "
              f"{row['granger_p']:>10.4f} {int(row['best_lag']):>5}   {stars}")

    print("\nPhase complete. Review heatmap and ranking figures before updating VAR spec.")
    print("Next: update 05_models.py with confirmed macro selection.")


if __name__ == "__main__":
    main()
