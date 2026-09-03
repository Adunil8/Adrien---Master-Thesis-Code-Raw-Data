"""
00b_curve_and_macro_description.py - Opening data-description figures for
Chapter 4, built once directly from source data rather than reusing the
1-6M-restricted curves.parquet the rest of the pipeline works from.

WHY A SEPARATE SCRIPT: the core Nelson-Siegel fit and every downstream model
stay correctly restricted to the 1-6M trade-finance-relevant segment
(MATURITIES_MONTHS, config.py). But that restriction makes a poor opening
picture of the data: two nearby, short maturities show only a modest
backwardation/contango spread. The raw Bloomberg extraction actually
contains LP1 through LP24 with 100% coverage over the full 2006-2026
sample, so the descriptive opening of Chapter 4 uses that wider, unrestricted
view purely to make the curve-shape story visible, before the thesis
narrows to the 1-6M segment everything else is built on.

FOUR FIGURES:
  fig_curve_multimaturity.png - front month, 6M, 12M, and 24M overnight
                                 the full sample, so backwardation/contango
                                 crossings are visible directly in price
                                 levels, not just inferred from a spread.
  fig_curve_spread_regime.png - 12M-1M and 24M-1M spreads over time,
                                 shaded by sign, the raw-price precursor to
                                 the beta1 slope factor introduced later
                                 in Chapter 4.
  fig_macro_raw_appendix.png  - all five final macro variables, raw levels,
                                 for Appendix III.
  fig_macro_transformed_ch4.png - the same five variables in the transform
                                 each actually enters the model through
                                 (Section 3.1), to show the stationarity
                                 case visually alongside Section 3.3's ADF/
                                 KPSS tables.

DUPLICATED, NOT IMPORTED: the Bloomberg loading and weekly-resampling logic
mirrors 01_data_cleaning.py's load_bloomberg_futures/remove_weekends_and_
resample exactly, kept independent here for the same isolation reason as
every other numbered script in this codebase (see
02c_lambda_classification_validation.py's docstring).

Inputs : data/raw/Copper_Futures_Extraction_Bloomberg_Values.xlsx
         data/processed/macro.parquet
         data/processed/macro_changes.parquet
Outputs: report/figures/fig_curve_multimaturity.png
         report/figures/fig_curve_spread_regime.png
         report/figures/fig_macro_raw_appendix.png
         report/figures/fig_macro_transformed_ch4.png

Run: python code/00b_curve_and_macro_description.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import config
from config import PROCESSED_DATA_DIR, FIGURES_DIR, RAW_DATA_DIR, RAW_FUTURES_FILE

STRESS_PERIODS = [
    ("2008-09-01", "2009-03-31", "2008 GFC"),
    ("2020-02-01", "2020-05-31", "2020 COVID"),
    ("2022-02-01", "2022-12-31", "2022 Ukraine"),
    ("2025-04-01", "2025-12-31", "2025 Tariffs"),
]


def load_bloomberg_futures_full(filepath: str) -> pd.DataFrame:
    """Same parsing as 01_data_cleaning.py's load_bloomberg_futures, kept
    independent here. Returns every maturity column present in the sheet
    (m01 through m24), not restricted to MATURITIES_MONTHS."""
    xl = pd.ExcelFile(filepath)
    frames = []
    for sheet in xl.sheet_names:
        try:
            df = pd.read_excel(xl, sheet_name=sheet, skiprows=1, header=0)
            df = df.rename(columns={df.columns[0]: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            n_maturities = len(df.columns) - 1
            maturity_cols = [f"m{i:02d}" for i in range(1, n_maturities + 1)]
            df.columns = ["date"] + maturity_cols
            for col in maturity_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            frames.append(df)
        except Exception:
            continue
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values("date").drop_duplicates(subset="date", keep="last")
    full = full.set_index("date")
    full.index = pd.DatetimeIndex(full.index)
    return full


def remove_weekends_and_resample(df: pd.DataFrame, freq: str = "W-FRI") -> pd.DataFrame:
    df_weekdays = df[df.index.dayofweek < 5].copy()
    return df_weekdays.resample(freq).last()


def shade_stress_periods(ax):
    for start, end, label in STRESS_PERIODS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.12)


def plot_gradient_curve(curves: pd.DataFrame, cols: list, near: str, far: str,
                         title: str, out_path: str, shade: bool = True) -> None:
    """Same technique as the project's own 00_eda.py plot_prices: every
    intermediate maturity plotted thin and semi-transparent on a Blues
    gradient (darker = longer maturity), with only the nearest and
    farthest maturity bold and labelled. The overlapping thin lines read
    as a gradient band, without needing an explicit fill_between."""
    fig, ax = plt.subplots(figsize=(13, 4.8))
    if shade:
        shade_stress_periods(ax)
    palette = plt.cm.Blues(np.linspace(0.35, 0.95, len(cols)))
    for i, col in enumerate(cols):
        lw = 1.4 if col in (near, far) else 0.6
        alpha = 1.0 if col in (near, far) else 0.5
        label = col.upper() if col in (near, far) else None
        ax.plot(curves.index, curves[col], color=palette[i], linewidth=lw, alpha=alpha, label=label)
    ax.set_ylabel("USD/tonne")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.grid(lw=0.3, color="#dddddd")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


def plot_spread_regime(curves: pd.DataFrame, out_dir: str) -> None:
    # Defined as near minus far, matching beta1's own "short minus long"
    # convention (Section 3.2) exactly: positive here means the same thing
    # as beta1 > 0, backwardation. Getting this sign right matters, since
    # it is the same regime distinction the whole thesis is built on.
    # Restricted to 1M-6M only, the thesis's own scope (Appendix II).
    spread_1_6 = curves["m01"] - curves["m06"]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    shade_stress_periods(ax)
    ax.fill_between(spread_1_6.index, 0, spread_1_6, where=(spread_1_6 >= 0), color=config.COLORS["backwardation"],
                     alpha=0.6, label="Backwardation (near > far)")
    ax.fill_between(spread_1_6.index, 0, spread_1_6, where=(spread_1_6 < 0), color=config.COLORS["contango"],
                     alpha=0.6, label="Contango (near < far)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("1M - 6M (USD/tonne)")
    ax.grid(lw=0.3, color="#dddddd")
    ax.set_title("Term-Structure Spread, 1M-6M, Raw Prices, 2006-2026\n"
                 "The same backwardation/contango signal Chapter 4 later formalises as the NS slope factor")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_curve_spread_regime.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print("  Saved: fig_curve_spread_regime.png")


def plot_macro_panel(df: pd.DataFrame, cols_labels: list, title: str, out_path: str) -> None:
    # Panel height kept small (1.3in) so the 5-variable stack fits well
    # within a page alongside body text, rather than filling the page on
    # its own.
    n = len(cols_labels)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (col, label, color) in zip(axes, cols_labels):
        shade_stress_periods(ax)
        ax.plot(df.index, df[col], linewidth=0.8, color=color)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4) if df[col].min() < 0 else None
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(lw=0.3, color="#dddddd")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(title, y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("=" * 70)
    print("Chapter 4 opening figures: full-curve description and macro raw/transformed")
    print("=" * 70)

    raw_path = os.path.join(RAW_DATA_DIR, RAW_FUTURES_FILE)
    full_daily = load_bloomberg_futures_full(raw_path)
    curves_weekly = remove_weekends_and_resample(full_daily)
    print(f"Loaded {full_daily.shape[1]} maturities, {len(curves_weekly)} weekly observations, "
          f"{curves_weekly.index.min().date()} -> {curves_weekly.index.max().date()}")
    needed = ["m01", "m06", "m12", "m24"]
    missing = [c for c in needed if c not in curves_weekly.columns]
    if missing:
        print(f"[ERROR] Missing maturities in raw extraction: {missing}")
        sys.exit(1)

    full_cols = [f"m{i:02d}" for i in range(1, 25)]
    plot_gradient_curve(curves_weekly, full_cols, "m01", "m24",
                        "LME Copper Futures, Full Term Structure (1M-24M), 2006-2026\n"
                        "Broader context; the thesis itself works only with the 1-6M segment below",
                        os.path.join(FIGURES_DIR, "fig_curve_full_range.png"),
                        shade=False)

    scope_cols = [f"m{i:02d}" for i in range(1, 7)]
    plot_gradient_curve(curves_weekly, scope_cols, "m01", "m06",
                        "LME Copper Futures, 1-6M Segment, 2006-2026\n"
                        "The actual scope of this thesis (Appendix II)",
                        os.path.join(FIGURES_DIR, "fig_curve_scope_range.png"))

    plot_spread_regime(curves_weekly, FIGURES_DIR)

    macro_levels = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro.parquet"))
    macro_changes = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro_changes.parquet"))

    raw_cols = [
        ("DXY", "DXY (level)", "#8B4513"),
        ("inventory", "LME inventory (level)", "#4169E1"),
        ("US3M", "US 3M yield (level)", "#2E8B57"),
        ("GPR", "GPR index (level)", "#DC143C"),
        ("VIX", "VIX (level)", "#555555"),
    ]
    plot_macro_panel(macro_levels, raw_cols,
                      "Macro Variables, Raw Levels, Weekly (Appendix III)",
                      os.path.join(FIGURES_DIR, "fig_macro_raw_appendix.png"))

    transformed = pd.concat([macro_changes[config.MACRO_CHANGE_COLS], macro_levels[config.MACRO_LEVEL_COLS]], axis=1)
    trans_cols = [
        ("DXY_z52W", "DXY, 52W z-score", "#8B4513"),
        ("inventory_lag2_d4W", "Inventory, 4W change (lag 2W)", "#4169E1"),
        ("US3M_d1W", "US 3M yield, 1W change", "#2E8B57"),
        ("GPR_d26W", "GPR, 26W change", "#DC143C"),
        ("VIX", "VIX (level, unchanged)", "#555555"),
    ]
    plot_macro_panel(transformed, trans_cols,
                      "Macro Variables, Transform Used in the Model (Section 3.1)",
                      os.path.join(FIGURES_DIR, "fig_macro_transformed_ch4.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
