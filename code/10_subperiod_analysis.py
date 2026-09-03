"""
10_subperiod_analysis.py - Section 4.4: Sub-period forecast accuracy.

Splits the 2023–2026 test window at the Trump tariff shock (11 Apr 2025)
and computes RMSE, MAE, and hit rate for each model × horizon × sub-period.

Split rationale: 11 Apr 2025 is the date copper fell ~8% in a single week
following the announcement of broad US tariffs. This constitutes a
structural break that was not in the 2006–2022 training distribution.

Outputs:
  data/processed/subperiod_metrics.parquet
  report/figures/fig_subperiod_rmse.png

Run: .venv/bin/python code/10_subperiod_analysis.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from config import PROCESSED_DATA_DIR, FIGURES_DIR

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

TARIFF_DATE   = pd.Timestamp("2025-04-11")
MODEL_LABELS  = {0: "Random Walk", 1: "AR-Direct", 2: "LP + Macro", 3: "LP + Macro + EW"}
MODEL_SHORT   = {0: "RW", 1: "AR", 2: "LP+Mac", 3: "LP+EW"}
HORIZON_LABELS = {1: "1W", 2: "2W", 4: "1M", 9: "2M", 13: "3M", 26: "6M"}
PRIMARY_MATURITIES = [3]          # 3M maturity for main tables; 3 is the standard
PRIMARY_HORIZONS   = [4, 13, 26]  # 1M, 3M, 6M


def compute_metrics(df):
    """RMSE, MAE, Hit Rate for a slice of forecast_errors."""
    rows = []
    for (model, h, mat), grp in df.groupby(["model", "horizon", "maturity"]):
        grp = grp.dropna(subset=["error", "forecast", "actual"])
        if len(grp) < 3:
            continue
        rmse = float(np.sqrt(np.mean(grp["error"] ** 2)))
        mae  = float(np.mean(np.abs(grp["error"])))
        # Hit rate: did forecast move in the right direction vs current price?
        grp2 = grp.dropna(subset=["current_price"])
        if len(grp2) > 0 and model != 0:
            correct = (
                np.sign(grp2["forecast"] - grp2["current_price"]) ==
                np.sign(grp2["actual"]   - grp2["current_price"])
            )
            hit_rate = float(correct.mean())
        else:
            hit_rate = np.nan
        rows.append({
            "model": model, "horizon": h, "maturity": mat,
            "rmse": round(rmse, 1), "mae": round(mae, 1),
            "hit_rate": round(hit_rate, 3) if not np.isnan(hit_rate) else np.nan,
            "n": len(grp),
        })
    return pd.DataFrame(rows)


def print_table(metrics_df, period_label, horizons, maturity=3):
    """Print a formatted RMSE/hit-rate table for one sub-period."""
    sub = metrics_df[
        (metrics_df["maturity"] == maturity) &
        (metrics_df["horizon"].isin(horizons))
    ].copy()

    print(f"\n{'─'*70}")
    print(f"  {period_label}  |  {maturity}M maturity futures")
    print(f"{'─'*70}")
    header = f"{'Model':18s}"
    for h in horizons:
        lbl = HORIZON_LABELS.get(h, f"{h}W")
        header += f"  {lbl+' RMSE':>10s}  {lbl+' Hit%':>9s}  {'N':>4s}"
    print(header)
    print("─" * 70)

    for model_id in [0, 1, 2, 3]:
        line = f"{MODEL_LABELS[model_id]:18s}"
        for h in horizons:
            row = sub[(sub["model"] == model_id) & (sub["horizon"] == h)]
            if row.empty:
                line += f"  {'-':>10s}  {'-':>9s}  {'-':>4s}"
            else:
                rmse = row["rmse"].values[0]
                hr   = row["hit_rate"].values[0]
                n    = row["n"].values[0]
                hr_str = f"{hr*100:.1f}%" if not np.isnan(hr) else "  n/a"
                line += f"  {rmse:>10.1f}  {hr_str:>9s}  {n:>4.0f}"
        print(line)


def plot_subperiod_rmse(full, stable, shock, save_path):
    """
    Bar chart: RMSE by model at 3M horizon for three periods side by side.
    3M horizon, 3M maturity futures - the primary trade finance horizon.
    """
    h, mat = 13, 3
    periods = [
        ("Full OOS\n2023–2026", full),
        ("Stable period\n2023–Mar 2025", stable),
        ("Tariff shock\nApr 2025–2026", shock),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#aaaaaa", "#e07b39", "#2a6099", "#1a4a8a"]
    n_models = 4
    n_periods = 3
    bar_w = 0.18
    x_base = np.arange(n_periods)

    for i, model_id in enumerate([0, 1, 2, 3]):
        rmse_vals = []
        for _, df_p in periods:
            row = df_p[(df_p["model"] == model_id) &
                       (df_p["horizon"] == h) &
                       (df_p["maturity"] == mat)]
            rmse_vals.append(row["rmse"].values[0] if not row.empty else 0)
        offset = (i - n_models / 2 + 0.5) * bar_w
        bars = ax.bar(x_base + offset, rmse_vals, bar_w,
                      label=MODEL_LABELS[model_id],
                      color=colors[i], alpha=0.88, edgecolor="white", lw=0.5)
        for bar, val in zip(bars, rmse_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 15,
                        f"{val:.0f}", ha="center", va="bottom",
                        fontsize=6.5, color="#333333")

    ax.set_xticks(x_base)
    ax.set_xticklabels([p[0] for p in periods], fontsize=9)
    ax.set_ylabel("RMSE (USD/tonne)", fontsize=9)
    ax.set_title(
        "Forecast RMSE by Sub-period - 3M Horizon, 3M Maturity Futures",
        fontsize=9.5, fontweight="bold"
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="y", lw=0.3, color="#dddddd")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_linewidth(0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {save_path}")
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("Section 4.4 - Sub-period Forecast Accuracy")
    print(f"Tariff shock split date: {TARIFF_DATE.date()}")
    print("=" * 60)

    fe = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "forecast_errors.parquet"))
    fe["date"] = pd.to_datetime(fe["date"])

    stable_mask = fe["date"] <  TARIFF_DATE
    shock_mask  = fe["date"] >= TARIFF_DATE

    metrics_full   = compute_metrics(fe)
    metrics_stable = compute_metrics(fe[stable_mask])
    metrics_shock  = compute_metrics(fe[shock_mask])

    # Attach period label and save
    metrics_full["period"]   = "full"
    metrics_stable["period"] = "stable"
    metrics_shock["period"]  = "shock"
    combined = pd.concat([metrics_full, metrics_stable, metrics_shock])
    combined.to_parquet(os.path.join(PROCESSED_DATA_DIR, "subperiod_metrics.parquet"))
    print("Saved: data/processed/subperiod_metrics.parquet")

    # Print n obs per sub-period
    n_full   = fe["date"].nunique()
    n_stable = fe[stable_mask]["date"].nunique()
    n_shock  = fe[shock_mask]["date"].nunique()
    print(f"\nTest observations: Full={n_full}  Stable={n_stable}  Shock={n_shock}")

    # Print tables for primary horizons, 3M maturity
    print_table(metrics_full,   "FULL OOS (Jan 2023 – May 2026)",    PRIMARY_HORIZONS)
    print_table(metrics_stable, "STABLE (Jan 2023 – Mar 2025)",      PRIMARY_HORIZONS)
    print_table(metrics_shock,  "TARIFF SHOCK (Apr 2025 – May 2026)", PRIMARY_HORIZONS)

    # RMSE improvement vs RW - shock sub-period
    print(f"\n{'─'*60}")
    print("  RMSE ratio vs Random Walk - 3M horizon, 3M maturity")
    print(f"{'─'*60}")
    for period_lbl, df_p in [("Stable", metrics_stable), ("Shock", metrics_shock)]:
        rw_row = df_p[(df_p["model"]==0) & (df_p["horizon"]==13) & (df_p["maturity"]==3)]
        if rw_row.empty:
            continue
        rw_rmse = rw_row["rmse"].values[0]
        print(f"  {period_lbl} period - RW RMSE = {rw_rmse:.1f} $/t")
        for mid in [1, 2, 3]:
            row = df_p[(df_p["model"]==mid) & (df_p["horizon"]==13) & (df_p["maturity"]==3)]
            if row.empty:
                continue
            ratio = row["rmse"].values[0] / rw_rmse
            print(f"    {MODEL_LABELS[mid]:20s}  ratio = {ratio:.3f}  "
                  f"({'beats RW' if ratio < 1 else f'+{(ratio-1)*100:.1f}% vs RW'})")

    # Bar chart figure
    plot_subperiod_rmse(
        metrics_full, metrics_stable, metrics_shock,
        save_path=os.path.join(FIGURES_DIR, "fig_subperiod_rmse.png"),
    )

    print("\nSection 4.4 complete.")
    print("Next: write Chapter 5 and Chapter 6 in main.qmd")


if __name__ == "__main__":
    main()
