"""
04c_multi_horizon_granger.py - Multi-Horizon Local Projection Granger Tests.

Extends the 1-step Granger screening (04b) to all forecast horizons of interest.
Tests ALL 77 macro variable × transformation combinations against all 3 NS factors
at 9 horizons (1W through 6M) using Jordà (2005) local projections.

WHY LOCAL PROJECTIONS:
  Standard 1-step Granger tests whether macro_{t-1} helps predict β_t (h=1).
  A variable significant at h=1 may have zero signal at h=13W (3M) if its
  effect decays through iterated dynamics. A slow-moving variable may show no
  1-step causality but predict β at 3M through a structural channel.
  Local projections test the DIRECT relationship at each horizon h separately:
    does macro observed today predict β in h weeks?
  This is the relevant question for conditional forecasting where macro is
  fixed at the current observed state.

METHOD:
  For each macro variable x, NS factor β_i, and horizon h:
    Restricted:   β_i^{t+h} = α + Σ_k γ_k·β_i^{t-k+1} + ε_t
    Unrestricted: β_i^{t+h} = α + Σ_k γ_k·β_i^{t-k+1} + Σ_k δ_k·x^{t-k+1} + ε_t
    H₀: δ₁ = δ₂ = 0 - macro adds no predictive power at horizon h
  Wald F-test with HAC Newey-West SE, bandwidth = max(h, p).

DATA WINDOW: Fixed training sample 2006–2022. Variable selection is a model
  specification question - use the full training sample for maximum power.
  Expanding window is reserved for the rolling backtest (script 07).

Horizons tested: {1, 2, 3, 4, 9, 13, 17, 22, 26} weeks
  = 1W, 2W, 3W, 4W(1M), 2M, 3M, 4M, 5M, 6M

Inputs  : data/processed/ns_factors.parquet
          data/processed/macro_all_transformed.parquet  (all 77 var×transform combos)
Outputs : data/processed/multi_horizon_granger.parquet  (all 2,079 tests)
          data/processed/multi_horizon_granger.xlsx      (thesis evidence tables)
          report/figures/fig_multi_horizon_granger.png   (heatmap - best transform per variable)

Run: python code/04c_multi_horizon_granger.py  (≈ 2–4 minutes)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tools import add_constant

from config import TRAIN_END, PROCESSED_DATA_DIR, FIGURES_DIR, FIGURE_DPI

# ── Configuration ─────────────────────────────────────────────────────────────

HORIZONS_WEEKS = [1, 2, 3, 4, 9, 13, 17, 22, 26]

HORIZON_LABELS = {
    1: "1W", 2: "2W", 3: "3W", 4: "4W",
    9: "2M", 13: "3M", 17: "4M", 22: "5M", 26: "6M",
}

# Primary horizons used to determine inclusion verdict
PRIMARY_HORIZONS = [4, 13]   # 4W = 1M, 13W = 3M

LAG_ORDER = 2   # consistent with VAR lag order in scripts 03–07

NS_FACTORS = ["beta0", "beta1", "beta2"]
NS_LABELS  = {
    "beta0": r"$\beta_0$  Level",
    "beta1": r"$\beta_1$  Slope",
    "beta2": r"$\beta_2$  Curvature",
}

# Currently selected 5-variable spec - used to flag confirmed vs new findings
# in this screening pass's own figures/tables (fig_multi_horizon_granger.png
# etc.). "Inventory_d4W" here (capital I, unlagged) is this screening
# universe's closest match to the actual production variable
# inventory_lag2_d4W (config.py MACRO_VAR_SPEC) - this pass, inherited from
# 04b_macro_screening.py's candidate set, does not carry a lag-2 variant of
# inventory, so the highlighting does not distinguish the lag correction
# documented in main.qmd Section 3.1/3.2. Cosmetic only: this affects which
# rows get a highlight box in the appendix figure, not any production number.
SELECTED_VARS = {"DXY_z52W", "Inventory_d4W", "US3M_d1W", "GPR_d26W", "VIX_level"}

ALPHA_01 = 0.01
ALPHA_05 = 0.05
ALPHA_10 = 0.10


def sig_star(p: float) -> str:
    if np.isnan(p):     return "-"
    if p < ALPHA_01:    return "***"
    if p < ALPHA_05:    return "**"
    if p < ALPHA_10:    return "*"
    return "n.s."


def parse_col(col: str) -> tuple[str, str]:
    """Split 'DXY_z52W' → ('DXY', 'z52W'), 'VIX_level' → ('VIX', 'level')."""
    for suffix in ["_z52W", "_d26W", "_d13W", "_d4W", "_d1W", "_level"]:
        if col.endswith(suffix):
            return col[: -len(suffix)], suffix[1:]
    return col, "unknown"


# ── Local Projection ──────────────────────────────────────────────────────────

def local_projection_pvalue(y: np.ndarray, x: np.ndarray,
                             h: int, p: int) -> tuple[float, int]:
    """
    Wald test of H₀: macro lags have no predictive power for y at horizon h.

    Unrestricted model: y_{t+h} = intercept + own_lags_p + macro_lags_p + ε
    Restricted model:  y_{t+h} = intercept + own_lags_p + ε
    HAC Newey-West SE with bandwidth = max(h, p).

    Returns (p_value, n_valid_obs).
    """
    T = len(y)
    max_start = p - 1
    max_end   = T - h - 1
    if max_end <= max_start + p + 2:
        return np.nan, 0

    idx  = np.arange(max_start, max_end + 1)
    Y    = y[idx + h]
    own  = np.column_stack([y[idx - k] for k in range(p)])
    xlag = np.column_stack([x[idx - k] for k in range(p)])

    valid = (
        ~np.isnan(Y) &
        ~np.any(np.isnan(own),  axis=1) &
        ~np.any(np.isnan(xlag), axis=1)
    )
    Y, own, xlag = Y[valid], own[valid], xlag[valid]
    n = valid.sum()

    if n < 2 * (2 * p + 1):
        return np.nan, n

    X_full = add_constant(np.hstack([own, xlag]), has_constant="add")
    bw     = max(h, p)

    try:
        from statsmodels.regression.linear_model import OLS
        res  = OLS(Y, X_full).fit(cov_type="HAC",
                                   cov_kwds={"maxlags": bw, "use_correction": True})
        # Macro lags occupy columns [1+p : 1+2p] (after intercept + own lags)
        macro_idx = list(range(1 + p, 1 + 2 * p))
        wald = res.wald_test(np.eye(X_full.shape[1])[macro_idx], use_f=True)
        return float(wald.pvalue), n
    except Exception:
        return np.nan, n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("04c - Multi-Horizon Local Projection Granger Tests")
    print("       All 77 macro transforms × 3 NS factors × 9 horizons")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    factors = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet")
    )[NS_FACTORS].loc[:TRAIN_END]

    macro_all = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "macro_all_transformed.parquet")
    ).loc[:TRAIN_END]

    # Align on common weekly index
    common   = factors.index.intersection(macro_all.index)
    factors  = factors.loc[common]
    macro_all = macro_all.loc[common]

    macro_cols = list(macro_all.columns)
    n_train    = len(common)
    n_tests    = len(macro_cols) * len(NS_FACTORS) * len(HORIZONS_WEEKS)

    print(f"Training: {common[0].date()} → {common[-1].date()} ({n_train} weeks)")
    print(f"Macro columns : {len(macro_cols)} (all variables × transforms)")
    print(f"NS factors    : {len(NS_FACTORS)}")
    print(f"Horizons      : {HORIZONS_WEEKS} weeks")
    print(f"Total tests   : {n_tests}\n")

    # ── Run all tests ─────────────────────────────────────────────────────────
    rows = []
    completed = 0
    for factor in NS_FACTORS:
        y_arr = factors[factor].values
        for col in macro_cols:
            x_arr      = macro_all[col].values
            base, tfm  = parse_col(col)
            for h in HORIZONS_WEEKS:
                p_val, n_obs = local_projection_pvalue(y_arr, x_arr, h, LAG_ORDER)
                rows.append({
                    "factor":       factor,
                    "col":          col,
                    "base_var":     base,
                    "transform":    tfm,
                    "horizon_w":    h,
                    "horizon_lbl":  HORIZON_LABELS[h],
                    "p_value":      round(p_val, 4) if not np.isnan(p_val) else np.nan,
                    "sig":          sig_star(p_val),
                    "sig_05":       (not np.isnan(p_val)) and p_val < ALPHA_05,
                    "sig_10":       (not np.isnan(p_val)) and p_val < ALPHA_10,
                    "n_obs":        n_obs,
                    "selected":     col in SELECTED_VARS,
                })
                completed += 1
        print(f"  {factor}: done ({completed}/{n_tests})")

    results = pd.DataFrame(rows)

    # ── Save raw results ──────────────────────────────────────────────────────
    parq_path = os.path.join(PROCESSED_DATA_DIR, "multi_horizon_granger.parquet")
    results.to_parquet(parq_path, index=False)
    print(f"\nSaved raw results: {parq_path}")

    # ── Console summary - best transform per base variable × factor ───────────
    h_lbls = [HORIZON_LABELS[h] for h in HORIZONS_WEEKS]
    col_w  = 6

    print(f"\n{'─'*80}")
    print("RESULTS: best transformation of each variable at each horizon (min p-value)")
    print(f"Significance: *** p<0.01  ** p<0.05  * p<0.10  n.s. not significant")
    print(f"Primary horizons (4W, 13W) starred → used for inclusion verdict")
    print(f"{'─'*80}")

    base_vars = results["base_var"].unique()

    for factor in NS_FACTORS:
        print(f"\n── {NS_LABELS[factor]} ──")
        header = f"{'Variable':<18}" + "".join(f"{l:>{col_w}}" for l in h_lbls) + \
                 f"  {'Verdict'}"
        print(header)
        print("─" * len(header))

        sub_f = results[results["factor"] == factor]

        for base in base_vars:
            sub_b = sub_f[sub_f["base_var"] == base]
            row_str = f"{base:<18}"

            for h in HORIZONS_WEEKS:
                sub_h = sub_b[sub_b["horizon_w"] == h]
                if sub_h.empty:
                    row_str += f"{'-':>{col_w}}"
                    continue
                # Best transform = min p-value at this horizon
                best_p = sub_h["p_value"].min()
                star   = sig_star(best_p)
                # Annotate primary horizons with parentheses
                mark = f"({star})" if h in PRIMARY_HORIZONS else star
                row_str += f"{mark:>{col_w}}"

            # Verdict: significant at ≥1 primary horizon (best transform)?
            primary_p = []
            for h in PRIMARY_HORIZONS:
                sub_h = sub_b[sub_b["horizon_w"] == h]
                if not sub_h.empty:
                    primary_p.append(sub_h["p_value"].min())

            short_p = []
            for h in [h for h in HORIZONS_WEEKS if h <= 3]:
                sub_h = sub_b[sub_b["horizon_w"] == h]
                if not sub_h.empty:
                    short_p.append(sub_h["p_value"].min())

            currently_in = sub_b["selected"].any()
            tag = " [SEL]" if currently_in else ""

            if any(p < ALPHA_05 for p in primary_p if not np.isnan(p)):
                verdict = f"CONFIRMED at primary{tag}"
            elif any(p < ALPHA_05 for p in short_p if not np.isnan(p)):
                verdict = f"Short-horizon only{tag}"
            else:
                verdict = f"Not significant{tag}"

            print(f"{row_str}  {verdict}")

    print(f"\n  [SEL] = currently in selected 4-variable VAR spec")
    print(f"  Primary horizons shown in (parentheses)")

    # ── Excel evidence table ──────────────────────────────────────────────────
    xlsx_path = os.path.join(PROCESSED_DATA_DIR, "multi_horizon_granger.xlsx")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:

        # Sheet per NS factor: rows = col, columns = horizon, values = p-value
        for factor in NS_FACTORS:
            sub_f = results[results["factor"] == factor]
            piv = sub_f.pivot_table(
                index="col",
                columns="horizon_lbl",
                values="p_value",
            )[h_lbls]
            piv.to_excel(writer, sheet_name=f"pval_{factor}")

        # Summary sheet: best-transform p-value per base variable × factor × horizon
        summary_rows = []
        for factor in NS_FACTORS:
            sub_f = results[results["factor"] == factor]
            for base in base_vars:
                sub_b = sub_f[sub_f["base_var"] == base]
                row = {"factor": factor, "base_var": base}
                for h in HORIZONS_WEEKS:
                    sub_h = sub_b[sub_b["horizon_w"] == h]
                    if sub_h.empty:
                        row[HORIZON_LABELS[h] + "_p"] = np.nan
                        row[HORIZON_LABELS[h] + "_sig"] = "-"
                        row[HORIZON_LABELS[h] + "_best_tfm"] = "-"
                    else:
                        best_row = sub_h.loc[sub_h["p_value"].idxmin()]
                        row[HORIZON_LABELS[h] + "_p"]   = best_row["p_value"]
                        row[HORIZON_LABELS[h] + "_sig"] = sig_star(best_row["p_value"])
                        row[HORIZON_LABELS[h] + "_best_tfm"] = best_row["transform"]

                # Verdict
                primary_p = [
                    row.get(HORIZON_LABELS[h] + "_p", np.nan)
                    for h in PRIMARY_HORIZONS
                ]
                if any(not np.isnan(p) and p < ALPHA_05 for p in primary_p):
                    row["verdict"] = "Confirmed"
                else:
                    row["verdict"] = "Not confirmed at primary horizons"
                row["in_selected_spec"] = base in {parse_col(s)[0] for s in SELECTED_VARS}
                summary_rows.append(row)

        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name="summary_best_transform", index=False
        )

        # Variable selection verdict sheet
        verdict_rows = []
        for factor in NS_FACTORS:
            sub_f = results[results["factor"] == factor]
            for base in base_vars:
                sub_b   = sub_f[sub_f["base_var"] == base]
                sig_at  = []
                for h in HORIZONS_WEEKS:
                    sub_h = sub_b[sub_b["horizon_w"] == h]
                    if not sub_h.empty and sub_h["p_value"].min() < ALPHA_05:
                        sig_at.append(HORIZON_LABELS[h])

                primary_sig = [
                    hl for hl in sig_at
                    if hl in {HORIZON_LABELS[h] for h in PRIMARY_HORIZONS}
                ]
                verdict_rows.append({
                    "factor":          NS_LABELS[factor],
                    "base_variable":   base,
                    "sig_horizons_p05": ", ".join(sig_at) if sig_at else "None",
                    "primary_horizon_sig": ", ".join(primary_sig) if primary_sig else "None",
                    "verdict":         "Include" if primary_sig else "Exclude",
                    "in_current_spec": base in {parse_col(s)[0] for s in SELECTED_VARS},
                })

        pd.DataFrame(verdict_rows).to_excel(
            writer, sheet_name="variable_verdict", index=False
        )

    print(f"\nSaved evidence table: {xlsx_path}")

    # ── Heatmap figure ────────────────────────────────────────────────────────
    _plot_heatmap(results, base_vars, h_lbls, FIGURES_DIR)

    print("\n04c complete.")
    print("Review the heatmap and variable_verdict sheet before updating macro spec.")
    print("Next: implement conditional forecasting in 07_forecast_evaluation.py")


def _plot_heatmap(results: pd.DataFrame, base_vars,
                  h_lbls: list, out_dir: str) -> None:
    """
    Three-panel heatmap - one panel per NS factor.
    Rows = base variables (best transform). Columns = horizons.
    Colour = -log10(best p-value across transforms).
    Red box = primary horizons. Stars annotated in cells.
    """
    primary_lbls = {HORIZON_LABELS[h] for h in PRIMARY_HORIZONS}
    n_vars  = len(base_vars)
    base_list = list(base_vars)

    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, n_vars * 0.45 + 2)))
    fig.suptitle(
        "Figure X.Y - Multi-Horizon Local Projection Granger Test\n"
        "Best transformation of each variable | Training 2006–2022 | HAC NW SE\n"
        "Colour = −log₁₀(min p-value across transforms) | Red box = primary horizon (1M, 3M)",
        fontsize=9, y=1.02,
    )

    vmin, vmax = 0.0, 3.0
    cmap = plt.cm.YlOrRd

    for ax, factor in zip(axes, ["beta0", "beta1", "beta2"]):
        sub_f = results[results["factor"] == factor]

        # Build matrix: rows = base vars, cols = horizons, value = best -log10(p)
        mat  = np.zeros((n_vars, len(h_lbls)))
        smat = [["" for _ in h_lbls] for _ in range(n_vars)]

        for i, base in enumerate(base_list):
            sub_b = sub_f[sub_f["base_var"] == base]
            for j, hl in enumerate(h_lbls):
                h_val = [k for k, v in HORIZON_LABELS.items() if v == hl][0]
                sub_h = sub_b[sub_b["horizon_w"] == h_val]
                if sub_h.empty or sub_h["p_value"].isna().all():
                    mat[i, j]  = 0
                    smat[i][j] = "-"
                else:
                    best_p     = sub_h["p_value"].min()
                    mat[i, j]  = min(-np.log10(best_p), vmax) if best_p > 0 else vmax
                    smat[i][j] = sig_star(best_p)

        im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax,
                       cmap=cmap, interpolation="none")

        ax.set_xticks(range(len(h_lbls)))
        ax.set_xticklabels(h_lbls, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_vars))
        ax.set_yticklabels(base_list, fontsize=7)
        ax.set_title(NS_LABELS[factor], fontsize=10, pad=6)

        # Annotate cells + primary horizon boxes
        for i in range(n_vars):
            for j, hl in enumerate(h_lbls):
                text_col = "white" if mat[i, j] > 1.8 else "black"
                ax.text(j, i, smat[i][j], ha="center", va="center",
                        fontsize=6.5, color=text_col, fontweight="bold")
                if hl in primary_lbls:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="royalblue",
                        linewidth=2.0, zorder=3,
                    ))

        # Mark currently selected variables
        for i, base in enumerate(base_list):
            in_sel = any(
                parse_col(s)[0] == base for s in
                {"DXY_z52W", "Inventory_d4W", "US3M_d1W", "GPR_d26W", "VIX_level"}
            )
            if in_sel:
                ax.add_patch(plt.Rectangle(
                    (-0.5, i - 0.5), len(h_lbls), 1,
                    fill=False, edgecolor="green",
                    linewidth=1.2, linestyle="--", zorder=2,
                ))

        if factor == "beta2":
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="−log₁₀(best p-value)")

    legend_text = (
        "Blue box = primary forecast horizon (1M, 3M)\n"
        "Green dashed row = variable in current 4-var VAR spec"
    )
    fig.text(0.5, -0.04, legend_text, ha="center", fontsize=8, color="dimgray")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_multi_horizon_granger.png")
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_multi_horizon_granger.png")


if __name__ == "__main__":
    main()
