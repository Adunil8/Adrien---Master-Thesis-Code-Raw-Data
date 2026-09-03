"""
10_granger_summary_table.py - Appendix table: Multi-Horizon Granger Causality.

Generates publication-quality PNG tables matching the LaTeX/Overleaf booktabs
aesthetic used throughout the thesis:
  - Times New Roman body font (matches HEC Lausanne thesis requirement)
  - Pure black and white - no background colours in data cells
  - Horizontal rules only: toprule (1.5 pt), midrule (0.8 pt), cmidrule
    between variable groups, bottomrule (1.5 pt)
  - No vertical rules
  - Stars + parenthesised p-values (significant cells only)
  - Landscape A4 (11.69 × 8.27 in) per panel at 300 DPI

Inputs : data/processed/multi_horizon_granger.parquet
Outputs: report/figures/fig_granger_beta0.png
         report/figures/fig_granger_beta1.png
         report/figures/fig_granger_beta2.png
         report/figures/fig_granger_combined.png

Run: .venv/bin/python code/10_granger_summary_table.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from config import PROCESSED_DATA_DIR, FIGURES_DIR

# ── Font: Times New Roman (HEC thesis standard) ───────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",          # STIX for maths - close to CM
    "text.usetex":      False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "xtick.bottom": False,
    "ytick.left":   False,
})

# ── Variable ordering ─────────────────────────────────────────────────────────
# (base_var key, display label in table, status)
VAR_ORDER = [
    # Selected in final 5-variable macro specification (Section 3.1)
    ("DXY",         r"DXY - U.S. Dollar Index",          "sel"),
    ("VIX",         r"VIX - CBOE Volatility Index",      "sel"),
    ("US3M",        r"US3M - 3-Month Treasury Bill",     "sel"),
    ("Inventory",   r"Inventory - LME On-Warrant Stock", "sel"),
    ("GPR",         r"GPR - Geopolitical Risk Index",    "sel"),
    # Candidates excluded for documented reasons (each shows a genuine
    # significant relationship at a primary horizon, but is dropped in
    # favour of a selected variable or another candidate - see notes below
    # and the Section 4.2 discussion for the reason attached to each)
    ("AUDUSD",      r"AUD/USD exchange rate",             "cand"),
    ("DFII10",      r"DFII10 - 10-Year Real Rate",       "cand"),
    ("BRENT",       r"Brent crude oil price",             "cand"),
    ("COT_net_abs", r"COT net positioning (absolute)",   "cand"),
    ("US2Y",        r"US2Y - 2-Year Treasury",           "cand"),
    ("CNYUSD",      r"CNY/USD exchange rate",             "cand"),
    ("COT_net_pct", r"COT net positioning (\%)",         "cand"),
    ("US10Y",       r"US10Y - 10-Year Treasury",         "cand"),
    # Excluded - genuinely not significant at any primary horizon, any factor
    ("T10Y2Y",      r"T10Y2Y - Yield spread (10Y--2Y)",  "excl"),
]

HORIZONS     = [1, 2, 3, 4, 9, 13, 17, 22, 26]
HORIZON_LBLS = ["1W", "2W", "3W", "4W", "2M", "3M", "4M", "5M", "6M"]
PRIMARY_H    = {4, 13}   # 1-month and 3-month

FACTOR_META = {
    "beta0": dict(
        math_label=r"$\hat{\beta}_0$ (Level)",
        desc=(r"Long-run copper price level. "
              r"Determines the overall height of the futures curve."),
        fname="fig_granger_beta0.png",
    ),
    "beta1": dict(
        math_label=r"$\hat{\beta}_1$ (Slope)",
        desc=(r"Term structure slope. Positive = backwardation; negative = contango. "
              r"Primary driver of margin call direction and roll yield."),
        fname="fig_granger_beta1.png",
    ),
    "beta2": dict(
        math_label=r"$\hat{\beta}_2$ (Curvature)",
        desc=(r"Mid-curve hump component. "
              r"Captures deviations from a linear slope across maturities."),
        fname="fig_granger_beta2.png",
    ),
}


# ── Signal pattern classification ─────────────────────────────────────────────

def classify_pattern(sigs: dict) -> str:
    short  = [1, 2, 3, 4]
    medium = [9, 13]
    long   = [17, 22, 26]
    ns = lambda hs: sum(1 for h in hs if sigs.get(h, "n.s.") != "n.s.")
    ns_, nm, nl = ns(short), ns(medium), ns(long)
    total = ns_ + nm + nl
    if total == 0:                       return "None"
    if total <= 2:                       return "Marginal"
    if ns_ >= 2 and nl >= 2:             return "Persistent"
    if ns_ >= 2 and nl == 0:             return "Fading"
    if ns_ == 0 and (nm + nl) >= 2:      return "Building"
    if nm >= 2 and ns_ <= 1 and nl <= 1: return "Peaked"
    if ns_ >= 2 and nm >= 1 and nl <= 1: return "Fading"
    return "Marginal"


# ── Core drawing routine ──────────────────────────────────────────────────────

def draw_booktabs_table(fig: plt.Figure,
                        best: pd.DataFrame,
                        left: float, bottom: float,
                        width: float, height: float) -> None:
    """
    Draw a complete booktabs-style significance table into a sub-region of fig.

    Coordinates are in figure-fraction units.
    Pure black-and-white: no cell background colours, no vertical rules.
    Horizontal rules only: toprule, column-header rule, group separators,
    bottomrule.  Column headers bold; data in normal weight.
    """
    # ── Geometry ─────────────────────────────────────────────────────────────
    N        = len(VAR_ORDER)
    N_HCOLS  = len(HORIZONS)
    # Column widths as fractions of `width`
    W_VAR  = 0.250   # variable name
    W_H    = (1.0 - W_VAR - 0.118 - 0.072) / N_HCOLS   # each horizon col
    W_PAT  = 0.118   # signal pattern
    W_STA  = 0.072   # status

    # Row heights as fractions of `height`
    HDR_FRAC   = 0.085   # two-line column header
    DATA_FRAC  = (1.0 - HDR_FRAC) / N

    # Convert to figure coordinates
    def fx(rel_x):   return left + rel_x * width
    def fy(rel_y):   return bottom + rel_y * height

    # Column left edges (in relative-x, 0–1)
    col_x = {"var": 0.0}
    x = W_VAR
    for i in range(N_HCOLS):
        col_x[f"h{i}"] = x;  x += W_H
    col_x["pat"] = x;  x += W_PAT
    col_x["sta"] = x

    # Row top edges: row 0 = top of data area
    hdr_top  = 1.0
    hdr_bot  = 1.0 - HDR_FRAC
    data_tops = [hdr_bot - i * DATA_FRAC for i in range(N)]

    def row_mid(i):
        return hdr_bot - (i + 0.5) * DATA_FRAC

    # Helper: draw a horizontal rule across the full table width
    def hline(rel_y, lw=0.6, ls="-", color="black"):
        fig.add_artist(matplotlib.lines.Line2D(
            [fx(0), fx(1)], [fy(rel_y), fy(rel_y)],
            linewidth=lw, linestyle=ls, color=color,
            transform=fig.transFigure, clip_on=False, zorder=10))

    # Helper: text in figure coordinates
    def ftext(rx, ry, s, **kw):
        kw.setdefault("transform", fig.transFigure)
        kw.setdefault("clip_on", False)
        fig.text(fx(rx), fy(ry), s, **kw)

    # Build lookup {(base_var, horizon_w): (sig, p_value)}
    lkp = {}
    for _, row in best.iterrows():
        lkp[(row["base_var"], int(row["horizon_w"]))] = (row["sig"], row["p_value"])

    # ── Rules ─────────────────────────────────────────────────────────────────
    hline(1.0,   lw=1.4)   # toprule
    hline(hdr_bot, lw=0.8) # midrule (after column headers)

    # Group separator lines (between sel/cand and cand/excl blocks)
    statuses = [v[2] for v in VAR_ORDER]
    for i in range(1, N):
        if statuses[i] != statuses[i-1]:
            hline(data_tops[i], lw=0.4, ls=(0, (4, 2)), color="#666666")

    hline(0.0,   lw=1.4)   # bottomrule

    # ── Column headers ────────────────────────────────────────────────────────
    # Row 1: group spans
    group_defs = [
        ("Short-term",   range(0, 3)),
        ("1-Month",      range(3, 4)),   # primary
        ("Medium-term",  range(4, 5)),
        ("3-Month",      range(5, 6)),   # primary
        ("Long-term",    range(6, 9)),
    ]
    hdr_mid = (hdr_top + hdr_bot) / 2
    hdr_q1  = hdr_bot + (hdr_top - hdr_bot) * 0.72   # upper sub-row mid
    hdr_q2  = hdr_bot + (hdr_top - hdr_bot) * 0.25   # lower sub-row mid

    # Variable column header
    ftext(col_x["var"] + W_VAR / 2, hdr_mid,
          "Macro variable",
          ha="center", va="center", fontsize=8.5, fontweight="bold")

    for glab, idxs in group_defs:
        idxs = list(idxs)
        x0 = col_x[f"h{idxs[0]}"]
        x1 = col_x[f"h{idxs[-1]}"] + W_H
        mid = (x0 + x1) / 2
        is_prim = any(HORIZONS[i] in PRIMARY_H for i in idxs)
        fw = "bold" if is_prim else "normal"
        # Underline primary groups with a thin accent line
        if is_prim:
            fig.add_artist(matplotlib.lines.Line2D(
                [fx(x0 + 0.005), fx(x1 - 0.005)],
                [fy(hdr_bot + 0.006), fy(hdr_bot + 0.006)],
                linewidth=1.0, color="black",
                transform=fig.transFigure, clip_on=False))
        ftext(mid, hdr_q1, glab,
              ha="center", va="center", fontsize=7.5, fontweight=fw,
              style="italic" if not is_prim else "normal")

    # Pattern and Status headers
    ftext(col_x["pat"] + W_PAT / 2, hdr_mid,
          "Signal pattern", ha="center", va="center",
          fontsize=8.5, fontweight="bold")
    ftext(col_x["sta"] + W_STA / 2, hdr_mid,
          "Status", ha="center", va="center",
          fontsize=8.5, fontweight="bold")

    # Row 2: individual horizon labels
    for i, (h, lbl) in enumerate(zip(HORIZONS, HORIZON_LBLS)):
        is_prim = h in PRIMARY_H
        fw = "bold" if is_prim else "normal"
        ftext(col_x[f"h{i}"] + W_H / 2, hdr_q2, lbl,
              ha="center", va="center", fontsize=7.5, fontweight=fw)

    # ── Data rows ─────────────────────────────────────────────────────────────
    for row_i, (base_var, disp_label, status) in enumerate(VAR_ORDER):
        ymid = row_mid(row_i)

        # Variable name - bold if selected, italic if excluded
        fw_var = "bold"   if status == "sel"  else "normal"
        fs_var = "italic" if status == "excl" else "normal"
        col_var_text = "#1a1a1a" if status != "excl" else "#666666"
        ftext(col_x["var"] + 0.006, ymid, disp_label,
              ha="left", va="center", fontsize=7.8,
              fontweight=fw_var, style=fs_var, color=col_var_text)

        # Horizon cells
        sigs_row = {}
        for i, h in enumerate(HORIZONS):
            sig, pval = lkp.get((base_var, h), ("n.s.", 1.0))
            sigs_row[h] = sig
            xmid = col_x[f"h{i}"] + W_H / 2

            if sig == "n.s.":
                ftext(xmid, ymid, "-",       # em-dash, clearly visible
                      ha="center", va="center",
                      fontsize=7.5, color="#888888")
            else:
                fw_s = "bold" if sig == "***" else "normal"
                # Stars
                ftext(xmid, ymid + DATA_FRAC * 0.14, sig,
                      ha="center", va="center",
                      fontsize=7.0, fontweight=fw_s, color="#1a1a1a")
                # p-value in parens below - smaller
                pstr = f"({pval:.3f})" if pval >= 0.001 else r"($<$.001)"
                ftext(xmid, ymid - DATA_FRAC * 0.22, pstr,
                      ha="center", va="center",
                      fontsize=5.5, color="#444444")

        # Signal pattern
        pattern = classify_pattern(sigs_row)
        ftext(col_x["pat"] + W_PAT / 2, ymid, pattern,
              ha="center", va="center", fontsize=7.2,
              fontweight="bold" if pattern in ("Persistent", "Building") else "normal",
              color="#1a1a1a" if status != "excl" else "#888888")

        # Status
        sta_text = {"sel": "Selected", "cand": "Candidate", "excl": "Excluded"}.get(status, "")
        ftext(col_x["sta"] + W_STA / 2, ymid, sta_text,
              ha="center", va="center", fontsize=7.0,
              fontweight="bold" if status == "sel" else "normal",
              style="italic" if status == "excl" else "normal",
              color="#1a1a1a" if status != "excl" else "#888888")

        # Light row separator (not toprule-weight)
        hline(data_tops[row_i], lw=0.25, color="#cccccc")


# ── Figure builder (single factor) ───────────────────────────────────────────

def make_factor_figure(df: pd.DataFrame, factor: str) -> plt.Figure:
    meta = FACTOR_META[factor]
    best = df[df["factor"] == factor].copy()
    best = best.loc[
        best.groupby(["base_var", "horizon_w"])["p_value"].idxmin()
    ].reset_index(drop=True)

    # Landscape A4 proportions
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")

    # ── Table title and description ──────────────────────────────────────────
    fig.text(0.04, 0.965,
             f"Table - Granger Causality (Local Projections): {meta['math_label']}",
             ha="left", va="top", fontsize=11.0, fontweight="bold", color="#1a1a1a")
    fig.text(0.04, 0.940,
             meta["desc"],
             ha="left", va="top", fontsize=9.0, style="italic", color="#444444")

    # ── Table area: left=4%, right=4%, below title, above notes ─────────────
    draw_booktabs_table(
        fig,
        best,
        left=0.040, bottom=0.095,
        width=0.920, height=0.830,
    )

    # ── Table note ───────────────────────────────────────────────────────────
    note = (
        r"\textit{Notes:} "
        r"Local projection (Jord\`{a}, 2005). "
        r"For each variable the best-performing transformation (minimum p-value "
        r"per horizon, across level, $\Delta$1W, $\Delta$4W, $\Delta$13W, "
        r"$\Delta$26W, $z_{52W}$) is reported. "
        r"Stars indicate significance of a joint Wald test on macro lag "
        r"coefficients: *** $p < 0.01$, ** $p < 0.05$, * $p < 0.10$; "
        r"p-values in parentheses; -- = not significant. "
        r"HAC Newey-West standard errors, bandwidth $= \max(h,\,2)$. "
        r"Training sample: 2006--2022 (887 weekly observations). "
        r"Underlined group headers mark the primary trade finance horizons "
        r"(4W $\approx$ 1 month; 3M $\approx$ 3 months). "
        r"\textit{Selected:} included in the 5-variable macro specification. "
        r"\textit{Candidate:} significant at a primary horizon for at least one "
        r"factor, but dropped in favour of a selected variable or another "
        r"candidate (collinearity, a weaker or secondary-factor-only signal, "
        r"or CFTC sample-size limits -- see Section 4.2). "
        r"\textit{Excluded:} not significant at any primary horizon, any factor."
    )
    # Note: usetex=False so we write plain text (no backslash commands)
    note_plain = (
        "Notes: Local projection (Jorda, 2005). For each variable the "
        "best-performing transformation (minimum p-value per horizon, across "
        "level, D1W, D4W, D13W, D26W, z52W) is reported. Stars indicate "
        "significance of a joint Wald test on macro lag coefficients: "
        "*** p < 0.01, ** p < 0.05, * p < 0.10; p-values in parentheses; "
        "-- = not significant. HAC Newey-West SE, bandwidth = max(h, 2). "
        "Training sample: 2006-2022 (887 weekly obs.). Underlined group headers "
        "mark primary trade finance horizons (4W = 1 month; 3M = 3 months). "
        "Bold rows: selected in the 5-variable macro specification. Plain rows: "
        "a candidate with a genuine primary-horizon signal, dropped in favour of "
        "a selected or another candidate variable. Italic rows: not significant "
        "at any primary horizon, any factor."
    )
    fig.text(0.040, 0.008, note_plain,
             ha="left", va="bottom",
             fontsize=6.5, style="italic", color="#444444",
             wrap=True)

    return fig


# ── Combined 3-panel figure ───────────────────────────────────────────────────

def make_combined_figure(df: pd.DataFrame) -> plt.Figure:
    """
    Three panels stacked vertically on a single portrait-A4-wide figure.
    Each panel gets its own toprule/bottomrule so it looks like three
    separate tables, clearly labelled Panel A / B / C.
    """
    fig = plt.figure(figsize=(11.69, 24.0), facecolor="white")

    # Overall title
    fig.text(0.5, 0.992,
             "Multi-Horizon Local Projection Granger Causality Tests",
             ha="center", va="top", fontsize=13.0,
             fontweight="bold", color="#1a1a1a")
    fig.text(0.5, 0.984,
             ("LME copper Nelson-Siegel factors.  14 macro candidates.  "
              "Nine forecast horizons (1W to 6M).  "
              "Training sample: 2006--2022 (887 weekly observations)."),
             ha="center", va="top",
             fontsize=8.5, style="italic", color="#444444")

    panel_height = 0.300   # fraction of figure height per panel
    gaps         = [0.975, 0.665, 0.355]   # top of each panel's table area

    for (factor, meta), top in zip(FACTOR_META.items(), gaps):
        panel = "ABC"[list(FACTOR_META).index(factor)]
        best = df[df["factor"] == factor].copy()
        best = best.loc[
            best.groupby(["base_var", "horizon_w"])["p_value"].idxmin()
        ].reset_index(drop=True)

        # Panel label + description
        fig.text(0.040, top + 0.005,
                 f"Panel {panel}: {meta['math_label']}",
                 ha="left", va="bottom",
                 fontsize=10.0, fontweight="bold", color="#1a1a1a")
        fig.text(0.040, top + 0.0005,
                 meta["desc"],
                 ha="left", va="top",
                 fontsize=8.0, style="italic", color="#444444")

        draw_booktabs_table(
            fig, best,
            left=0.040, bottom=top - panel_height + 0.010,
            width=0.920, height=panel_height - 0.022,
        )

    # Shared note
    note = (
        "Notes: Local projection (Jorda, 2005). Best-performing transformation per "
        "variable x horizon (minimum p-value across level, D1W, D4W, D13W, D26W, z52W). "
        "*** p<0.01, ** p<0.05, * p<0.10; p-values in parentheses; -- = not significant. "
        "HAC Newey-West SE, bandwidth = max(h, 2). Underlined group headers: primary trade "
        "finance horizons (4W, 3M). Bold rows: selected in the final 5-variable macro "
        "specification. Plain rows: a candidate with a genuine primary-horizon signal, "
        "dropped in favour of a selected or another candidate variable (Section 4.2). "
        "Italic rows: not significant at any primary horizon, any factor."
    )
    fig.text(0.040, 0.004, note,
             ha="left", va="bottom",
             fontsize=6.5, style="italic", color="#444444", wrap=True)

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("Appendix IV.5 - Granger Causality Tables (booktabs style)")
    print("=" * 60)

    df = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "multi_horizon_granger.parquet")
    )
    print(f"Loaded {len(df):,} rows | {df['base_var'].nunique()} variables | "
          f"{df['horizon_w'].nunique()} horizons | {df['factor'].nunique()} factors")

    for factor, meta in FACTOR_META.items():
        print(f"  Generating {meta['math_label']} ...", end=" ", flush=True)
        fig = make_factor_figure(df, factor)
        out = os.path.join(FIGURES_DIR, meta["fname"])
        fig.savefig(out, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"saved -> {meta['fname']}")

    print("  Generating combined figure ...", end=" ", flush=True)
    fig = make_combined_figure(df)
    out = os.path.join(FIGURES_DIR, "fig_granger_combined.png")
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("saved -> fig_granger_combined.png")

    print("\nDone. Embed in main.qmd with:")
    print("  ![](../report/figures/fig_granger_combined.png){width=100%}")


if __name__ == "__main__":
    main()
