"""
13d_walkforward_stability_diagnostic.py - DIAGNOSTIC ONLY. Determines how
much history each model needs before its coefficients can be trusted, using
criteria that NEVER reference forecast accuracy, BSS, or AUC -- so the choice
of burn-in start for the (not yet built) walk-forward residual-pool fix is
not itself a look-ahead-tainted, outcome-driven pick.

STANDALONE / NON-DESTRUCTIVE. Reads existing processed data only. Writes its
own suffixed output. Does not touch 07/13/14/15, config.py, or main.qmd.
Nothing about the production pipeline or the thesis text changes because of
this script -- it exists purely to inform a decision before anything is built.

TWO INDEPENDENT, LOOK-AHEAD-FREE CRITERIA:

  (1) MECHANICAL FLOOR -- observations required per estimated parameter, a
      standard regression rule of thumb (~10-20 obs/parameter), applied to
      each model's own, fixed regressor count. For the EW-weighted model
      (Model 3), also reports the EFFECTIVE sample size at each window
      length via Kish's formula, n_eff = (sum w)^2 / sum(w^2), since
      recency weighting means a calendar year of data does not count as a
      full year's worth of independent information toward the fit.

  (2) COEFFICIENT-STABILITY DIAGNOSTIC -- walks the SAME expanding-window
      procedure Phase 4 already uses (07_forecast_evaluation.py) forward
      from the earliest mechanically-valid origin through TRAIN_END only
      (2022-12-31 -- deliberately NOT into the test period, to keep this
      diagnostic uncontaminated by anything resembling the eventual
      out-of-sample result). At each origin, records the fitted
      coefficient matrix's Frobenius norm and the week-to-week CHANGE in
      that norm. The logic is the same as an MCMC burn-in diagnostic:
      discard samples until the chain (here, the coefficient sequence)
      settles down, decided by watching the chain itself, never by
      checking whether the eventual answer looks good.

Horizons checked: 4W (1M) and 13W (3M) -- the thesis's two primary horizons.
26W is not run here since a longer horizon can only need >= as much history
as the shorter ones to reach the same row count; if 4W/13W stabilise by
date X, 26W's own binding constraint is examined separately once the
residual-pool script is actually built.

Inputs : data/processed/ns_factors.parquet
         data/processed/macro_changes.parquet, macro.parquet
         data/processed/lag_selection.parquet
         data/processed/lambda_ew_optimal.parquet
Outputs: data/processed/walkforward_stability_diagnostic.parquet
         report/figures/fig_walkforward_stability_diagnostic.png

Run: python code/13d_walkforward_stability_diagnostic.py   (~30-60 seconds)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    TRAIN_END, PROCESSED_DATA_DIR, FIGURES_DIR,
    MACRO_CHANGE_COLS, MACRO_LEVEL_COLS, MACRO_VAR_SPEC, LAMBDA_EW,
)
from utils import build_lp_matrices, fit_lp

DIAGNOSTIC_HORIZONS = [4, 13]
MODEL_LABELS = {1: "AR-Direct", 2: "LP + Macro", 3: "LP + Macro + EW"}
OBS_PER_PARAM_LOW, OBS_PER_PARAM_HIGH = 10, 20  # standard regression rule-of-thumb range


def kish_effective_n(weights: np.ndarray) -> float:
    """Effective sample size for a weighted sum (Kish 1965): (sum w)^2 / sum(w^2).
    Equals len(weights) when all weights are equal; shrinks as weighting
    concentrates on a smaller effective subset of the data."""
    return float((weights.sum() ** 2) / (weights ** 2).sum())


def walk_forward_coefficients(factors, macro, macro_cols, p, q, lambda_ew, h, origins):
    """One row per origin: n_obs, k_params, n_eff (model 3 only), coef Frobenius
    norm, and the week-to-week change in that norm. Mirrors exactly how
    07_forecast_evaluation.py fits at each test date, just walked earlier and
    further, and only through TRAIN_END (see module docstring)."""
    records = {1: [], 2: [], 3: []}
    prev_coef = {1: None, 2: None, 3: None}

    for t in origins:
        factors_train = factors.loc[:t]
        macro_train = macro.loc[:t]

        X_ar, Y_ar, _ = build_lp_matrices(factors_train, None, p=p, q=0, h=h, macro_cols=None)
        X_lp, Y_lp, _ = build_lp_matrices(factors_train, macro_train, p=p, q=q, h=h, macro_cols=macro_cols)

        if X_ar is not None and len(X_ar) >= p + 2:
            coef1 = fit_lp(X_ar, Y_ar)
            norm1 = float(np.linalg.norm(coef1))
            change1 = float(np.linalg.norm(coef1 - prev_coef[1])) if prev_coef[1] is not None else np.nan
            resid1 = Y_ar - X_ar @ coef1
            records[1].append({"date": t, "n_obs": len(X_ar), "k_params": X_ar.shape[1],
                                "n_eff": len(X_ar), "coef_norm": norm1, "coef_change": change1,
                                "resid_rmse": float(np.sqrt(np.mean(resid1 ** 2)))})
            prev_coef[1] = coef1

        if X_lp is not None and len(X_lp) >= p + q * len(macro_cols) + 2:
            coef2 = fit_lp(X_lp, Y_lp)
            norm2 = float(np.linalg.norm(coef2))
            change2 = float(np.linalg.norm(coef2 - prev_coef[2])) if prev_coef[2] is not None else np.nan
            resid2 = Y_lp - X_lp @ coef2
            records[2].append({"date": t, "n_obs": len(X_lp), "k_params": X_lp.shape[1],
                                "n_eff": len(X_lp), "coef_norm": norm2, "coef_change": change2,
                                "resid_rmse": float(np.sqrt(np.mean(resid2 ** 2)))})
            prev_coef[2] = coef2

            T_w = len(X_lp)
            w = lambda_ew ** np.arange(T_w - 1, -1, -1)
            coef3 = fit_lp(X_lp, Y_lp, weights=w)
            norm3 = float(np.linalg.norm(coef3))
            change3 = float(np.linalg.norm(coef3 - prev_coef[3])) if prev_coef[3] is not None else np.nan
            resid3 = Y_lp - X_lp @ coef3
            n_eff3 = kish_effective_n(w)
            records[3].append({"date": t, "n_obs": len(X_lp), "k_params": X_lp.shape[1],
                                "n_eff": n_eff3, "coef_norm": norm3, "coef_change": change3,
                                "resid_rmse": float(np.sqrt(np.mean(resid3 ** 2)))})
            prev_coef[3] = coef3

    return {m: pd.DataFrame(r) for m, r in records.items()}


def find_stabilisation_date(df: pd.DataFrame, window: int = 26, tolerance: float = 1.5) -> pd.Timestamp | None:
    """First date after which the trailing `window`-origin mean of coef_change
    never again exceeds `tolerance` times its own eventual (last 20%) median
    level, for the remainder of the series. A disclosed, simple stabilisation
    rule -- not the only possible one -- applied identically across models so
    the comparison across them is fair."""
    s = df.set_index("date")["coef_change"].dropna()
    if len(s) < window * 2:
        return None
    trailing = s.rolling(window).mean()
    terminal_level = s.iloc[-int(len(s) * 0.2):].median()
    threshold = tolerance * terminal_level
    for i in range(len(trailing)):
        if trailing.iloc[i:].max() <= threshold and not np.isnan(trailing.iloc[i]):
            return trailing.index[i]
    return None


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("=" * 70)
    print("Walk-forward coefficient stability diagnostic (informs burn-in choice)")
    print("=" * 70)

    factors = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet"))[
        ["beta0", "beta1", "beta2"]]
    macro_changes = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro_changes.parquet"))
    macro_levels = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro.parquet"))
    macro = pd.concat([macro_changes[MACRO_CHANGE_COLS], macro_levels[MACRO_LEVEL_COLS]], axis=1)
    macro_cols = MACRO_VAR_SPEC
    lag_sel = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "lag_selection.parquet"))
    p = int(lag_sel["recommended"].iloc[0])
    q = 1
    opt_path = os.path.join(PROCESSED_DATA_DIR, "lambda_ew_optimal.parquet")
    lambda_ew = float(pd.read_parquet(opt_path)["lambda_ew_optimal"].iloc[0]) if os.path.exists(opt_path) else LAMBDA_EW

    # ── (1) Mechanical floor, computed from model structure alone ──────────
    k1 = 3 * p + 1
    k2 = 3 * p + len(macro_cols) * q + 1
    k3 = k2  # same design as model 2; WLS does not change the parameter count
    print(f"\np={p}, q={q}, lambda_ew={lambda_ew:.4f}, macro vars: {macro_cols}")
    print("\nMechanical floor (obs/parameter rule of thumb):")
    for m, k in [(1, k1), (2, k2), (3, k3)]:
        lo, hi = OBS_PER_PARAM_LOW * k, OBS_PER_PARAM_HIGH * k
        print(f"  Model {m} ({MODEL_LABELS[m]}): k={k} params -> "
              f"{lo}-{hi} obs -> {lo/52:.1f}-{hi/52:.1f} years of weekly data (raw N)")
    print("  NOTE: NS factors are highly persistent (near-unit-root, Section 3.3), so effective\n"
          "  independent information per week is below 1 -- treat the figures above as an\n"
          "  optimistic (too-early) floor, not a sufficient condition on their own.")

    macro_start = macro[MACRO_CHANGE_COLS + MACRO_LEVEL_COLS].dropna().index.min()
    factors_start = factors.dropna().index.min()
    print(f"\nEarliest data availability: NS factors from {factors_start.date()}, "
          f"all 5 macro variables from {macro_start.date()}")

    earliest_origin = macro_start  # binding constraint for models 2 and 3; model 1 is available earlier
    all_stability_dfs = {}

    for h in DIAGNOSTIC_HORIZONS:
        print(f"\n{'-'*70}\nHorizon h={h}W\n{'-'*70}")
        origins = factors.loc[earliest_origin:TRAIN_END].index
        results = walk_forward_coefficients(factors, macro, macro_cols, p, q, lambda_ew, h, origins)

        for m in [1, 2, 3]:
            df = results[m]
            df["horizon"] = h
            df["model"] = m
            all_stability_dfs[(m, h)] = df
            stab_date = find_stabilisation_date(df)
            n_at_stab = df.loc[df["date"] == stab_date, "n_obs"].iloc[0] if stab_date is not None and (df["date"] == stab_date).any() else np.nan
            terminal_change = df["coef_change"].iloc[-int(len(df) * 0.2):].median()
            print(f"  Model {m} ({MODEL_LABELS[m]}): {len(df)} origins fitted "
                  f"({df['date'].min().date()} -> {df['date'].max().date()}), "
                  f"terminal median week-to-week coef change = {terminal_change:.5f}")
            if stab_date is not None:
                years_of_data = n_at_stab / 52
                print(f"    -> stabilises at {stab_date.date()} (N={n_at_stab} obs, "
                      f"~{years_of_data:.1f} yrs of data at that point)")
            else:
                print(f"    -> did not clearly stabilise by {TRAIN_END} under this rule")

    full_df = pd.concat(all_stability_dfs.values(), ignore_index=True)
    out_path = os.path.join(PROCESSED_DATA_DIR, "walkforward_stability_diagnostic.parquet")
    full_df.to_parquet(out_path)
    print(f"\nSaved: {out_path}")

    # ── Plot: coefficient trajectory + week-to-week change, both horizons ──
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    colors = {1: "#999999", 2: "#5a9bd4", 3: "#1a4a8a"}
    for col, h in enumerate(DIAGNOSTIC_HORIZONS):
        for m in [1, 2, 3]:
            df = all_stability_dfs[(m, h)]
            axes[0, col].plot(df["date"], df["coef_norm"], color=colors[m], lw=1.1, label=MODEL_LABELS[m])
            axes[1, col].plot(df["date"], df["coef_change"], color=colors[m], lw=0.9, alpha=0.8)
        axes[0, col].set_title(f"h={h}W")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("Origin date")
    axes[0, 0].set_ylabel("Coefficient matrix Frobenius norm")
    axes[1, 0].set_ylabel("Week-to-week change (log scale)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax in axes.ravel():
        ax.grid(lw=0.3, color="#dddddd")
    fig.suptitle("Walk-forward coefficient stability by model and horizon (expanding window, ends at TRAIN_END)")
    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig_walkforward_stability_diagnostic.png")
    fig.savefig(fig_path, dpi=150, facecolor="white")
    print(f"Saved: {fig_path}")

    print("\nDone. No production file, config.py, or main.qmd was modified.")


if __name__ == "__main__":
    main()
