"""
04_granger.py - Phase 3b: Granger causality tests (Section 4.2).

Tests whether macro variables carry predictive information for NS factors,
and checks for bidirectional causality (simultaneity risk with Inventory).

STEP 1 - Raw levels (standard specification, matches CLAUDE.md test pairs)
STEP 2 - Multi-lookback changes (motivated by DRA 2006 and user insight):
  Raw macro LEVELS may appear null in Granger tests when the effect on the
  curve has an unknown delay. A 4W change in DXY captures cumulative USD
  strengthening; a 13W change captures a structural shift. Testing multiple
  lookback windows (1W, 4W, 13W) and rolling z-scores identifies the
  optimal macro specification for the VAR.

Inputs  : data/processed/ns_factors.parquet
          data/processed/macro.parquet
          data/processed/macro_changes.parquet   (NEW - from 01_data_cleaning.py)
          data/processed/lag_selection.parquet
Outputs : data/processed/granger_results.parquet        (Step 1, levels)
          data/processed/granger_results_changes.parquet (Step 2, changes)
          Console output for Section 4.2

Run: python code/04_granger.py

Step 1 tests (raw levels):
  DXY        → β₀  (USD strength → copper price level)
  Inventory  → β₁  (storage theory: scarcity → backwardation)
  β₁         → Inventory  (bidirectional - Theory of Storage feedback)
  VIX        → β₂  (financial stress → curve curvature)
  US3M       → β₀  (short-rate / carry cost → price level)
  GPR        → β₁  (geopolitical risk → curve slope)

Step 2 tests (multi-lookback changes):
  For each lookback window w ∈ {1W, 4W, 13W, 26W} and z-score:
    DXY_{d_w}W       → β₀, β₁
    inventory_{d_w}W → β₁
    VIX_{d_w}W       → β₂
    US3M_{d_w}W      → β₀, β₁
    GPR_{d_w}W       → β₁  (see main.qmd Section 3.1 for the sub-sample
                             stability evidence motivating inclusion)

NOTE: T10Y2Y was tested and is retained in macro.parquet for the Appendix IV
robustness table, but is NOT part of the production macro set (superseded
by US3M - see 05_models.py). This script tests the variables that actually
enter the VAR/LP, i.e. US3M and GPR, not T10Y2Y.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from config import TRAIN_END, VAR_MAX_LAGS, PROCESSED_DATA_DIR


def run_granger(endog: pd.DataFrame, cause: str, effect: str,
                max_lag: int) -> pd.DataFrame:
    """
    Test whether `cause` Granger-causes `effect` using statsmodels.

    Returns DataFrame with p-values for each lag order (F-test).
    """
    # statsmodels expects [effect, cause] order
    data = endog[[effect, cause]].dropna()
    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    rows = []
    for lag, res in results.items():
        f_stat = res[0]["ssr_ftest"][0]
        p_val  = res[0]["ssr_ftest"][1]
        rows.append({
            "cause":  cause,
            "effect": effect,
            "lag":    lag,
            "F_stat": round(f_stat, 4),
            "p_val":  round(p_val,  4),
            "significant_5pct": p_val < 0.05,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Phase 3b - Granger Causality Tests")
    print("=" * 60)

    # Load data
    factors = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet")
    )[["beta0", "beta1", "beta2"]]
    macro   = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro.parquet"))
    lag_sel = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "lag_selection.parquet"))

    recommended_lag = int(lag_sel["recommended"].iloc[0])
    max_test_lag    = max(recommended_lag, VAR_MAX_LAGS)
    print(f"VAR-selected lag p = {recommended_lag} (from Phase 3a)")
    print(f"Testing bivariate Granger causality across lags 1–{max_test_lag} "
          f"(not just the VAR-selected lag - a relationship invisible at p={recommended_lag} "
          f"may still appear at a different lag).")

    # Restrict to training period
    factors_train = factors.loc[:TRAIN_END]
    macro_train   = macro.loc[:TRAIN_END]
    endog = pd.concat([factors_train, macro_train], axis=1).dropna()

    # ── Granger causality tests ────────────────────────────────────────────────
    # Hypotheses from theoretical framework (Section 2.4, 3.2)
    test_pairs = [
        ("DXY",            "beta0",  "USD strength → copper price level"),
        ("DXY",            "beta1",  "USD strength → curve slope"),
        ("inventory",      "beta1",  "Storage scarcity → backwardation (Theory of Storage, unlagged)"),
        ("beta1",          "inventory", "Curve slope → inventory (bidirectional - simultaneity check)"),
        ("inventory_lag2", "beta1",  "Lagged storage scarcity → backwardation (VAR specification)"),
        ("VIX",            "beta2",  "Financial stress → curve curvature (theoretical channel)"),
        ("VIX",            "beta0",  "Financial stress → price level (risk-off/flight-to-quality selling)"),
        ("US3M",           "beta0",  "Short-rate level (cost of carry) → price level"),
        ("US3M",           "beta1",  "Short-rate level → curve slope"),
        ("GPR",             "beta1",  "Geopolitical risk (level) → curve slope"),
        ("T10Y2Y",         "beta0",  "[Appendix IV robustness] Yield spread → price level"),
        ("T10Y2Y",         "beta1",  "[Appendix IV robustness] Yield spread → curve slope"),
    ]

    all_results = []
    print("\n" + "─" * 60)

    for cause, effect, description in test_pairs:
        if cause not in endog.columns or effect not in endog.columns:
            print(f"  SKIP: {cause} not in dataset (check macro data loading)")
            continue

        print(f"\nTest: {cause} → {effect}  [{description}]")
        res_df = run_granger(endog, cause, effect, max_lag=max_test_lag)
        print(res_df.to_string(index=False))
        all_results.append(res_df)

        # Flag if significant at ANY tested lag, not just the VAR-selected one
        any_sig = res_df["significant_5pct"].any()
        if any_sig:
            sig_lags = res_df.loc[res_df["significant_5pct"], "lag"].tolist()
            print(f"  >>> SIGNIFICANT at lag(s) {sig_lags}: {cause} Granger-causes {effect}")
        else:
            print(f"  >>> NOT significant at any lag 1–{max_test_lag}")

    # ── Simultaneity warning ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Simultaneity check (Theory of Storage feedback loop):")
    inv_to_b1 = [r for r in all_results if r["cause"].iloc[0] == "inventory"
                 and r["effect"].iloc[0] == "beta1"]
    b1_to_inv = [r for r in all_results if r["cause"].iloc[0] == "beta1"
                 and r["effect"].iloc[0] == "inventory"]

    if inv_to_b1 and b1_to_inv:
        inv_sig = inv_to_b1[0]["significant_5pct"].any()
        b1_sig  = b1_to_inv[0]["significant_5pct"].any()
        if inv_sig and b1_sig:
            print("  BIDIRECTIONAL causality detected: inventory ↔ β₁ (unlagged)")
            print(f"  Mitigation: INVENTORY_LAG = 2 periods already applied - VAR uses inventory_lag2.")
            print("  Document this as an endogeneity limitation in Section 4.2.")
        else:
            print("  No strong bidirectional causality detected (unlagged inventory).")

    # ── Save Step 1 results ────────────────────────────────────────────────────
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out_path = os.path.join(PROCESSED_DATA_DIR, "granger_results.parquet")
        combined.to_parquet(out_path)
        print(f"\nSaved Granger results (levels): {out_path}")

    # ── STEP 2: Multi-lookback macro change variables ──────────────────────────
    changes_path = os.path.join(PROCESSED_DATA_DIR, "macro_changes.parquet")
    if not os.path.exists(changes_path):
        print("\n[SKIP] macro_changes.parquet not found - run 01_data_cleaning.py first.")
    else:
        print("\n" + "=" * 60)
        print("STEP 2 - Granger Tests: Multi-Lookback Macro Changes")
        print("Motivation: raw level Granger null may reflect unknown lag/integration")
        print("horizon. Testing Δ(1W), Δ(4W), Δ(13W), z(52W) per variable.")
        print("=" * 60)

        macro_changes = pd.read_parquet(changes_path)
        factors_train_c = factors.loc[:TRAIN_END]

        # Build a single endog combining NS factors + all change variables
        # All factors in levels (no beta0 differencing here - bivariate tests)
        endog_changes = pd.concat(
            [factors_train_c[["beta0", "beta1", "beta2"]],
             macro_changes.loc[:TRAIN_END]],
            axis=1
        ).dropna()

        # Key change-variable → factor pairs to test
        # Organised by variable and then window so results are easy to compare
        change_test_pairs = []
        # NOTE: "inventory_lag2" (not "inventory") is tested here for the
        # change transform - the production VAR uses inventory_lag2_d4W (the
        # 4-week inventory change, lagged 2 weeks) specifically so the
        # regressor cannot have been contemporaneously influenced by the same
        # week's curve shape (INVENTORY_LAG=2, config.py). Differencing the
        # RAW (unlagged) series instead would reintroduce exactly the
        # simultaneity risk the lag exists to remove - see main.qmd Section 3.1.
        for var_base in ["DXY", "inventory_lag2", "VIX", "US3M", "GPR", "T10Y2Y"]:
            windows = ["d1W", "d4W", "d13W", "d26W", "z52W"]
            for w in windows:
                col = f"{var_base}_{w}"
                if col not in endog_changes.columns:
                    continue
                if var_base == "DXY":
                    change_test_pairs += [
                        (col, "beta0", f"USD change ({w}) → price level"),
                        (col, "beta1", f"USD change ({w}) → slope"),
                    ]
                elif var_base == "inventory_lag2":
                    change_test_pairs += [
                        (col, "beta1", f"Lagged inventory change ({w}) → slope (Theory of Storage)"),
                    ]
                elif var_base == "VIX":
                    change_test_pairs += [
                        (col, "beta2", f"VIX change ({w}) → curvature"),
                    ]
                elif var_base == "US3M":
                    change_test_pairs += [
                        (col, "beta0", f"Short-rate change ({w}) → price level"),
                        (col, "beta1", f"Short-rate change ({w}) → slope"),
                    ]
                elif var_base == "GPR":
                    change_test_pairs += [
                        (col, "beta1", f"GPR change ({w}) → slope"),
                    ]
                elif var_base == "T10Y2Y":
                    change_test_pairs += [
                        (col, "beta0", f"[Appendix IV] Yield spread change ({w}) → price level"),
                    ]

        all_change_results = []
        print(f"\n{'Cause':35s} {'Effect':8s} {'Lag':4s} {'F':>8s} {'p':>8s} {'Sig*'}")
        print("-" * 75)

        for cause, effect, description in change_test_pairs:
            if cause not in endog_changes.columns:
                continue
            res_df = run_granger(endog_changes, cause, effect, max_lag=max_test_lag)
            all_change_results.append(res_df)

            # Print one-line summary per pair (best lag across all tested)
            best_row = res_df.loc[res_df["p_val"].idxmin()]
            sig_flag = "***" if best_row["p_val"] < 0.01 else (
                       "**"  if best_row["p_val"] < 0.05 else (
                       "*"   if best_row["p_val"] < 0.10 else ""))
            print(f"  {cause:33s} → {effect:6s}  "
                  f"lag={int(best_row['lag'])}  "
                  f"F={best_row['F_stat']:7.3f}  "
                  f"p={best_row['p_val']:.4f}  {sig_flag}")

        # Summary table: for each (variable, window) highlight best p-value
        print("\n" + "─" * 60)
        print("SUMMARY - minimum p-value across all tested lags:")
        print(f"{'Variable':35s} {'β₀ p_min':>10s} {'β₁ p_min':>10s} {'β₂ p_min':>10s}")
        print("-" * 70)
        if all_change_results:
            combined_c = pd.concat(all_change_results, ignore_index=True)
            for cause in combined_c["cause"].unique():
                sub = combined_c[combined_c["cause"] == cause]
                p_b0 = sub[sub["effect"] == "beta0"]["p_val"].min() if "beta0" in sub["effect"].values else np.nan
                p_b1 = sub[sub["effect"] == "beta1"]["p_val"].min() if "beta1" in sub["effect"].values else np.nan
                p_b2 = sub[sub["effect"] == "beta2"]["p_val"].min() if "beta2" in sub["effect"].values else np.nan
                def fmt(v):
                    if np.isnan(v): return "        -"
                    s = "***" if v < 0.01 else ("**" if v < 0.05 else ("*" if v < 0.10 else "  "))
                    return f"  {v:.4f}{s}"
                print(f"  {cause:33s} {fmt(p_b0):>12s} {fmt(p_b1):>12s} {fmt(p_b2):>12s}")

            # Save
            out_c = os.path.join(PROCESSED_DATA_DIR, "granger_results_changes.parquet")
            combined_c.to_parquet(out_c)
            print(f"\nSaved Granger results (changes): {out_c}")

            # Identify best specification per variable
            print("\n" + "─" * 60)
            print("RECOMMENDATION - best macro specification per variable:")
            for var_base in ["DXY", "inventory_lag2", "VIX", "US3M", "GPR", "T10Y2Y"]:
                candidates = [c for c in combined_c["cause"].unique()
                              if c.startswith(var_base + "_")]
                if not candidates:
                    continue
                best_overall = combined_c[combined_c["cause"].isin(candidates)
                              ]["p_val"].min()
                best_row_r = combined_c[
                    (combined_c["cause"].isin(candidates)) &
                    (combined_c["p_val"] == best_overall)
                ].iloc[0]
                sig = "SIGNIFICANT" if best_overall < 0.05 else "not significant"
                print(f"  {var_base:12s}: best = {best_row_r['cause']:25s} "
                      f"→ {best_row_r['effect']}  p={best_overall:.4f}  [{sig}]")

    # ── Sub-sample stability, final variable set ───────────────────────────────
    # main.qmd Section 4.2 and Appendix III cite an early-half vs late-half
    # split for inventory, US3M, and GPR, checking whether "stable across
    # training" is itself a stable property, not a full-sample fluke. Saved
    # here, from the same endog_changes block STEP 2 already builds, so every
    # number in that table traces back to this script rather than being
    # computed ad hoc.
    if all_change_results:
        print("\n" + "=" * 60)
        print("STEP 3 - Sub-Sample Stability: Final Variable Set")
        print("Splitting the training sample at its midpoint. Is each variable's")
        print("Granger relationship stable, or driven by one half of training?")
        print("=" * 60)

        stability_targets = [
            ("inventory_lag2_d4W", "beta1"),
            ("US3M_d1W",           "beta0"),
            ("GPR_d26W",           "beta1"),
        ]
        mid_date = endog_changes.index[len(endog_changes) // 2]
        halves = [
            ("early", endog_changes.loc[:mid_date]),
            ("late",  endog_changes.loc[mid_date:]),
            ("full",  endog_changes),
        ]
        stability_rows = []
        for cause, effect in stability_targets:
            for half_name, half_df in halves:
                data = half_df[[effect, cause]].dropna()
                res = grangercausalitytests(data, maxlag=VAR_MAX_LAGS, verbose=False)
                p_min = min(res[lag][0]["ssr_ftest"][1] for lag in range(1, VAR_MAX_LAGS + 1))
                stability_rows.append({
                    "cause": cause, "effect": effect, "sub_sample": half_name,
                    "n_obs": len(data), "p_val_min": round(p_min, 4),
                    "significant_5pct": p_min < 0.05,
                })
                print(f"  {cause:20s} -> {effect}  [{half_name:5s}, N={len(data):3d}]  "
                      f"min p={p_min:.4f}")

        stability_df = pd.DataFrame(stability_rows)
        stability_path = os.path.join(PROCESSED_DATA_DIR, "granger_subsample_stability.parquet")
        stability_df.to_parquet(stability_path)
        print(f"\nSaved sub-sample stability: {stability_path}")

    print("\nPhase 3b complete.")
    print("Next: run python code/05_models.py")


if __name__ == "__main__":
    main()
