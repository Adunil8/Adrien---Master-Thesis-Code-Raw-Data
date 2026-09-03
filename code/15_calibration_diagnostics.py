"""
15_calibration_diagnostics.py - Diagnose WHERE (if anywhere) the probability
outputs carry genuine rank-discrimination skill, before deciding how to
frame calibration results in the thesis.

CONTEXT: 14_platt_calibration.py, once correctly constrained to a >= 0,
collapsed to a=0 (predict the base rate) at the PRIMARY grid (model=3,
maturity=3M, threshold=8%, horizons 1M/3M/6M). This script checks four
possible explanations before concluding the null result is general:
  (1) Sub-period effect: does skill exist within the stable period or the
      tariff-shock period separately, but wash out when pooled?
  (2) Model choice: does AR-Direct (best Ch.5 hit rate) show skill where
      LP+Macro+EW does not?
  (3) Horizon specificity: is skill present at some horizons and not others?
  (4) Maturity specificity: is skill confined to the 3M contract used as the
      Chapter 5 headline cell, or does it hold across the 1M-6M curve?

METRIC: AUC (probability that a randomly chosen "event" week has a higher
predicted probability than a randomly chosen "non-event" week) via the
Mann-Whitney rank-sum identity - a direct, assumption-free discrimination
measure, independent of any calibration fit. AUC=0.5 is chance; AUC=1.0 is
perfect discrimination; AUC<0.5 is discrimination in the WRONG direction.
Computed via scipy.stats.rankdata (no sklearn, per project convention).

VERDICT LABELS - two failure modes are distinguished explicitly (an earlier
version of this script mislabeled both as "INVERTED", see classify_verdict()):
  UNDEFINED: one outcome class is empty (n_pos=0 or n_neg=0) in that slice,
    almost always because a short sub-period had a persistent one-directional
    move leaving no comparison group. AUC is not computed, not "bad" - it does
    not exist for that slice.
  INSUFFICIENT SAMPLE: both classes non-empty but the smaller has < 5 members
    -- AUC is computable but not evidence of anything, given by construction.

Inputs  : data/processed/probability_outputs_bootstrap.parquet
Outputs : data/processed/calibration_diagnostics.parquet (printed + saved)

Run: python code/15_calibration_diagnostics.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config import PROCESSED_DATA_DIR, HORIZON_LABELS, MATURITIES_MONTHS

MODEL_LABELS = {0: "Random Walk", 1: "AR-Direct", 2: "LP+Macro", 3: "LP+Macro+EW"}
TARIFF_SHOCK_START = pd.Timestamp("2025-04-11")  # aligned with 08_probability_outputs.py / 10_subperiod_analysis.py official split date
PRIMARY_MATURITY, PRIMARY_THRESHOLD = 3, 0.08


def auc(p: np.ndarray, y: np.ndarray) -> tuple[float, int, int]:
    """AUC via Mann-Whitney rank-sum. Returns (auc, n_pos, n_neg)."""
    y = y.astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan, int(n_pos), int(n_neg)
    ranks = rankdata(p)
    auc_val = (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc_val), int(n_pos), int(n_neg)


MIN_CLASS_COUNT = 5  # below this, AUC is technically defined but statistically
                      # too noisy (e.g. n_pos=1 lets a single observation decide
                      # the whole value) to support any verdict


def classify_verdict(a: float, n_pos: int, n_neg: int) -> str:
    """AUC -> verdict label, with two failure modes handled explicitly (both
    were previously mislabeled INVERTED by an incomplete if/elif/else):

    1. UNDEFINED: n_pos=0 or n_neg=0 -> auc() returns NaN. AUC is the
       probability a random "event" week ranks above a random "non-event"
       week; with one class empty there is nothing to rank against, so the
       quantity does not exist. Must be checked BEFORE the a>0.58 /
       0.42<=a<=0.58 comparisons: NaN comparisons are silently False in
       Python, so an unguarded chain falls through to the final `else`
       (INVERTED) even though nothing was computed.
    2. INSUFFICIENT SAMPLE: both classes non-empty but the smaller one has
       fewer than MIN_CLASS_COUNT members -> AUC is mathematically defined
       but is effectively "where does this handful of points rank," not
       evidence of skill either way.
    """
    if np.isnan(a):
        return "UNDEFINED (degenerate class - no variation to rank)"
    if min(n_pos, n_neg) < MIN_CLASS_COUNT:
        return f"INSUFFICIENT SAMPLE (min class n={min(n_pos, n_neg)})"
    if a > 0.58:
        return "GENUINE SKILL"
    if a >= 0.42:
        return "weak/none"
    return "INVERTED (worse than random)"


def main():
    print("=" * 90)
    print("Calibration diagnostics - where (if anywhere) is there genuine rank skill?")
    print("=" * 90)

    prob_all = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "probability_outputs_bootstrap.parquet"))
    prob_all["date"] = pd.to_datetime(prob_all["date"])
    prob_all["period"] = np.where(prob_all["date"] >= TARIFF_SHOCK_START, "tariff_shock", "stable")

    # Checks 1-3 (full-period horizon scan, sub-period split) stay on the
    # PRIMARY_MATURITY cell that Chapter 5's narrative tables report.
    prob = prob_all[(prob_all.maturity == PRIMARY_MATURITY) & (prob_all.threshold == PRIMARY_THRESHOLD)].copy()

    rows = []

    # ── Check 3: horizon specificity, all 4 models, full period ───────────────
    print("\n--- Full out-of-sample period (2023-2026), all models, all horizons ---")
    print(f"{'Model':<14} {'Horizon':>8} {'Dir':>5} {'AUC':>7} {'N_pos':>6} {'N_neg':>6}  Verdict")
    print("-" * 70)
    for model_id in [0, 1, 2, 3]:
        for h in sorted(prob.horizon.unique()):
            sub = prob[(prob.model == model_id) & (prob.horizon == h)]
            for direction in ["up", "down"]:
                p_col, act_col = f"p_{direction}", f"actual_{direction}"
                valid = sub[[p_col, act_col]].dropna()
                if len(valid) < 20:
                    continue
                a, npos, nneg = auc(valid[p_col].values, valid[act_col].values)
                verdict = classify_verdict(a, npos, nneg)
                h_lbl = HORIZON_LABELS.get(h, f"{h}W")
                print(f"{MODEL_LABELS[model_id]:<14} {h_lbl:>8} {direction:>5} {a:>7.3f} "
                      f"{npos:>6} {nneg:>6}  {verdict}")
                rows.append({"scope": "full", "model": model_id, "horizon": h,
                             "direction": direction, "auc": round(a, 4),
                             "n_pos": npos, "n_neg": nneg, "verdict": verdict})

    # ── Check 1: sub-period split, primary model + horizons ───────────────────
    print("\n--- Sub-period split: stable (pre-Apr2025) vs tariff shock (Apr2025+) ---")
    print("(Primary grid: all 4 models, horizons 4W/13W/26W)")
    print(f"{'Model':<14} {'Horizon':>8} {'Dir':>5} {'Period':<13} {'AUC':>7} {'N':>5}  Verdict")
    print("-" * 80)
    for model_id in [0, 1, 2, 3]:
        for h in [4, 13, 26]:
            for direction in ["up", "down"]:
                p_col, act_col = f"p_{direction}", f"actual_{direction}"
                for period in ["stable", "tariff_shock"]:
                    sub = prob[(prob.model == model_id) & (prob.horizon == h) & (prob.period == period)]
                    valid = sub[[p_col, act_col]].dropna()
                    if len(valid) < 15:
                        continue
                    a, npos, nneg = auc(valid[p_col].values, valid[act_col].values)
                    verdict = classify_verdict(a, npos, nneg)
                    h_lbl = HORIZON_LABELS.get(h, f"{h}W")
                    print(f"{MODEL_LABELS[model_id]:<14} {h_lbl:>8} {direction:>5} {period:<13} "
                          f"{a:>7.3f} {len(valid):>5}  {verdict}")
                    rows.append({"scope": "subperiod", "model": model_id, "horizon": h,
                                 "direction": direction, "period": period, "auc": round(a, 4),
                                 "n_pos": npos, "n_neg": nneg, "verdict": verdict})

    # ── Check 4: maturity robustness - does the pattern found at the 3M    ────
    # headline cell (Checks 1-3) hold across the rest of the curve, or is it
    # specific to that one cell? All 6 maturities x 4 models x horizons
    # {4,13,26}W x both directions x both sub-periods.
    print("\n--- Maturity robustness: same sub-period grid, all maturities 1M-6M ---")
    maturities = sorted(prob_all.maturity.unique())
    for model_id in [0, 1, 2, 3]:
        for h in [4, 13, 26]:
            for direction in ["up", "down"]:
                p_col, act_col = f"p_{direction}", f"actual_{direction}"
                for period in ["stable", "tariff_shock"]:
                    for mat in maturities:
                        sub = prob_all[(prob_all.model == model_id) & (prob_all.horizon == h) &
                                       (prob_all.period == period) & (prob_all.maturity == mat) &
                                       (prob_all.threshold == PRIMARY_THRESHOLD)]
                        valid = sub[[p_col, act_col]].dropna()
                        if len(valid) < 10:
                            continue
                        a, npos, nneg = auc(valid[p_col].values, valid[act_col].values)
                        verdict = classify_verdict(a, npos, nneg)
                        rows.append({"scope": "maturity_grid", "model": model_id, "horizon": h,
                                     "direction": direction, "period": period, "maturity": mat,
                                     "auc": round(a, 4), "n_pos": npos, "n_neg": nneg,
                                     "verdict": verdict})

    grid = pd.DataFrame([r for r in rows if r["scope"] == "maturity_grid"])
    if not grid.empty:
        # condensed view: mean AUC and verdict mix per (maturity, model, period),
        # averaged over horizon and direction, restricted to well-posed cells
        # (drop UNDEFINED/INSUFFICIENT so the mean isn't distorted by NaN or noise)
        well_posed = grid[~grid["verdict"].str.startswith(("UNDEFINED", "INSUFFICIENT"))]
        summary = (well_posed.groupby(["maturity", "model", "period"])
                   .agg(mean_auc=("auc", "mean"), n_cells=("auc", "count"))
                   .round(3))
        print(summary.to_string())
        n_genuine = (grid["verdict"] == "GENUINE SKILL").sum()
        n_inverted = (grid["verdict"] == "INVERTED (worse than random)").sum()
        n_undefined = grid["verdict"].str.startswith("UNDEFINED").sum()
        n_insuff = grid["verdict"].str.startswith("INSUFFICIENT").sum()
        print(f"\n  Across {len(grid)} (model,horizon,direction,period,maturity) cells: "
              f"{n_genuine} GENUINE SKILL, {n_inverted} INVERTED, "
              f"{n_undefined} UNDEFINED, {n_insuff} INSUFFICIENT SAMPLE")

    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(PROCESSED_DATA_DIR, "calibration_diagnostics.parquet"))
    print(f"\nSaved: data/processed/calibration_diagnostics.parquet")

    # ── Summary ─────────────────────────────────────────────────────────────
    genuine = out[out["verdict"] == "GENUINE SKILL"]
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(genuine)} / {len(out)} (model, horizon, direction[, period]) "
          f"combinations show AUC > 0.58")
    if not genuine.empty:
        print(genuine.to_string(index=False))
    else:
        print("None. No configuration tested shows AUC meaningfully above 0.5.")


if __name__ == "__main__":
    main()
