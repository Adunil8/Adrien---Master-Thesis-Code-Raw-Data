"""
09_stylised_deal.py - Phase 6: Application to the stylised trade finance deal.

Illustrates how the probabilistic forecasts translate into a monitoring
signal for contingent liquidity capacity, compared against a reference
point built from trailing realised volatility.

Stylised deal (from config.py):
  Metal: copper, Notional: 1,000t, Tenor: 13W (3M)
  Position: "both" - outputs both SHORT and LONG estimates

MODEL AND DATA SOURCE (see 09c/13e/14b for the full chain of evidence
this rests on):
  - This script reads data/processed/deal_rolldown_probabilities_walkforward.parquet
    (09c_deal_rolldown_walkforward.py), the RAW (uncalibrated) probability,
    not deal_rolldown_calibrated.parquet's Platt-corrected one. Two separate
    fixes sit behind this file: the residual pool is now walk-forward and
    point-in-time-honest (13e), and Platt scaling is not used here at all,
    since a genuinely honest test of it (14b) found it does not survive
    out-of-sample, however good it looks fit-and-scored on the same window.
    Using the raw probability is the more conservative, defensible choice:
    it is the model's own signal, with no additional untested correction
    layered on top.
  - AR-Direct (model=1, DEAL_MODEL_ID below) is used for deployment. 09c's
    own AUC comparison across all four models under this script's
    volatility-threshold design confirms the same ranking Chapter 5 found
    under the fixed-threshold convention: AR-Direct has the strongest and
    most consistent skill for the SHORT-side (upward) signal, stable-period
    AUC 0.88, holding up well where the shock period is testable (AUC 0.80).
    The LONG-side (downward) signal is weaker and murkier across every
    model, including AR-Direct itself, on a thin sample (8 downward-
    crossing events in 118 stable-period dates): read as suggestive, not
    as validated skill on the long side.
  - There is no calibration-domain fallback any more: the raw probability
    is available for every test date, so the gap that the old Platt-based
    fallback existed to cover (no fit-worthy sample in the shock period)
    does not apply here.

WHAT "BUFFER" MEANS HERE, AND WHY THERE IS NO 8% ANYWHERE IN THIS SCRIPT:
  A fixed percentage threshold, meant to stand in for a TPA credit line,
  is not used here. That kind of number is real in the sense that TPAs do
  set credit thresholds, but Section 3.9 already documents that no single
  number can be cited for what any given bank actually uses, since it is a
  bilateral, unpublished credit decision. Testing against an asserted
  percentage would rest on a number this thesis cannot independently
  verify. The threshold used here instead is trailing realised volatility
  itself, for both the buffer sizing and the threshold definition:
  a self-referential question, "is the market about to move more than its
  own recent normal range implies," rather than a comparison against an
  external, asserted number. It is built from the front-month contract
  (LP1, m01, "closest to cash"), the same reference a treasury desk would
  use as a sense check on plausible price moves, independent of any model:
    1. Weekly log returns on LP1.
    2. Rolling 52-week (trailing 12-month) standard deviation of those
       returns, re-estimated at every date - a genuine trailing window,
       not expanding.
    3. Annualised, then scaled down to the deal's own tenor in weeks - 13
       weeks (3M) for this stylised deal, so a shorter or longer deal
       tenor would read a different number off the same underlying
       volatility, and the same tenor on two different dates reads two
       different numbers too, since volatility itself moves with market
       conditions (see utils.trailing_vol_reference and 09a's docstring).
    4. Applied to the deal's notional value (1,000t × 3M price) with no
       extra safety multiplier: the reference is exactly what trailing
       volatility implies for a plausible move over the deal's tenor,
       nothing more.
  This same number is also the barrier the model's simulated prices are
  tested against, so the dollar-sizing and the probability threshold are
  no longer two unrelated conventions bolted together: both come from the
  same sigma at the same date.
  Separately from all of this: neither the reference nor the model-informed
  estimate is cash a bank sets aside and leaves untouched for the deal's
  life. Margin calls are met daily, funded out of capital that is
  otherwise working (e.g. placed overnight at SOFR - Section 1.1). Both
  numbers here are MONITORING signals, re-estimated at every date, for how
  large a call the desk should be ready to fund at short notice.

Inputs  : data/processed/deal_rolldown_probabilities_walkforward.parquet
          data/processed/curves.parquet
Outputs : data/processed/stylised_deal_results.parquet
          report/figures/fig_liquidity_buffer.png
          report/figures/fig_case_studies.png

Run: python code/09a_deal_rolldown_probability.py  then  python code/09_stylised_deal.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import STYLISED_DEAL, PROCESSED_DATA_DIR, FIGURES_DIR

DEAL_MODEL_ID = 1   # AR-Direct - see module docstring; confirmed again under
                     # the volatility-threshold design by 09a's own AUC table


def compute_liquidity_buffer(prob_df: pd.DataFrame,
                              copper: pd.DataFrame) -> pd.DataFrame:
    """
    For each deal signing date in the test period, compute two monitoring
    estimates of the capital the desk should be ready to fund at short
    notice over the deal's own 3M life - not an amount of cash held aside
    (see the module docstring):
      - Volatility reference : notional × price × sigma_threshold (no
                                model, no probability - the treasury
                                sense check, sigma_threshold coming
                                straight from 09a's own computation)
      - Dynamic estimate (short): reference × P_raw(price rises past
                                    the reference's own +sigma barrier)
      - Dynamic estimate (long) : reference × P_raw(price falls past
                                    the reference's own -sigma barrier)

    P_raw is the model's own raw bootstrap probability, not a Platt-
    calibrated one (see module docstring): a genuinely honest test of Platt
    scaling did not survive out-of-sample, so no correction is layered on
    top here. There is no calibration-domain fallback either, since raw
    probabilities are available for every test date.

    prob_df here is already scoped to DEAL_MODEL_ID, tau=3M, h=13W by
    09c_deal_rolldown_walkforward.py, one row per date, each carrying its
    own sigma_threshold: there is no separate threshold dimension to
    filter on any more (see module docstring for why the old fixed
    5/8/10/15% grid does not apply to this specific illustration).
    """
    notional = STYLISED_DEAL["notional_tonnes"]

    sub = prob_df[prob_df["model"] == DEAL_MODEL_ID].copy()
    m3col = "m03"

    records = []
    for _, row in sub.iterrows():
        t = row["date"]
        if t not in copper.index or pd.isna(copper.loc[t, m3col]):
            continue
        price = float(copper.loc[t, m3col])
        if price <= 0:
            continue
        sigma_t = row["sigma_threshold"]
        if pd.isna(sigma_t):
            continue  # no full trailing 52-week window yet at this date

        p_rise = row["p_up"]
        p_fall = row["p_down"]
        notional_value = notional * price
        vol_reference  = notional_value * float(sigma_t)

        dyn_short = vol_reference * p_rise if not np.isnan(p_rise) else vol_reference
        dyn_long  = vol_reference * p_fall if not np.isnan(p_fall) else vol_reference

        records.append({
            "date":             t,
            "copper_price":     round(price, 2),
            "notional_usd":     round(notional_value, 0),
            "vol_threshold":    round(float(sigma_t), 4),
            "naive_buffer":     round(vol_reference, 0),
            "dynamic_buffer_short": round(dyn_short, 0),
            "dynamic_buffer_long":  round(dyn_long, 0),
            "P_rise_vol":       round(p_rise, 4) if not np.isnan(p_rise) else np.nan,
            "P_fall_vol":       round(p_fall, 4) if not np.isnan(p_fall) else np.nan,
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    if not df.empty:
        df["buffer_saving_short"] = df["naive_buffer"] - df["dynamic_buffer_short"]
        df["buffer_saving_long"]  = df["naive_buffer"] - df["dynamic_buffer_long"]

    return df


def print_summary(deal_df: pd.DataFrame) -> None:
    """
    Print summary statistics for the stylised deal illustration.

    Read these numbers as illustrative monitoring signals, not as a
    validated record of what a bank actually did or saved. There is no
    disclosed data on real trade finance liquidity decisions to compare
    against (Section 6.6 / 7 limitations) - the comparison here is between
    two ways of ESTIMATING a monitoring reference (trailing volatility
    alone, vs trailing volatility scaled by the model's raw probability
    of clearing that same volatility-implied bar), not a
    backtest against actual bank outcomes.
    """
    print("\n" + "─" * 60)
    print("STYLISED DEAL SUMMARY (1,000t copper, 3M tenor, volatility-based threshold)")
    print("─" * 60)
    print(f"  Test period:          {deal_df['date'].min().date()} → {deal_df['date'].max().date()}")
    print(f"  N deal signing dates: {len(deal_df)}")
    print(f"\n  Volatility threshold, 13W-scaled (mean):  {deal_df['vol_threshold'].mean() * 100:>6.1f}%")
    print(f"  Volatility threshold, annualised (mean):  {deal_df['vol_threshold'].mean() * np.sqrt(52 / STYLISED_DEAL['tenor_weeks']) * 100:>6.1f}%")
    print(f"\n  Vol reference, $ (mean):            ${deal_df['naive_buffer'].mean():>12,.0f}")
    print(f"  Dynamic estimate SHORT (mean):      ${deal_df['dynamic_buffer_short'].mean():>12,.0f}")
    print(f"  Dynamic estimate LONG  (mean):      ${deal_df['dynamic_buffer_long'].mean():>12,.0f}")
    print(f"\n  Mean capacity freed (SHORT):        ${deal_df['buffer_saving_short'].mean():>12,.0f}")
    print(f"  Mean capacity freed (LONG):         ${deal_df['buffer_saving_long'].mean():>12,.0f}")

    # High-risk periods
    pct_high_risk_short = (deal_df["P_rise_vol"] > 0.5).mean() * 100
    pct_high_risk_long  = (deal_df["P_fall_vol"] > 0.5).mean() * 100
    print(f"\n  % dates with P(rise past vol threshold)>50%:  {pct_high_risk_short:.1f}%  (elevated SHORT risk)")
    print(f"  % dates with P(fall past vol threshold)>50%:  {pct_high_risk_long:.1f}%  (elevated LONG risk)")


def plot_liquidity_buffer(deal_df: pd.DataFrame, out_dir: str) -> None:
    """Plot dynamic vs volatility-reference buffer over the test period."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Panel 1: monitoring-estimate comparison (SHORT position - most common in trade finance)
    ax1.fill_between(deal_df["date"], deal_df["naive_buffer"] / 1e6,
                     alpha=0.3, color="grey", label="Trailing-volatility reference")
    ax1.plot(deal_df["date"], deal_df["dynamic_buffer_short"] / 1e6,
             color="firebrick", linewidth=1.5, label="Model-informed estimate (SHORT)")
    ax1.set_ylabel("Capital to be ready to fund (USD millions)")
    ax1.set_title("Stylised Deal: Model-Informed vs Trailing-Volatility Monitoring Reference\n1,000t copper, 3M tenor, 3M horizon, volatility-based threshold")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # Panel 2: Probability time series
    ax2.plot(deal_df["date"], deal_df["P_rise_vol"], color="firebrick",
             linewidth=1.0, label="P(rise past vol threshold) - SHORT risk")
    ax2.plot(deal_df["date"], deal_df["P_fall_vol"], color="steelblue",
             linewidth=1.0, linestyle="--", label="P(fall past vol threshold) - LONG risk")
    ax2.axhline(0.5, color="grey", linestyle=":", linewidth=0.7,
                label="50% threshold")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Margin call probability")
    ax2.legend(loc="upper right")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_liquidity_buffer.png"), dpi=150)
    plt.close(fig)
    print("  Saved: fig_liquidity_buffer.png")


def plot_case_studies(deal_df: pd.DataFrame, out_dir: str) -> None:
    """
    Zoom into key stress episodes to show what the model signalled.
    For the test period (2023–2026): focus on Trump tariff announcements.
    Historical case studies (2020 COVID, 2022 Ukraine) shown if data available.
    """
    episodes = [
        ("2023-01-01", "2023-12-31", "2023 - Early Trump tariff signals"),
        ("2024-01-01", "2024-12-31", "2024"),
        ("2025-01-01", "2025-12-31", "2025 - Trump tariff implementation"),
    ]

    n_panels = sum(1 for s, e, _ in episodes
                   if not deal_df[(deal_df["date"] >= s) & (deal_df["date"] <= e)].empty)
    if n_panels == 0:
        print("  No data for case study windows - skipping fig_case_studies.png")
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=False)
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    for start, end, label in episodes:
        sub = deal_df[(deal_df["date"] >= start) & (deal_df["date"] <= end)]
        if sub.empty:
            continue
        ax = axes[ax_idx]
        ax.plot(sub["date"], sub["P_rise_vol"], color="firebrick",
                linewidth=1.2, label="P(rise past vol threshold)")
        ax.plot(sub["date"], sub["P_fall_vol"], color="steelblue",
                linewidth=1.2, linestyle="--", label="P(fall past vol threshold)")
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7)
        ax.set_ylim(0, 1)
        ax.set_title(label)
        ax.set_ylabel("Probability")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax_idx += 1

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_case_studies.png"), dpi=150)
    plt.close(fig)
    print("  Saved: fig_case_studies.png")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("Phase 6 - Stylised Deal Application")
    print("=" * 60)

    prob_path = os.path.join(PROCESSED_DATA_DIR, "deal_rolldown_probabilities_walkforward.parquet")
    cu_path   = os.path.join(PROCESSED_DATA_DIR, "curves.parquet")

    if not os.path.exists(prob_path):
        raise FileNotFoundError(
            f"{prob_path} not found. Run 09c_deal_rolldown_walkforward.py first "
            f"(which itself requires 05_models.py, 07_forecast_evaluation.py, "
            f"13e_walkforward_residual_pool.py, and 01d_lme_cash_cleaning.py "
            f"to have already run)."
        )

    prob_df = pd.read_parquet(prob_path)
    copper  = pd.read_parquet(cu_path)

    deal_df = compute_liquidity_buffer(prob_df, copper)

    if deal_df.empty:
        print("WARNING: Stylised deal DataFrame is empty. Check probability forecast data.")
        return

    print_summary(deal_df)

    # Save
    out_path = os.path.join(PROCESSED_DATA_DIR, "stylised_deal_results.parquet")
    deal_df.to_parquet(out_path)
    print(f"\nSaved: {out_path}")

    # Figures
    print("\nGenerating figures...")
    plot_liquidity_buffer(deal_df, FIGURES_DIR)
    plot_case_studies(deal_df, FIGURES_DIR)

    print("\nPhase 6 complete.")
    print("Pipeline fully executed.")


if __name__ == "__main__":
    main()
