"""
01d_lme_cash_cleaning.py -- Clean LME Copper Cash-Settlement data.

WHY THIS EXISTS: the Bloomberg extraction (Appendix II) pulls only the
generic rolling contracts LP1 Comdty through LP24 Comdty, not LME Cash
(Bloomberg ticker LMCADY Comdty). Without it, the stylised deal
(09a_deal_rolldown_probability.py) would have no directly-observed
zero-maturity price to compare against at the deal's own closing date, and
would need LP1 as an approximate stand-in instead -- itself a rolling
series that does not represent the same delivery date as the deal's own
contract at the exact date it is read (see 09a's module docstring).

This script processes a free, public alternative source -- daily LME Copper
Cash-Settlement and 3-month prices published by westmetall.com, a standard
reference site for LME base metals history -- covering 2008 through the
current date, sourced and saved to data/raw/LME_Copper_Cash_Settlement.xlsx.
This is a genuine, directly observed zero-maturity (T+2 spot) price, unlike
LP1, and lets 09a's realized-side comparison be exact rather than
approximated.

SCOPE: this fixes the stylised deal's own realized-price comparison only
(Chapter 6). It does NOT add Cash as a point in the core Nelson-Siegel
cross-sectional fit (Section 3.2), which would ripple through the entire
factor-extraction and forecasting pipeline (Chapters 3-6) and require
re-verifying every downstream empirical result. That remains documented as
future work (Section 3.9), not undertaken here.

Data quirks handled:
  - Each yearly sheet (2008-2026) has its own header row, and several
    sheets have that header row repeated mid-sheet (a pagination artefact
    from the source website); both are dropped.
  - Dates are given as "DD. Month YYYY" (e.g. "25. August 2026").
  - Prices use "," as a thousands separator in some cells; commas are
    stripped before numeric conversion.

Input : data/raw/LME_Copper_Cash_Settlement.xlsx  (sheets "2008".."2026",
                                                     plus a "Source" sheet,
                                                     not read here)
Output: data/processed/lme_copper_cash.parquet     (weekly Friday close,
                                                     columns: cash, m03,
                                                     matching the main
                                                     pipeline's convention)

Run: python code/01d_lme_cash_cleaning.py   (before 09a_deal_rolldown_probability.py)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

RAW_FILE = os.path.join(RAW_DATA_DIR, "LME_Copper_Cash_Settlement.xlsx")
YEAR_SHEETS = [str(y) for y in range(2008, 2027)]


def _clean_price(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def _load_sheet(xl: pd.ExcelFile, year: str) -> pd.DataFrame:
    df = xl.parse(year)
    if "date" not in df.columns:
        # Flagged, not silently worked around: every other year has a
        # "date" column; if this one doesn't, something upstream in the
        # source file changed and needs a human to look at it, not a
        # guessed column-shift fix.
        print(f"  [FLAG] Sheet {year} has no 'date' column (columns: {list(df.columns)}) "
              f"-- skipping this sheet rather than guessing its structure.")
        return pd.DataFrame(columns=["date", "cash", "m03"])

    df = df[df["date"].astype(str).str.strip().str.lower() != "date"].copy()
    df["date"] = pd.to_datetime(
        df["date"].astype(str).str.replace(".", "", regex=False).str.strip(),
        format="%d %B %Y", errors="coerce",
    )
    n_bad_dates = df["date"].isna().sum()
    if n_bad_dates:
        print(f"  [FLAG] Sheet {year}: {n_bad_dates} row(s) with unparseable dates, dropped.")

    missing_cols = {"LME Copper Cash-Settlement", "LME Copper 3-month"} - set(df.columns)
    if missing_cols:
        print(f"  [FLAG] Sheet {year} is missing expected column(s) {missing_cols} "
              f"(columns present: {list(df.columns)}) -- skipping this sheet.")
        return pd.DataFrame(columns=["date", "cash", "m03"])

    df["cash"] = _clean_price(df["LME Copper Cash-Settlement"])
    df["m03"] = _clean_price(df["LME Copper 3-month"])
    df = df.dropna(subset=["date"])
    return df[["date", "cash", "m03"]]


def main():
    print("=" * 78)
    print("Cleaning LME Copper Cash-Settlement data (westmetall.com, 2008-2026)")
    print("=" * 78)

    if not os.path.exists(RAW_FILE):
        print(f"[Raw file not found: {RAW_FILE} -- skipping]")
        return

    xl = pd.ExcelFile(RAW_FILE)
    missing_sheets = [y for y in YEAR_SHEETS if y not in xl.sheet_names]
    if missing_sheets:
        print(f"[FLAG] Expected yearly sheets not found in workbook: {missing_sheets}")

    frames = [_load_sheet(xl, y) for y in YEAR_SHEETS if y in xl.sheet_names]
    frames = [f for f in frames if not f.empty]
    df = pd.concat(frames, ignore_index=True).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    n_dupe = df.index.duplicated(keep="last").sum()
    if n_dupe:
        print(f"  [FLAG] {n_dupe} duplicate date(s) across sheets, kept the last occurrence.")
    df = df[~df.index.duplicated(keep="last")]

    print(f"\nRaw daily observations: {len(df)}  ({df.index.min().date()} to {df.index.max().date()})")
    years_present = sorted(df.index.year.unique())
    years_expected = list(range(2008, pd.Timestamp.today().year + 1))
    years_absent = [y for y in years_expected if y not in years_present]
    if years_absent:
        print(f"  [FLAG] No data at all for year(s): {years_absent}")

    weekly = df[df.index.dayofweek < 5].resample("W-FRI").last()
    n_missing = weekly["cash"].isna().sum()
    print(f"Weekly (Friday close) observations: {len(weekly)}, {n_missing} weeks missing Cash")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DATA_DIR, "lme_copper_cash.parquet")
    weekly.to_parquet(out_path)
    print(f"\nSaved: {out_path}")
    print("\nNext: rerun code/09a_deal_rolldown_probability.py, which reads this "
          "file for the deal's realized/exit price and threshold volatility.")


if __name__ == "__main__":
    main()
