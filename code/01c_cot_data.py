"""
01c_cot_data.py - Parse and clean CFTC Commitments of Traders data for copper.

Reads ALL .txt files found in the CFTC folder - any naming convention works.
Files are identified by their actual date content, not by filename.
Duplicate rows (same date + commodity code across files) are dropped automatically.

Known structural facts:
  - Three copper commodity names across history (all share CFTC codes 085691/085692):
      Pre-~2017 : "COPPER - COMMODITY EXCHANGE INC."
      ~2017     : "COPPER-GRADE #1 - COMMODITY EXCHANGE INC."
      Recent    : "COPPER- #1 - COMMODITY EXCHANGE INC."
  - COT was published bi-monthly before 1993 → gaps > 14 days pre-1993 are expected
  - Our sample starts 2006 → no gaps in the relevant period

Input:
  data/raw/CFTC - Future Only Reports/*.txt   (any number of files)

Outputs:
  data/processed/cot_copper_weekly.parquet
  data/processed/cot_copper_weekly.csv

Output columns:
  date             : Friday-aligned weekly date (W-FRI resample)
  open_interest    : total open interest (all)
  noncomm_long     : non-commercial long positions
  noncomm_short    : non-commercial short positions
  noncomm_spread   : non-commercial spreading positions
  net_speculative  : noncomm_long - noncomm_short  (KEY VARIABLE for VAR)
  net_spec_pct_oi  : net_speculative / open_interest × 100 (normalised)

Run: python code/01c_cot_data.py
     (from thesis/ root with .venv active)
"""

import os
import sys
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Constants ─────────────────────────────────────────────────────────────────

CFTC_DIR     = os.path.join(config.RAW_DATA_DIR, "CFTC - Future Only Reports")
COPPER_CODES = {"085691", "085692"}   # COMEX copper - both historical code variants
SAMPLE_START = pd.Timestamp(config.START_DATE)
SAMPLE_END   = pd.Timestamp(config.END_DATE)

# CFTC legacy format column names → our internal names
COL_MAP = {
    "Market and Exchange Names":               "commodity_name",
    "As of Date in Form YYYY-MM-DD":           "date",
    "CFTC Contract Market Code":               "cftc_code",
    "Open Interest (All)":                     "open_interest",
    "Noncommercial Positions-Long (All)":      "noncomm_long",
    "Noncommercial Positions-Short (All)":     "noncomm_short",
    "Noncommercial Positions-Spreading (All)": "noncomm_spread",
}
KEEP_COLS     = list(COL_MAP.keys())
REQUIRED_COLS = {"date", "cftc_code", "open_interest", "noncomm_long", "noncomm_short"}


# ── File loading ───────────────────────────────────────────────────────────────

def _is_cftc_file(filepath: str) -> bool:
    """
    Peek at the first line of a file to verify it has the expected CFTC header.
    Returns True if the file looks like a CFTC legacy COT report.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
        return (
            "Market and Exchange Names" in header
            and "As of Date in Form YYYY-MM-DD" in header
            and "Noncommercial" in header
        )
    except Exception:
        return False


def load_cftc_file(filepath: str) -> pd.DataFrame | None:
    """
    Load one CFTC legacy CSV file.

    Returns a cleaned DataFrame with renamed columns, or None if the file
    does not match the expected CFTC format (so non-CFTC .txt files in the
    folder are silently skipped).
    """
    if not _is_cftc_file(filepath):
        return None

    try:
        df = pd.read_csv(
            filepath,
            usecols=lambda c: c in KEEP_COLS,
            dtype=str,          # read as string - numeric columns contain leading spaces
            low_memory=False,
        )
    except Exception as exc:
        print(f"  [ERROR] Could not read {os.path.basename(filepath)}: {exc}")
        return None

    df = df.rename(columns=COL_MAP)

    # Validate that required columns were loaded
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"  [ERROR] {os.path.basename(filepath)} missing columns: {missing}")
        print(f"          Columns found: {list(df.columns)}")
        return None

    # Parse date - CFTC format is YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"].str.strip(), format="%Y-%m-%d", errors="coerce")
    n_bad_dates = df["date"].isna().sum()
    if n_bad_dates > 0:
        df = df.dropna(subset=["date"])

    # Strip whitespace from commodity code (CFTC pads with spaces)
    df["cftc_code"] = df["cftc_code"].str.strip()

    # Convert numeric columns (CFTC uses leading spaces, commas in some versions)
    for col in ["open_interest", "noncomm_long", "noncomm_short", "noncomm_spread"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].str.replace(",", "").str.strip(), errors="coerce"
            )

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("01c_cot_data.py - CFTC copper COT data cleaning")
    print("=" * 60)

    # ── 1. Discover all .txt files - any naming convention ────────────────────
    if not os.path.isdir(CFTC_DIR):
        print(f"[ERROR] CFTC folder not found: {CFTC_DIR}")
        print("  Create the folder and place CFTC .txt files inside.")
        sys.exit(1)

    all_txt = sorted(glob.glob(os.path.join(CFTC_DIR, "*.txt")))

    if not all_txt:
        print(f"[ERROR] No .txt files found in: {CFTC_DIR}")
        print("  Download from cftc.gov → Historical Compressed → Futures Only Reports")
        sys.exit(1)

    print(f"\nDiscovered {len(all_txt)} .txt file(s) in {CFTC_DIR}:")

    # ── 2. Load each file ─────────────────────────────────────────────────────
    frames = []
    skipped = []
    for fp in all_txt:
        fname = os.path.basename(fp)
        df = load_cftc_file(fp)

        if df is None:
            skipped.append(fname)
            print(f"  [SKIP] {fname:<40s}  not a CFTC legacy format file")
            continue

        min_d = df["date"].min().date() if len(df) else "empty"
        max_d = df["date"].max().date() if len(df) else "empty"
        print(f"  ✓ {fname:<40s}  {len(df):>7,d} rows   {min_d} → {max_d}")
        frames.append(df)

    if not frames:
        print("\n[ERROR] No valid CFTC files could be loaded.")
        sys.exit(1)

    if skipped:
        print(f"\n  Skipped {len(skipped)} non-CFTC file(s): {skipped}")

    # ── 3. Concatenate and deduplicate ────────────────────────────────────────
    print("\nConcatenating all files ...")
    all_data = pd.concat(frames, ignore_index=True)
    n_before = len(all_data)

    # Deduplicate on date + commodity code - handles:
    #   (a) exact file duplicates (e.g. two copies of the same year)
    #   (b) overlap between the combined 1986-2016 file and any annual file that
    #       also covers part of 2016
    all_data = all_data.drop_duplicates(subset=["date", "cftc_code"])
    n_after  = len(all_data)
    n_dupes  = n_before - n_after

    print(f"  Total rows before dedup : {n_before:>9,d}")
    print(f"  Total rows after  dedup : {n_after:>9,d}  ({n_dupes:,d} duplicate rows removed)")

    # ── 4. Filter for copper ──────────────────────────────────────────────────
    print(f"\nFiltering for copper (CFTC codes: {sorted(COPPER_CODES)}) ...")
    copper = all_data[all_data["cftc_code"].isin(COPPER_CODES)].copy()

    if len(copper) == 0:
        print("[ERROR] No copper rows found.")
        print(f"  Codes present in data: {sorted(all_data['cftc_code'].unique()[:20])}")
        print(f"  Expected: {sorted(COPPER_CODES)}")
        sys.exit(1)

    names_found = copper["commodity_name"].str.strip().unique()
    print(f"  Copper rows : {len(copper):,d}")
    print(f"  Name variants found ({len(names_found)}):")
    for nm in sorted(names_found):
        n_dates = copper[copper["commodity_name"].str.strip() == nm]["date"].nunique()
        print(f"    '{nm}'  ({n_dates:,d} dates)")

    # ── 5. Sort and compute net speculative position ──────────────────────────
    copper = copper.sort_values("date").reset_index(drop=True)

    copper["net_speculative"] = copper["noncomm_long"] - copper["noncomm_short"]
    copper["net_spec_pct_oi"] = (
        copper["net_speculative"] / copper["open_interest"] * 100
    ).round(2)

    # ── 6. Coverage diagnostics ───────────────────────────────────────────────
    full_min = copper["date"].min().date()
    full_max = copper["date"].max().date()
    print(f"\nFull copper COT coverage: {full_min} → {full_max}  ({len(copper):,d} obs)")

    # Restrict gap check to our sample period - pre-1993 bi-monthly gaps are expected
    copper_sample_check = copper[copper["date"] >= SAMPLE_START].copy()
    diffs_sample = copper_sample_check["date"].diff().dt.days.dropna()
    gaps_sample  = diffs_sample[diffs_sample > 14]

    if len(gaps_sample) > 0:
        print(f"\n  [WARNING] {len(gaps_sample)} gap(s) > 14 days in sample period:")
        for idx in gaps_sample.index:
            gap_date = copper_sample_check.loc[idx, "date"].date()
            gap_days = int(gaps_sample[idx])
            print(f"    {gap_date}  gap = {gap_days} days")
    else:
        print(f"  ✓ No gaps > 14 days in sample period ({config.START_DATE} → {config.END_DATE})")

    # Pre-sample gaps (expected - bi-monthly reporting before 1993)
    diffs_pre = copper[copper["date"] < SAMPLE_START]["date"].diff().dt.days.dropna()
    n_pre_gaps = (diffs_pre > 14).sum()
    if n_pre_gaps > 0:
        print(f"  Note: {n_pre_gaps} gaps > 14 days pre-{config.START_DATE} "
              f"(expected - CFTC published bi-monthly before 1993)")

    # ── 7. Restrict to sample period and resample to W-FRI ───────────────────
    copper_sample = copper[
        (copper["date"] >= SAMPLE_START) & (copper["date"] <= SAMPLE_END)
    ].copy()

    print(f"\nSample period ({config.START_DATE} → {config.END_DATE}):")
    print(f"  Observations before resample: {len(copper_sample):,d}")

    copper_sample = copper_sample.set_index("date")
    numeric_cols  = ["open_interest", "noncomm_long", "noncomm_short",
                     "noncomm_spread", "net_speculative", "net_spec_pct_oi"]

    weekly = (
        copper_sample[numeric_cols]
        .resample("W-FRI")
        .last()          # most recent COT report within each Friday-ending week
        .ffill(limit=1)  # forward-fill at most 1 missing week (holidays only)
    )

    n_nan = weekly["net_speculative"].isna().sum()
    print(f"  Observations after W-FRI resample: {len(weekly):,d}  ({n_nan} NaN)")

    if n_nan > 0:
        nan_dates = weekly[weekly["net_speculative"].isna()].index
        print(f"  NaN dates: {[d.date() for d in nan_dates[:10]]}")

    # ── 8. Summary statistics ─────────────────────────────────────────────────
    ns = weekly["net_speculative"].dropna()
    print(f"\nNet speculative position summary:")
    print(f"  Mean   : {ns.mean():>10,.0f} contracts")
    print(f"  Std    : {ns.std():>10,.0f} contracts")
    print(f"  Min    : {ns.min():>10,.0f}  on {ns.idxmin().date()}")
    print(f"  Max    : {ns.max():>10,.0f}  on {ns.idxmax().date()}")
    print(f"  Net short (< 0): {(ns < 0).sum():>3d} weeks  ({(ns < 0).mean()*100:.1f}%)")

    # ── 9. Save ───────────────────────────────────────────────────────────────
    parquet_path = os.path.join(config.PROCESSED_DATA_DIR, "cot_copper_weekly.parquet")
    csv_path     = os.path.join(config.PROCESSED_DATA_DIR, "cot_copper_weekly.csv")

    weekly.to_parquet(parquet_path)
    weekly.to_csv(csv_path)

    print(f"\n  Saved: {parquet_path}  shape: {weekly.shape}")
    print(f"  Saved: {csv_path}")

    # ── 10. FRED coverage audit ───────────────────────────────────────────────
    _check_fred_files()

    print("\n01c_cot_data.py complete.")
    print("  Action required: download fred_wti.csv (DCOILWTICO from FRED)")
    print("  Then re-run: python code/01_data_cleaning.py")


def _check_fred_files() -> None:
    """
    Audit all FRED CSVs in data/raw/.
    Checks weekly-resampled NaN (not raw daily NaN) to avoid false positives
    from weekends and US holidays in daily FRED data.
    """
    print("\n" + "=" * 60)
    print("FRED FILE AUDIT  (weekly-resampled NaN - not raw daily)")
    print("=" * 60)

    raw_dir    = config.RAW_DATA_DIR
    fred_files = sorted(
        f for f in os.listdir(raw_dir)
        if f.startswith("fred_") and f.endswith(".csv")
    )

    # Build expected set from config
    expected_files = set(config.RAW_FRED_FILES.values())

    real_issues = []

    for fname in fred_files:
        path = os.path.join(raw_dir, fname)
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            real_issues.append(f"{fname}: read error - {exc}")
            print(f"  ✗ {fname:<24s}  READ ERROR: {exc}")
            continue

        # Identify date and value columns
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col is None:
            real_issues.append(f"{fname}: no date column")
            print(f"  ✗ {fname:<24s}  no date column")
            continue

        val_col = next((c for c in df.columns if c != date_col), None)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df[val_col]  = pd.to_numeric(df[val_col], errors="coerce")

        raw_min = df[date_col].min().date()
        raw_max = df[date_col].max().date()

        # Resample to weekly and count NaN in sample period
        df_s = df.set_index(date_col)[val_col]
        df_s = df_s[
            (df_s.index >= SAMPLE_START) & (df_s.index <= SAMPLE_END)
        ]

        # Remove weekends before resampling (same logic as 01_data_cleaning.py)
        df_s = df_s[df_s.index.dayofweek < 5]
        weekly = df_s.resample("W-FRI").last().ffill(limit=1)

        n_weekly     = len(weekly)
        n_nan_weekly = weekly.isna().sum()

        # Real issue: starts significantly after sample start (>30 days)
        start_lag_days = (df[date_col].min() - SAMPLE_START).days
        status = "✓"

        if start_lag_days > 30:
            msg = (f"starts {raw_min}, which is {abs(start_lag_days)} days "
                   f"after sample start {config.START_DATE}")
            real_issues.append(f"{fname}: {msg}")
            status = "⚠"

        if n_nan_weekly > 5:
            msg = f"{n_nan_weekly} NaN weeks after W-FRI resample (check for data gaps)"
            real_issues.append(f"{fname}: {msg}")
            status = "⚠"

        in_config = "  [in config]" if fname in expected_files else ""
        print(f"  {status} {fname:<24s}  {raw_min} → {raw_max}  "
              f"weekly={n_weekly}  NaN_weeks={n_nan_weekly}{in_config}")

    # Check for missing expected files
    present = set(fred_files)
    missing = expected_files - present
    for m in sorted(missing):
        real_issues.append(f"MISSING from data/raw/: {m}  (required by config.RAW_FRED_FILES)")
        print(f"  ✗ {m:<24s}  MISSING - download from FRED")

    if real_issues:
        print(f"\n  [{len(real_issues)} real issue(s)]:")
        for iss in real_issues:
            print(f"    → {iss}")
    else:
        print("\n  ✓ All FRED files present, full sample coverage, no NaN gaps.")


if __name__ == "__main__":
    main()
