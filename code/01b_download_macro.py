"""
01b_download_macro.py - Auto-download all FRED macro variables from config.

Downloads every series listed in config.RAW_FRED_FILES directly from FRED
using the public bulk-download endpoint (no API key required). Saves raw CSVs
to data/raw/ exactly as expected by 01_data_cleaning.py.

After running this script:
  python code/01_data_cleaning.py   ← rebuilds macro.parquet with new variables

Bloomberg data (inventory, cancelled warrants) must be added manually:
  data/raw/lme_inventory.xlsx           ← Bloomberg LCUASTOT export
  data/raw/lme_cancelled_warrants.xlsx  ← Bloomberg LCUCANCL export
Then re-run 01_data_cleaning.py again.

Optional extras (uncomment in config.RAW_FRED_FILES after VIF check in Phase 3):
  fred_cnyusd.csv  ← DEXCHUS  - Chinese yuan/USD (China demand proxy)
  fred_us2y.csv    ← DGS2     - US 2-year yield

CFTC COT data (COMEX copper speculative positioning):
  Manual download required - see instructions printed at end of this script.

Run: python code/01b_download_macro.py
     (from thesis/ root with .venv active)
     Requires internet access - run from personal machine if BRED blocks FRED.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={ticker}"
HEADERS  = {"User-Agent": "Mozilla/5.0 (research/thesis download)"}


# ── Download helpers ───────────────────────────────────────────────────────────

def download_fred_series(ticker: str, dest_path: str, timeout: int = 30) -> bool:
    """
    Download a single FRED series CSV to dest_path.
    Returns True on success, False on failure.
    """
    url = FRED_URL.format(ticker=ticker)

    if not HAS_REQUESTS:
        print(f"  [SKIP] requests not installed - cannot auto-download {ticker}")
        print(f"         Manual URL: {url}")
        return False

    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()

        if "observation_date" not in resp.text[:100]:
            print(f"  [ERROR] Unexpected response for {ticker} - not a FRED CSV")
            print(f"         First 200 chars: {resp.text[:200]}")
            return False

        with open(dest_path, "w", encoding="utf-8") as fh:
            fh.write(resp.text)

        # Count rows for quick sanity check
        n_rows = resp.text.count("\n") - 1
        print(f"  ✓  {ticker:12s}  {n_rows:6,d} rows  →  {os.path.basename(dest_path)}")
        return True

    except requests.exceptions.Timeout:
        print(f"  [TIMEOUT] {ticker} - server did not respond within {timeout}s")
        print(f"           Manual URL: {url}")
        return False

    except requests.exceptions.RequestException as exc:
        print(f"  [ERROR] {ticker}: {exc}")
        print(f"          Manual URL: {url}")
        return False


def already_fresh(path: str, max_age_days: int = 7) -> bool:
    """Return True if file exists and is less than max_age_days old."""
    if not os.path.exists(path):
        return False
    age_days = (time.time() - os.path.getmtime(path)) / 86_400
    return age_days < max_age_days


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    raw_dir = config.RAW_DATA_DIR
    os.makedirs(raw_dir, exist_ok=True)

    print("=" * 60)
    print("01b_download_macro.py - FRED macro variable download")
    print("=" * 60)
    print(f"Destination:  {os.path.abspath(raw_dir)}")
    print(f"Series count: {len(config.RAW_FRED_FILES)} (from config.RAW_FRED_FILES)")
    print()

    if not HAS_REQUESTS:
        print("[WARNING] 'requests' library not found in venv.")
        print("          Install: pip install requests")
        print("          OR download each file manually using the URLs below.\n")

    # ── Build the full download list: RAW_FRED_FILES + optional extras ─────────
    # Include optional extras (CNYUSD, US2Y) even if commented out in config -
    # having the raw file ready costs nothing and avoids a re-download later.
    OPTIONAL_EXTRAS = {
        "CNYUSD": "fred_cnyusd.csv",
        "US2Y":   "fred_us2y.csv",
    }
    all_series = dict(config.RAW_FRED_FILES)
    for name, fname in OPTIONAL_EXTRAS.items():
        if name not in all_series:
            all_series[name] = fname

    # Resolve ticker via MACRO_TICKERS_FRED; fall back to key name
    ticker_map = {
        **config.MACRO_TICKERS_FRED,
        "CNYUSD": "DEXCHUS",
        "US2Y":   "DGS2",
    }

    success, skipped, failed = [], [], []

    for var_name, filename in all_series.items():
        ticker = ticker_map.get(var_name, var_name)
        dest   = os.path.join(raw_dir, filename)

        if already_fresh(dest):
            size_kb = os.path.getsize(dest) / 1024
            print(f"  ↩  {ticker:12s}  already fresh ({size_kb:.0f} KB)  →  {filename}")
            skipped.append(var_name)
            continue

        ok = download_fred_series(ticker, dest)
        (success if ok else failed).append(var_name)
        time.sleep(0.5)   # be polite to FRED servers

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Downloaded: {len(success)}")
    print(f"  Skipped (fresh): {len(skipped)}")
    print(f"  Failed:     {len(failed)}")

    if failed:
        print(f"\n  Failed series: {failed}")
        print("  Download manually from:")
        for var in failed:
            tk = ticker_map.get(var, var)
            fn = all_series.get(var, "")
            url = FRED_URL.format(ticker=tk)
            print(f"    {tk:12s}  →  {url}")
            print(f"               Save as: data/raw/{fn}")

    print()
    print("  Next steps:")
    print("    python code/01c_cot_data.py       ← builds cot_copper_weekly.parquet")
    print("    python code/01_data_cleaning.py   ← rebuilds macro.parquet")
    print()

    # ── CFTC COT data instructions ────────────────────────────────────────────
    print("=" * 60)
    print("  CFTC COT DATA (optional - COMEX copper speculative positioning)")
    print("=" * 60)
    print("""
  Source: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm

  Steps:
  1. Click "Historical Compressed" in the left menu
  2. Under "Futures Only Reports" → "All Years Combined" → download the zip
  3. Unzip - you get a large text file (CSV format)
  4. Save the unzipped file as:  data/raw/cot_copper_historical.csv
  5. Run:  python code/01c_cot_data.py   (parses copper data, saves to raw)

  The relevant commodity name in the file is:  COPPER- #1  (COMEX exchange)
  Columns used: non-commercial long and short positions → net speculative

  This variable is OPTIONAL. Include in the VAR only if:
    (a) Granger-causes β₁ at 5% level (Phase 3, 04_granger.py)
    (b) VIF < 8 against other macro variables
  """)

    # ── Bloomberg reminder ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  BLOOMBERG DATA (add when access returns in ~9 days)")
    print("=" * 60)
    print("""
  Required exports (same sheet-per-year structure as copper futures):

  1. LME on-warrant stocks (inventory):
       Ticker:   LCUASTOT Index
       Field:    PX_LAST
       Freq:     Daily, 2006-01-01 to today
       Save as:  data/raw/lme_inventory.xlsx

  2. LME cancelled warrants:
       Ticker:   LCUCANCL Index
       Field:    PX_LAST
       Freq:     Daily, 2006-01-01 to today
       Save as:  data/raw/lme_cancelled_warrants.xlsx

  After adding both files:
    python code/01_data_cleaning.py     ← adds inventory to macro.parquet
    python code/01c_cancelled_warrants.py ← computes cancelled ratio (to build)
  """)
    print("=" * 60)


if __name__ == "__main__":
    main()
