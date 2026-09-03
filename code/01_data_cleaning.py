# =============================================================================
# 01_data_cleaning.py
# =============================================================================
# PURPOSE:
#   Load raw Bloomberg futures data and FRED macro variables,
#   clean and align all series to weekly Friday close,
#   save to data/processed/ as parquet files for downstream analysis.
#
# INPUTS (place in data/raw/):
#   - bloomberg_futures.xlsx  : Bloomberg generic futures export
#                               Structure: one sheet per year (e.g. "2006"..."2026")
#                               Row 0: Bloomberg tickers (e.g. XB1 Comdty)
#                               Row 1: Maturity labels (e.g. "1 Month")
#                               Row 2+: Daily price data, col 0 = date
#   - bloomberg_inventory.xlsx: LME on-warrant inventory (LCUASTOT)
#                               Same sheet-per-year structure, one column of data
#   - fred_dxy.csv            : US Dollar Index (DTWEXBGS) from FRED
#   - fred_vix.csv            : CBOE VIX (VIXCLS) from FRED
#   - fred_us10y.csv          : US 10Y Treasury yield (DGS10) from FRED
#
# OUTPUTS (saved to data/processed/):
#   - curves.parquet          : Weekly futures curve data [m01...m15]
#   - macro.parquet           : Weekly macro variables [DXY, VIX, US10Y, inventory]
#   - combined.parquet        : Merged curves + macro, fully aligned
#
# SWITCHING FROM GASOLINE TO COPPER:
#   Change RAW_FUTURES_FILE in config.py - nothing else needs to change.
#   The script detects column count and maturity labels automatically.
#
# USAGE:
#   python code/01_data_cleaning.py
#
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures

# Add project root to path so config.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# =============================================================================
# SECTION 1 - Bloomberg Futures Data
# =============================================================================

def load_bloomberg_futures(filepath: str) -> pd.DataFrame:
    """
    Load Bloomberg generic futures data from the standard export format:
    - One sheet per year
    - Row 0: Bloomberg tickers
    - Row 1: Maturity labels ("1 Month", "2 Month", etc.)
    - Row 2+: Daily data, first column = date

    Returns:
        DataFrame with DatetimeIndex and columns m01, m02, ..., mNN
        All values numeric (Bloomberg error strings converted to NaN)
    """
    print(f"\n  Loading Bloomberg futures: {os.path.basename(filepath)}")

    xl = pd.ExcelFile(filepath)
    print(f"  Found {len(xl.sheet_names)} sheets: {xl.sheet_names[:3]}...{xl.sheet_names[-3:]}")

    frames = []
    for sheet in xl.sheet_names:
        try:
            # skiprows=1 skips the ticker row (row 0)
            # Row 1 becomes the header (maturity labels)
            df = pd.read_excel(xl, sheet_name=sheet, skiprows=1, header=0)

            # First column is always dates - rename to 'date'
            df = df.rename(columns={df.columns[0]: 'date'})

            # Parse dates - Bloomberg exports as datetime strings
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # Drop rows where date parsing failed (e.g. label rows)
            df = df.dropna(subset=['date'])

            # Rename maturity columns to standard m01, m02, ... format
            n_maturities = len(df.columns) - 1
            maturity_cols = [f'm{i:02d}' for i in range(1, n_maturities + 1)]
            df.columns = ['date'] + maturity_cols

            # Convert all price columns to numeric
            # Bloomberg sometimes inserts error strings like "#N/A Requesting Data..."
            for col in maturity_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            frames.append(df)

        except Exception as e:
            print(f"  [WARNING] Could not load sheet '{sheet}': {e}")
            continue

    if not frames:
        raise ValueError(f"No valid sheets found in {filepath}")

    # Concatenate all years
    full = pd.concat(frames, ignore_index=True)

    # Sort by date and remove duplicates (keep last - most recent Bloomberg value)
    full = full.sort_values('date').drop_duplicates(subset='date', keep='last')

    # Set date as index
    full = full.set_index('date')
    full.index = pd.DatetimeIndex(full.index)

    print(f"  Raw: {len(full)} daily observations, {full.shape[1]} maturities")
    print(f"  Date range: {full.index.min().date()} → {full.index.max().date()}")

    return full


def load_bloomberg_inventory(filepath: str) -> pd.Series:
    """
    Load Bloomberg inventory data (e.g. LCUASTOT - LME copper on-warrant stocks).
    Same sheet-per-year structure as futures data.

    Returns:
        Series with DatetimeIndex named 'inventory'
    """
    print(f"\n  Loading Bloomberg inventory: {os.path.basename(filepath)}")

    xl = pd.ExcelFile(filepath)
    frames = []

    for sheet in xl.sheet_names:
        try:
            df = pd.read_excel(xl, sheet_name=sheet, skiprows=1, header=0)
            df = df.rename(columns={df.columns[0]: 'date'})
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])

            # Keep only the first data column (inventory series)
            data_col = df.columns[1]
            df = df[['date', data_col]].copy()
            df.columns = ['date', 'inventory']
            df['inventory'] = pd.to_numeric(df['inventory'], errors='coerce')

            frames.append(df)
        except Exception as e:
            print(f"  [WARNING] Sheet '{sheet}': {e}")
            continue

    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values('date').drop_duplicates(subset='date', keep='last')
    full = full.set_index('date')
    full.index = pd.DatetimeIndex(full.index)
    series = full['inventory']
    series.name = 'inventory'

    print(f"  Raw: {len(series)} daily observations")
    return series


# =============================================================================
# SECTION 2 - FRED Macro Variables
# =============================================================================

def load_fred_csv(filepath: str, colname: str) -> pd.Series:
    """
    Load a single macro variable from a FRED CSV download.
    FRED CSVs have format: DATE, VALUE
    Missing values are coded as '.' - converted to NaN.

    Args:
        filepath: path to CSV file
        colname:  name to give the resulting Series

    Returns:
        Series with DatetimeIndex
    """
    print(f"\n  Loading FRED variable '{colname}': {os.path.basename(filepath)}")

    # FRED uses 'DATE' or 'observation_date' depending on download format
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    date_col = next(c for c in df.columns if 'date' in c.lower())
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = 'date'

    # FRED uses '.' for missing values
    series = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    series.name = colname
    series.index = pd.DatetimeIndex(series.index)

    print(f"  Raw: {len(series)} daily observations, "
          f"{series.isna().sum()} NaN values")
    return series


# =============================================================================
# SECTION 2b - GPR (Caldara & Iacoviello 2022)
# =============================================================================

def load_gpr_excel(filepath: str) -> pd.Series:
    """
    Load the daily Geopolitical Risk index (Caldara & Iacoviello, 2022,
    American Economic Review) from the policyuncertainty.com Excel export.

    GPR is part of the production macro set: the multi-horizon Granger
    screen (code/04c_multi_horizon_granger.py) and a sub-sample stability
    check show a genuine, temporally stable GPR → β₁ (slope) relationship,
    significant in both the 2006–2014 and 2014–2022 halves of the training
    sample, i.e. not a Trump-era-only artefact. See main.qmd Section 3.1.

    The raw file has a non-standard transposed layout (metadata columns
    interleaved with the data columns); this parses the 'date' and 'GPRD'
    (daily GPR index) columns directly.
    """
    print(f"\n  Loading GPR index: {os.path.basename(filepath)}")
    raw = pd.read_excel(filepath)
    if "date" not in raw.columns or "GPRD" not in raw.columns:
        raise ValueError(
            f"Expected 'date' and 'GPRD' columns in {filepath}, "
            f"found: {list(raw.columns)}"
        )
    dates = pd.to_datetime(raw["date"], errors="coerce")
    vals  = pd.to_numeric(raw["GPRD"], errors="coerce")
    series = pd.Series(vals.values, index=dates, name="GPR").dropna().sort_index()
    print(f"  Raw: {len(series)} daily observations, "
          f"{series.index.min().date()} → {series.index.max().date()}")
    return series


# =============================================================================
# SECTION 3 - Cleaning and Resampling
# =============================================================================

def remove_weekends_and_resample(df: pd.DataFrame,
                                  freq: str = 'W-FRI') -> pd.DataFrame:
    """
    Remove weekend observations and resample to weekly frequency.

    Bloomberg copies Friday's price to Saturday and Sunday - these are not
    real market observations and must be removed before resampling.

    Steps:
        1. Keep only Monday–Friday (dayofweek 0–4)
        2. Resample to weekly, taking the Friday close (last business day of week)
        3. Drop weeks with no valid observations

    Args:
        df:   DataFrame with DatetimeIndex (daily frequency)
        freq: resample frequency (default 'W-FRI' = weekly Friday close)

    Returns:
        DataFrame resampled to weekly frequency
    """
    # Step 1: Remove weekends
    df_weekdays = df[df.index.dayofweek < 5].copy()

    weekend_count = len(df) - len(df_weekdays)
    if weekend_count > 0:
        print(f"  Removed {weekend_count} weekend observations")

    # Step 2: Resample to weekly Friday close
    df_weekly = df_weekdays.resample(freq).last()

    # Step 3: Drop fully empty weeks
    df_weekly = df_weekly.dropna(how='all')

    return df_weekly


def filter_sample_period(df: pd.DataFrame) -> pd.DataFrame:
    """Apply START_DATE and END_DATE from config.py."""
    return df.loc[config.START_DATE:config.END_DATE]


def apply_inventory_lag(series: pd.Series, lag: int) -> pd.Series:
    """
    Apply a lag to the inventory series to mitigate simultaneity bias.

    The Theory of Storage predicts bidirectional causality between inventory
    levels and the futures curve slope (beta1). Using a lagged inventory
    value reduces this simultaneity when inventory is used as a VAR input.

    Args:
        series: inventory time series (weekly)
        lag:    number of periods to lag (from config.INVENTORY_LAG)

    Returns:
        Lagged series with same index
    """
    lagged = series.shift(lag)
    lagged.name = f'inventory_lag{lag}'
    return lagged


# =============================================================================
# SECTION 4 - Data Coverage Report
# =============================================================================

def print_coverage_report(curves: pd.DataFrame,
                           macro: pd.DataFrame) -> None:
    """Print a detailed coverage report to the console."""

    print("\n" + "=" * 60)
    print("DATA COVERAGE REPORT")
    print("=" * 60)

    print(f"\nSample period:  {curves.index.min().date()} → "
          f"{curves.index.max().date()}")
    print(f"Weekly obs:     {len(curves)}")

    # Futures coverage per maturity
    print("\nFutures curve coverage (% of weeks with valid price):")
    maturities = [c for c in curves.columns if c.startswith('m')]
    for col in maturities:
        pct = curves[col].notna().mean() * 100
        bar = '█' * int(pct / 5)
        print(f"  {col}: {pct:5.1f}%  {bar}")

    # Macro coverage
    print("\nMacro variable coverage (% of weeks with valid value):")
    for col in macro.columns:
        pct = macro[col].notna().mean() * 100
        print(f"  {col}: {pct:5.1f}%")

    # Flag dates with many missing maturities simultaneously
    mat_cols = [c for c in curves.columns if c.startswith('m')]
    missing_count = curves[mat_cols].isna().sum(axis=1)
    bad_dates = missing_count[missing_count > 3]
    if len(bad_dates) > 0:
        print(f"\n[WARNING] {len(bad_dates)} weeks with >3 missing maturities:")
        print(f"  First 5: {list(bad_dates.head().index.date)}")
    else:
        print("\n✓ No weeks with >3 simultaneously missing maturities")

    print("=" * 60)


# =============================================================================
# SECTION 5 - Diagnostic Figures
# =============================================================================

def save_diagnostic_figures(curves: pd.DataFrame,
                              macro: pd.DataFrame) -> None:
    """
    Save diagnostic figures to thesis/figures/:
        fig_data_coverage.png   - NaN heatmap by maturity and year
        fig_spot_price.png      - 1M futures price over full sample
        fig_macro_variables.png - All macro variables on 4 panels
    """
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    mat_cols = [c for c in curves.columns if c.startswith('m')]

    # ----------------------------------------------------------------
    # Figure 1: Data coverage heatmap
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    # Build year × maturity coverage matrix
    curves_copy = curves[mat_cols].copy()
    curves_copy['year'] = curves_copy.index.year
    coverage = curves_copy.groupby('year')[mat_cols].apply(
        lambda x: x.notna().mean() * 100
    )

    im = ax.imshow(coverage.T, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=100, interpolation='none')
    ax.set_xticks(range(len(coverage.index)))
    ax.set_xticklabels(coverage.index, rotation=45, fontsize=9)
    ax.set_yticks(range(len(mat_cols)))
    ax.set_yticklabels([f'{i+1}M' for i in range(len(mat_cols))], fontsize=9)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Maturity', fontsize=10)
    ax.set_title('Data Coverage by Maturity and Year (% of weeks with valid price)',
                 fontsize=11)
    plt.colorbar(im, ax=ax, label='Coverage (%)')
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, 'fig_data_coverage.png')
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ----------------------------------------------------------------
    # Figure 2: Front month (1M) price history
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(curves.index, curves['m01'], linewidth=1.0,
            color='#2c5f9e', label='1M futures')
    ax.set_title('Front Month (1M) Futures Price - Full Sample', fontsize=11)
    ax.set_ylabel('Price', fontsize=10)
    ax.set_xlabel('')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, 'fig_spot_price.png')
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ----------------------------------------------------------------
    # Figure 3: Macro variables
    # ----------------------------------------------------------------
    macro_cols = [c for c in macro.columns if c != f'inventory_lag{config.INVENTORY_LAG}']
    n = len(macro_cols)

    # Skip figure if no macro data loaded yet
    if n == 0:
        print("  Skipping macro figure - no macro data loaded yet.")
        print("  Add FRED CSV files to data/raw/ and rerun to generate this figure.")
        return
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = {'DXY': '#8B4513', 'VIX': '#DC143C',
              'US10Y': '#2E8B57', 'inventory': '#4169E1'}

    for ax, col in zip(axes, macro_cols):
        color = colors.get(col, '#555555')
        ax.plot(macro.index, macro[col], linewidth=0.8,
                color=color, label=col)
        ax.set_ylabel(col, fontsize=10)
        ax.legend(fontsize=9, loc='upper right')

    axes[-1].set_xlabel('Date', fontsize=10)
    fig.suptitle('Macro Variables - Weekly Series', fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, 'fig_macro_variables.png')
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# SECTION 5b - Multi-Lookback Macro Change Features
# =============================================================================

def compute_macro_changes(macro_df: pd.DataFrame,
                          windows: list[int],
                          zscore_window: int = 52) -> pd.DataFrame:
    """
    Compute multi-lookback change features for all macro level variables.

    Motivation (DRA 2006 discrepancy):
      Raw macro LEVELS are often non-stationary and may exhibit no contemporaneous
      Granger causality with NS factors because their impact on the curve has an
      unknown lag and an unknown integration horizon. A 4-week change in DXY
      captures cumulative USD strengthening; a 13-week change captures a
      structural shift. Rolling z-scores capture unusual deviation from the
      trailing year's mean - a normalised regime signal.

    For each level variable X (e.g. DXY, VIX, T10Y2Y, inventory):
      X_d{w}W  = X_t - X_{t-w}           (w-period price change)
      X_z{Z}W  = (X_t - MA_{Z}(X)) / σ_{Z}(X)  (rolling z-score, training-blind)

    Parameters
    ----------
    macro_df     : DataFrame of weekly raw levels (from macro.parquet)
    windows      : list of lookback windows in weeks, e.g. [1, 4, 13]
    zscore_window: rolling window for z-score computation (weeks), default 52

    Returns
    -------
    DataFrame with the same index as macro_df, containing only the new change
    and z-score columns (NOT the raw levels - keep those in macro.parquet).
    """
    # Only compute changes for level variables. Lagged columns (e.g.
    # inventory_lag2) are excluded by default to avoid redundant clutter,
    # EXCEPT the ones in LAGGED_COLS_TO_DIFFERENCE: the production VAR needs
    # a properly-lagged change variable (inventory_lag2_d4W) rather than a
    # change computed from raw, unlagged inventory: differencing the RAW
    # series does not honour the
    # INVENTORY_LAG=2 anti-simultaneity safeguard (config.py) the two-week
    # lag exists for. Verified empirically - see main.qmd Section 3.1: the
    # raw-inventory change shows Granger p=0.040 to beta1, which drops to a
    # non-significant p=0.123 once the lag is correctly applied, consistent
    # with the raw version partly reflecting the contemporaneous feedback the
    # lag is meant to remove.
    LAGGED_COLS_TO_DIFFERENCE = {"inventory_lag2"}
    level_cols = [c for c in macro_df.columns
                  if not c.endswith(('_d1W', '_d4W', '_d13W', '_z52'))
                  and ('lag' not in c or c in LAGGED_COLS_TO_DIFFERENCE)]

    changes = pd.DataFrame(index=macro_df.index)

    for col in level_cols:
        s = macro_df[col]
        for w in windows:
            changes[f"{col}_d{w}W"] = s.diff(w)
        # Rolling z-score: training-blind (uses all available history ≤ t)
        roll_mean = s.rolling(zscore_window, min_periods=zscore_window // 2).mean()
        roll_std  = s.rolling(zscore_window, min_periods=zscore_window // 2).std()
        changes[f"{col}_z{zscore_window}W"] = (s - roll_mean) / roll_std.replace(0, np.nan)

    return changes


# =============================================================================
# SECTION 6 - Main Pipeline
# =============================================================================

def main():
    print("=" * 60)
    print("01_DATA_CLEANING.PY - THESIS DATA PIPELINE")
    print(f"Metal:       {config.METAL}")
    print(f"Sample:      {config.START_DATE} → {config.END_DATE}")
    print(f"Resampling:  Weekly Friday close")
    print(f"Inv. lag:    {config.INVENTORY_LAG} periods")
    print("=" * 60)

    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: Load Bloomberg futures
    # ------------------------------------------------------------------
    futures_path = os.path.join(config.RAW_DATA_DIR,
                                config.RAW_FUTURES_FILE)
    if not os.path.exists(futures_path):
        print(f"\n[ERROR] Bloomberg futures file not found: {futures_path}")
        print("  Export from Bloomberg and save to data/raw/ before running.")
        print("  Tickers: LPM1 COMDTY through LPM15 COMDTY, field PX_LAST")
        sys.exit(1)

    futures_raw = load_bloomberg_futures(futures_path)

    # Keep only maturities defined in config
    max_mat = max(config.MATURITIES_MONTHS)
    mat_cols = [f'm{i:02d}' for i in range(1, max_mat + 1)
                if f'm{i:02d}' in futures_raw.columns]
    futures_raw = futures_raw[mat_cols]

    # ------------------------------------------------------------------
    # STEP 2: Load Bloomberg inventory
    # ------------------------------------------------------------------
    inventory_series = None
    inventory_path = os.path.join(config.RAW_DATA_DIR,
                                  config.RAW_INVENTORY_FILE)
    if os.path.exists(inventory_path):
        inventory_series = load_bloomberg_inventory(inventory_path)
    else:
        print(f"\n[WARNING] Inventory file not found: {inventory_path}")
        print("  Proceeding without inventory - add file to data/raw/ later.")

    # ------------------------------------------------------------------
    # STEP 3: Load FRED macro variables
    # ------------------------------------------------------------------
    macro_series = {}

    for name, filename in config.RAW_FRED_FILES.items():
        path = os.path.join(config.RAW_DATA_DIR, filename)
        if os.path.exists(path):
            macro_series[name] = load_fred_csv(path, name)
        else:
            print(f"\n[WARNING] FRED file not found: {path}")
            print(f"  Download from fred.stlouisfed.org - ticker: "
                  f"{config.MACRO_TICKERS_FRED.get(name, name)}")

    # ------------------------------------------------------------------
    # STEP 3b: Load GPR (geopolitical risk index) - see Section 2b
    # ------------------------------------------------------------------
    gpr_path = os.path.join(config.RAW_DATA_DIR, config.RAW_GPR_FILE)
    if os.path.exists(gpr_path):
        macro_series["GPR"] = load_gpr_excel(gpr_path)
    else:
        print(f"\n[WARNING] GPR file not found: {gpr_path}")
        print("  Download from policyuncertainty.com/gpr.html")

    # ------------------------------------------------------------------
    # STEP 4: Remove weekends and resample to weekly
    # ------------------------------------------------------------------
    print("\n  Resampling to weekly Friday close...")

    curves_weekly = remove_weekends_and_resample(futures_raw)
    curves_weekly = filter_sample_period(curves_weekly)
    print(f"  Curves: {len(curves_weekly)} weekly observations")

    macro_weekly_dict = {}

    if inventory_series is not None:
        inv_weekly = remove_weekends_and_resample(
            inventory_series.to_frame()
        )['inventory']
        inv_weekly = filter_sample_period(inv_weekly)
        # Apply lag AFTER resampling
        inv_lagged = apply_inventory_lag(inv_weekly, config.INVENTORY_LAG)
        macro_weekly_dict['inventory'] = inv_weekly
        macro_weekly_dict[f'inventory_lag{config.INVENTORY_LAG}'] = inv_lagged

    for name, series in macro_series.items():
        weekly = remove_weekends_and_resample(series.to_frame())[name]
        weekly = filter_sample_period(weekly)
        macro_weekly_dict[name] = weekly

    # Combine macro into one DataFrame
    if macro_weekly_dict:
        macro_df = pd.DataFrame(macro_weekly_dict)
        # Forward fill up to 1 week for minor gaps (e.g. holidays)
        macro_df = macro_df.ffill(limit=1)
        print(f"  Macro: {len(macro_df)} weekly observations, "
              f"{len(macro_df.columns)} variables")
    else:
        print("  [WARNING] No macro data loaded - macro.parquet will be empty.")
        macro_df = pd.DataFrame(index=curves_weekly.index)

    # ------------------------------------------------------------------
    # STEP 5: Align and merge
    # ------------------------------------------------------------------
    print("\n  Aligning curves and macro to common weekly index...")

    combined = curves_weekly.join(macro_df, how='left')

    # Report alignment
    n_curves = len(curves_weekly)
    n_combined = len(combined)
    print(f"  Combined: {n_combined} weekly observations "
          f"({n_curves} curve dates, {len(macro_df)} macro dates)")

    if n_combined != n_curves:
        print("  [WARNING] Some curve dates have no macro data - check alignment")

    # ------------------------------------------------------------------
    # STEP 5b: Compute multi-lookback macro change features
    # ------------------------------------------------------------------
    if not macro_df.empty:
        print("\n  Computing multi-lookback macro change features...")
        macro_changes = compute_macro_changes(
            macro_df,
            windows=[1, 4, 13, 26],   # 26W needed for US3M_d26W and GPR_d26W
            zscore_window=52,
        )
        print(f"  Change features: {list(macro_changes.columns)}")
    else:
        macro_changes = pd.DataFrame(index=macro_df.index)

    # ------------------------------------------------------------------
    # STEP 6: Save to parquet
    # ------------------------------------------------------------------
    print("\n  Saving processed files...")

    curves_path = os.path.join(config.PROCESSED_DATA_DIR, 'curves.parquet')
    curves_weekly.to_parquet(curves_path)
    print(f"  ✓ {curves_path}  shape: {curves_weekly.shape}")

    macro_path = os.path.join(config.PROCESSED_DATA_DIR, 'macro.parquet')
    macro_df.to_parquet(macro_path)
    print(f"  ✓ {macro_path}  shape: {macro_df.shape}")

    combined_path = os.path.join(config.PROCESSED_DATA_DIR, 'combined.parquet')
    combined.to_parquet(combined_path)
    print(f"  ✓ {combined_path}  shape: {combined.shape}")

    if not macro_changes.empty:
        changes_path = os.path.join(config.PROCESSED_DATA_DIR, 'macro_changes.parquet')
        macro_changes.to_parquet(changes_path)
        print(f"  ✓ {changes_path}  shape: {macro_changes.shape}")
        print(f"     Change variables: {list(macro_changes.columns)}")

    # ------------------------------------------------------------------
    # STEP 7: Coverage report and figures
    # ------------------------------------------------------------------
    print_coverage_report(curves_weekly, macro_df)

    print("\n  Generating diagnostic figures...")
    save_diagnostic_figures(curves_weekly, macro_df)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  curves.parquet:   {curves_weekly.shape[0]} obs × "
          f"{curves_weekly.shape[1]} maturities")
    print(f"  macro.parquet:    {macro_df.shape[0]} obs × "
          f"{macro_df.shape[1]} variables")
    print(f"  combined.parquet: {combined.shape[0]} obs × "
          f"{combined.shape[1]} columns")
    print(f"\nNext step: run python code/02_nelson_siegel.py")
    print("=" * 60)


if __name__ == "__main__":
    main()