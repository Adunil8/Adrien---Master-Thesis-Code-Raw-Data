# =============================================================================
# config.py - Master configuration for thesis analysis
# ALL parameters live here. Change one value → re-render → everything updates.
# Never hardcode values in analysis scripts - always import from here.
# =============================================================================

# =============================================================================
# INSTRUMENT
# =============================================================================
METAL = "copper"          # scripts use this label for output filenames and plot titles only

# =============================================================================
# RAW DATA FILE NAMES (place files in data/raw/)
# =============================================================================
# Bloomberg futures export - one sheet per year, tickers as column headers
RAW_FUTURES_FILE    = "Copper_Futures_Extraction_Bloomberg_Values.xlsx"

# Bloomberg inventory - same sheet-per-year structure, one data column
# LME copper on-warrant (live) stocks, world aggregate: NLECA Index, field PX_LAST
# Confirmed via LMEI screen: NLECA "Warrant" (World) = on-warrant stock,
# distinct from "Wrnt ANNU" (cancelled warrants) - Total stock = Warrant + Cancelled.
RAW_INVENTORY_FILE           = "LME Warrants Copper Extraction Bloomberg.xlsx"  # Bloomberg NLECA Index
RAW_CANCELLED_WARRANTS_FILE  = "lme_cancelled_warrants.xlsx"  # Bloomberg ticker TBD - cancelled warrants, still pending
# Cancelled / (Warrant + Cancelled) = cancelled warrants ratio (forward-looking β₁ signal)
# Cancelled warrants file: same sheet-per-year structure as futures. Add when available.

# FRED macro variable CSV files - auto-download via code/01b_download_macro.py
# Each file has two columns: observation_date, VALUE
# All series are daily; 01_data_cleaning.py resamples to W-FRI.
RAW_FRED_FILES = {
    # ── Production macro set (VIF + Granger + sub-sample-stability disciplined,
    #    see main.qmd Section 3.1) ────────────────────────────────────────────
    "DXY":    "fred_dxy.csv",       # DTWEXBGS  - US Dollar Index (broad).      Used as DXY_z52W
    "VIX":    "fred_vix.csv",       # VIXCLS    - CBOE Volatility Index.        Used as VIX_level
    "US3M":   "fred_us3m.csv",      # DGS3MO    - US 3M yield (short rate).     Used as US3M_d26W
    # GPR (Caldara & Iacoviello 2022) loaded separately - see RAW_GPR_FILE below. Used as GPR_d26W
    # (Inventory is loaded from RAW_INVENTORY_FILE, not RAW_FRED_FILES - used as inventory_d4W)
    #
    # ── Retained for the robustness/appendix comparison table only - NOT in the
    #    production VAR/LP spec (superseded by US3M: "stronger, more direct
    #    carry" signal per 05_models.py) ─────────────────────────────────────
    "T10Y2Y": "fred_t10y2y.csv",    # T10Y2Y - 10Y–2Y yield spread (recession signal)
    # "BRENT": "fred_brent.csv",    # DCOILBRENTEU - DROPPED: VIF=10.3 (>10 threshold), null Granger
    #                                  causality to β0, redundant with DXY (USD-commodity channel).
    #                                  See Section 3.1 / 4.3 for full VIF comparison.
    # "US10Y": "fred_us10y.csv",    # DGS10 - DROPPED: VIF=17, collinear with T10Y2Y and DXY
    # "AUDUSD": "fred_audusd.csv",  # DEXUSAL  - DROPPED: VIF=143 due to collinearity with DXY
    # NOTE: Brent chosen over WTI because:
    #   (1) LME is a London/global market - Brent is the global oil benchmark
    #   (2) WTI went negative in April 2020 (Cushing logistics artifact, not economic
    #       signal) - would create a severe outlier in VAR residuals during COVID case study
    #   (3) Brent and WTI correlate >0.98 so forecast power is identical; Brent is cleaner
    # ── Optional (uncomment after VIF check in Phase 3) ──────────────────────
    # "CNYUSD": "fred_cnyusd.csv",  # DEXCHUS    - CNY per USD (China demand proxy)
    # "US2Y":   "fred_us2y.csv",    # DGS2       - US 2-year yield (short rate)
    # DO NOT ADD: fred_us1m.csv (DGS1MO) - only available from 2021-06-11, too short
}

# FRED tickers - for documentation and 01b_download_macro.py auto-download
# Note: AUDUSD/CNYUSD in FRED = foreign-currency per 1 USD (rise = weaker AUD = bearish copper)
MACRO_TICKERS_FRED = {
    "DXY":    "DTWEXBGS",
    "VIX":    "VIXCLS",
    "US10Y":  "DGS10",
    "T10Y2Y": "T10Y2Y",
    "AUDUSD": "DEXUSAL",
    "BRENT":  "DCOILBRENTEU",   # Brent, not WTI - see note in RAW_FRED_FILES above
    "CNYUSD": "DEXCHUS",    # optional - uncomment in RAW_FRED_FILES after VIF check
    "US2Y":   "DGS2",       # optional - derivable as US10Y - T10Y2Y
}
# NOTE: PMI excluded - monthly frequency creates artificial autocorrelation
# when forward-filled to weekly. Industrial demand proxied by LME inventory.
# NOTE: Inventory + CancelledWarrants sourced from Bloomberg - add when available.

# Geopolitical Risk index (Caldara & Iacoviello 2022, American Economic Review)
# Source: policyuncertainty.com/gpr.html - daily GPR index (GPRD column)
# Part of the production macro set - see 01_data_cleaning.py Section 2b and
# main.qmd Section 3.1 for the selection evidence (multi-horizon Granger +
# sub-sample stability test, GPR -> beta1 slope factor).
RAW_GPR_FILE = "data_gpr_daily_recent.xlsx"

# =============================================================================
# MATURITIES
# =============================================================================
# Months to include in the analysis
MATURITIES_MONTHS = list(range(1, 7))   # 1M through 6M - trade-finance-relevant segment only
# Rationale: commodity trade finance positions at the target institution average
# 3M tenor with a hard ceiling at 6M (high portfolio turnover, working-capital nature).
# Maturities beyond 6M are outside the operational scope and the three-parameter NS
# specification's accuracy degrades at longer maturities relative to the short end.

# =============================================================================
# SAMPLE PERIOD
# =============================================================================
START_DATE  = "2006-01-01"   # First date in the Bloomberg export
END_DATE    = "2026-06-01"   # Last date in the Bloomberg export
TRAIN_END   = "2022-12-31"   # End of in-sample training period
TEST_START  = "2023-01-01"   # Start of out-of-sample test period (Trump era)

# Calibration burn-in ONLY (code/13c_calibration_preperiod.py, 14b): two extra
# years of the same walk-forward, expanding-window AR-Direct forecast that
# TEST_START marks the official start of, generated purely so the rolling
# Platt fit (Section 3.8) has real, point-in-time-correct history to draw on
# from day one of the officially reported test period, instead of starting
# from zero. Does NOT change TRAIN_END/TEST_START or any Chapter 4-5 metric:
# no RMSE, hit rate, or AUC number is ever computed over this window.
CALIBRATION_BURNIN_START = "2021-01-01"

# Earliest origin INCLUDED in the walk-forward residual pool (Section 3.8's
# residual-pool construction, code/13d + code/13e). Chosen from
# code/13d_walkforward_stability_diagnostic.py's coefficient-stability check
# ALONE -- never from any forecast-accuracy, BSS, or AUC result, to avoid the
# same look-ahead-via-hyperparameter-selection risk already found and
# rejected for the Platt window-size grid search. All three affected models' walk-forward
# coefficients show clear small-sample instability from 2006-2011 (median
# week-to-week coefficient change 0.02-0.53, declining), settling into a
# flat, stable regime from 2012 onward (median 0.003-0.03) that persists for
# a decade, interrupted only by legitimate, informative shocks (2020 COVID,
# 2022 Ukraine/copper surge) that a correctly-specified model SHOULD react
# to -- those spikes are a feature, not evidence of continued instability.
# Origins before this date are still used to FIT each walk-forward model
# (the window itself starts at true data availability, ~mid-2006) -- this
# constant only controls which origins' resulting residuals are KEPT in the
# pool, i.e. a burn-in exclusion, not a restart of the estimation window.
RESIDUAL_POOL_BURNIN_START = "2012-01-01"

# =============================================================================
# DATA PROCESSING
# =============================================================================
RESAMPLE_FREQ  = "W-FRI"    # Weekly Friday close
INVENTORY_LAG  = 2          # Periods to lag inventory in VAR
                            # Mitigates simultaneity: curve ↔ inventory
                            # (Theory of Storage bidirectional causality)

# =============================================================================
# NELSON-SIEGEL MODEL
# =============================================================================
# Lambda controls the maturity at which the curvature factor peaks
# Estimated via grid search (Phase 2, script 02_nelson_siegel.py)
# Initial value - will be overwritten by grid search result
LAMBDA_NS           = 1.95    # placeholder - overwritten by constrained grid search in 02_nelson_siegel.py
# NS curvature peak formula (from utils.py ns_loadings): τ* = 1.793 × λ  (where lt = τ/λ)
# With maturities τ ∈ {1,2,3,4,5,6} months, the curvature is identifiable ONLY when τ* ∈ [1,6]:
#   λ_min_theoretical = 1/1.793 ≈ 0.56  (peak at τ*=1M - front end of curve)
#   λ_max_theoretical = 6/1.793 ≈ 3.35  (peak at τ*=6M - back end of curve)
# With only 6 maturities and 3 NS parameters, the RMSE landscape over λ is nearly flat
# (log-scale RMSE < 0.001 for all λ) - standard grid search cannot discriminate.
# SOLUTION: Constrain the grid to λ ∈ [0.4, 4.0] so the curvature peak is restricted to
# fall within [0.72M, 7.2M] - i.e. always within or immediately adjacent to the 1–6M range.
# Within this constrained range, the grid returns the empirically optimal λ.
# Diebold-Li (2006) convention: peak at median maturity (3.5M) → λ = 3.5/1.793 ≈ 1.95.
LAMBDA_NS_GRID_MIN  = 0.4     # peak at τ*=0.72M - just before the 1M front end
LAMBDA_NS_GRID_MAX  = 4.0     # peak at τ*=7.2M - just beyond the 6M ceiling
LAMBDA_NS_GRID_N    = 500     # 500-point grid over [0.4, 4.0] → step ≈ 0.007 per node

# NS fit quality thresholds - expressed as % of mean m01 price (training period)
# Computed dynamically in 02_nelson_siegel.py using mean_price * pct
NS_RMSE_WARN_PCT = 0.02   # 2% of mean price - warn if mean RMSE exceeds this
NS_RMSE_FLAG_PCT = 0.05   # 5% of mean price - flag individual dates above this

# Price transformation for NS fitting
# 'log' : fit NS on log prices - following Bianchi et al. (2023)
#         factors are dimensionless, percentage-scale; more stable across price regimes
#         recommended for production runs
# 'raw' : fit NS on raw prices in original units (e.g. ¢/gal, $/t)
#         simpler to interpret directly; less stable when price level shifts
NS_PRICE_SCALE = 'log'   # change to 'raw' to switch

# =============================================================================
# REGIME CLASSIFICATION (seasonally-adjusted z-score)
# =============================================================================
# Z-score thresholds for regime classification
# Computed relative to same calendar week in prior years (training data only)
REGIME_ZSCORE_BACKWARDATION =  1.0   # beta1 z-score > +1.0 → backwardation
REGIME_ZSCORE_CONTANGO      = -1.0  # beta1 z-score < -1.0 → contango
# Between -1.0 and +1.0 → neutral (within seasonal norm)
# ±1.0 identifies UNUSUAL curve states vs seasonal norm. Neutral = no abnormal risk signal.

# =============================================================================
# DYNAMIC FACTOR MODELS
# =============================================================================
# Exponential weighting decay factor for VAR/LP parameter estimation
# Half-life = log(0.5) / log(LAMBDA_EW) ≈ 230.7 weeks ≈ 4.4 years
#
# The standard reference value for exponential weighting is LAMBDA_EW = 0.97,
# the RiskMetrics convention (J.P. Morgan, 1994; half-life ≈ 22.7 weeks ≈ 5.5
# months) - see RiskMetrics1994 in references.bib. That convention was
# calibrated for daily equity-return volatility estimation, a single-variable,
# high-frequency setting very different from this thesis's application: a
# weekly multi-variable regression with 7 VAR/LP variables and p=2 lags
# (15 parameters per equation). Tested directly against this setting in
# 07b_lambda_robustness.py, 0.97 gives an effective-sample-to-parameter ratio
# of only ~4.4:1 - borderline for stable estimation - and measurably worse
# out-of-sample RMSE (14% higher at 1M, 27% higher at 3M) than the data-optimal
# value found by the grid search. LAMBDA_EW = 0.997 is used here for that
# reason: not a rejection of the RiskMetrics convention in general, but a
# finding that it is a poor fit for this specific estimation problem, which
# the grid search was run to check rather than assume. See lambda_ew_optimal.
# parquet and main.qmd Section 3.6 for the full evidence.
LAMBDA_EW = 0.997

# Maximum lag order to test in AIC/BIC selection
VAR_MAX_LAGS = 4    # test p = 1, 2, 3, 4 - select via AIC

# =============================================================================
# MACRO VARIABLE SPEC - SINGLE SOURCE OF TRUTH FOR THE VAR/LP MODEL
# =============================================================================
# This is the ONLY place the 5-variable macro spec should be written. Every
# script that builds the VAR/LP endogenous block (03, 04, 05, 06, 07, 07b, 08,
# 09, 11) MUST import MACRO_VAR_SPEC from here rather than hardcoding the list.
#
# This list is written here ONCE on purpose: an earlier draft had the same
# variable list typed independently into 5+ scripts, and when the spec
# changed, some of those copies fell out of sync, silently producing
# internally-inconsistent results across chapters. Centralising here closes
# off that bug class: change the spec ONCE, then run code/run_all.py to
# regenerate everything downstream.
#
# Selection evidence: multi-horizon local-projection Granger causality
# (04b/04c) + sub-sample stability test (training split 2006-2014 vs
# 2014-2022) + full stationarity re-audit (ADF+KPSS on the ACTUAL transform
# used, not a proxy) + VIF (all < 1.15, no collinearity) - see main.qmd
# Section 3.1. Two variables use a transform chosen over the more obvious
# candidate for a specific reason:
#   - inventory uses inventory_lag2_d4W, not the unlagged d4W. Lagging by
#     two periods is what gives INVENTORY_LAG its anti-simultaneity
#     protection (Theory of Storage feedback between the curve and
#     inventory levels). Correctly lagged, Granger significance drops from
#     p=0.040 to p=0.123 (not significant) -- kept in the model anyway for
#     VIF/theoretical reasons, but this is reported honestly, not hidden.
#   - US3M uses d1W, not d26W. d26W is non-stationary in this sample
#     (ADF p=0.234, KPSS p=0.01), the weakest stationarity case of any
#     candidate tested. d1W is stationary under the same "conflicting"
#     verdict as DXY_z52W etc., and has equal-or-better Granger
#     significance for both beta0 and beta1.
MACRO_VAR_SPEC = ["DXY_z52W", "inventory_lag2_d4W", "US3M_d1W", "GPR_d26W", "VIX"]

# Split by source file, for scripts that read macro_changes.parquet and
# macro.parquet separately (levels come from macro.parquet, everything else
# from macro_changes.parquet):
MACRO_CHANGE_COLS = ["DXY_z52W", "inventory_lag2_d4W", "US3M_d1W", "GPR_d26W"]  # from macro_changes.parquet
MACRO_LEVEL_COLS  = ["VIX"]                                                     # from macro.parquet

# =============================================================================
# FORECAST HORIZONS
# =============================================================================
# Horizons in weeks - covering deal-relevant timescales
# 1W=1, 2W=2, 1M=4, 2M=9, 3M=13, 6M=26
FORECAST_HORIZONS_WEEKS = [1, 2, 4, 9, 13, 26]

# Human-readable labels for each horizon (used in figures and tables)
HORIZON_LABELS = {
    1:  "1 Week",
    2:  "2 Weeks",
    4:  "1 Month",
    9:  "2 Months",
    13: "3 Months",
    26: "6 Months",
}

# PRIMARY horizons for institutional application (match trade finance tenors)
PRIMARY_HORIZONS = [4, 13]      # 1M and 3M - feature prominently in results
# 6M horizon (26W) reported as boundary condition - accuracy degrades

# =============================================================================
# MARGIN CALL THRESHOLDS
# =============================================================================
# Initial margin thresholds as % of notional
# Used to compute P(ΔF > k%) and P(ΔF < -k%)
MARGIN_THRESHOLDS = [0.05, 0.08, 0.10, 0.15]  # 5%, 8%, 10%, 15%

# =============================================================================
# STYLISED DEAL (Phase 6 - institutional application)
# =============================================================================
STYLISED_DEAL = {
    "notional_tonnes":    1000,     # deal size in metric tonnes
    "tenor_weeks":        13,       # 3-month deal tenor
    "initial_margin_pct": 0.08,     # 8% initial margin threshold
    "position":           "both",   # output P(up) for short AND P(down) for long
                                    # bank selects based on actual book direction
    # SHORT paper (short futures) → margin call when price RISES
    #   typical for: bank finances BUYER (long physical, short paper hedge)
    # LONG paper (long futures)  → margin call when price FALLS
    #   typical for: bank finances SELLER (short physical, long paper hedge)
}

# =============================================================================
# PERFORMANCE METRICS
# =============================================================================
RISK_FREE_RATE       = 0.02    # annualised, for Sharpe ratio
ANNUALISATION_FACTOR = 52      # weekly → annualise by sqrt(52)

# =============================================================================
# FILE PATHS
# =============================================================================
RAW_DATA_DIR        = "data/raw/"
PROCESSED_DATA_DIR  = "data/processed/"
FIGURES_DIR         = "report/figures/"

# =============================================================================
# PLOTTING
# =============================================================================
FIGURE_DPI    = 150
FIGURE_FORMAT = "png"
PLOT_STYLE    = "seaborn-v0_8-whitegrid"

# Colour palette - consistent across all figures
COLORS = {
    "backwardation": "#16a34a",   # green
    "contango":      "#dc2626",   # red
    "neutral":       "#6b7280",   # grey
    "model0":        "#9ca3af",   # random walk - light grey
    "model1":        "#60a5fa",   # AR(p) - blue
    "model2":        "#f59e0b",   # VAR+macro - amber
    "model3":        "#7c3aed",   # VAR+macro+EW - purple (proposed model)
    "actual":        "#111827",   # black - actual/realised values
}