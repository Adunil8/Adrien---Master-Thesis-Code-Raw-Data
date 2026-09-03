"""
05_models.py - Phase 3c: Estimate all four models (Section 3.5).

Model hierarchy:
  Model 0: Random walk - naïve baseline
  Model 1: AR(p) per NS factor - equal weights (Diebold & Li 2006)
  Model 2: VAR(p) + macro - equal weights (macro-augmented)
  Model 3: VAR(p) + macro + EW (λ=0.97) - full proposed model

Inputs  : data/processed/ns_factors.parquet
          data/processed/macro.parquet
          data/processed/lag_selection.parquet
Outputs : data/processed/model_fits.pkl  (fitted model objects - in-sample)

Run: python code/05_models.py

This script fits models on the FULL TRAINING sample (2006–2022) for
diagnostic purposes. The rolling out-of-sample loop in 07_forecast_evaluation.py
re-estimates models at each test-period step.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR, AutoReg

from config import (
    TRAIN_END, LAMBDA_EW, VAR_MAX_LAGS,
    PROCESSED_DATA_DIR, MACRO_CHANGE_COLS, MACRO_LEVEL_COLS,
)
from utils import fit_var_ew


def prepare_endog(factors: pd.DataFrame, macro: pd.DataFrame,
                  beta0_differenced: bool = True) -> pd.DataFrame:
    """
    Combine NS factors and macro into a single DataFrame.
    Optionally first-difference β₀ if unit root was detected.
    """
    df = pd.concat([factors[["beta0", "beta1", "beta2"]], macro], axis=1).dropna()
    if beta0_differenced:
        df["beta0"] = df["beta0"].diff()
        df = df.dropna()
    return df


# ── Model 0: Random Walk ───────────────────────────────────────────────────────

def forecast_random_walk(last_observation: np.ndarray, h: int) -> np.ndarray:
    """
    Random walk forecast: all horizons equal the last observed value.
    Returns shape (h, n_vars).
    """
    return np.tile(last_observation, (h, 1))


# ── Model 1: AR(p) per factor - equal weights ─────────────────────────────────

def fit_ar_per_factor(factors: pd.DataFrame, p: int) -> dict:
    """
    Fit independent AR(p) model for each of β₀, β₁, β₂.
    Returns dict of {factor_name: fitted AutoReg model}.
    """
    models = {}
    for col in factors.columns:
        s = factors[col].dropna()
        model  = AutoReg(s, lags=p, old_names=False)
        result = model.fit()
        models[col] = result
    return models


def forecast_ar(ar_models: dict, h: int) -> pd.DataFrame:
    """
    Generate h-step AR forecasts for each factor.
    Returns DataFrame shape (h, 3).
    """
    forecasts = {}
    for col, res in ar_models.items():
        # AutoReg.predict returns in-sample; use forecast for OOS
        fcast = res.forecast(steps=h)
        forecasts[col] = fcast.values
    return pd.DataFrame(forecasts)


# ── Model 2: VAR(p) + macro - equal weights ──────────────────────────────────

def fit_var_equal(endog: pd.DataFrame, p: int) -> object:
    """
    Fit VAR(p) on combined [factors, macro] with equal weights.
    Uses HAC-robust covariance (Newey-West) regardless of ARCH test result.
    """
    model  = VAR(endog)
    result = model.fit(maxlags=p, ic=None, trend="c")
    # Note: statsmodels VAR does not natively support HAC SE at fit time.
    # HAC correction applied at diagnostic stage (06_diagnostics.py).
    return result


def forecast_var(var_result, h: int, factor_cols: list) -> np.ndarray:
    """
    Generate h-step VAR forecast. Returns forecasted factor values only.
    Shape: (h, len(factor_cols))
    """
    fcast_full = var_result.forecast(var_result.endog, steps=h)  # (h, n_vars)
    factor_idx = [list(var_result.model.endog_names).index(c) for c in factor_cols]
    return fcast_full[:, factor_idx]


# ── Model 3: VAR(p) + macro + EW (λ=0.97) ────────────────────────────────────
#
# Proper exponentially-weighted VAR via per-equation WLS on a lag design matrix
# built from the ORIGINAL (unscaled) series - see utils.fit_var_ew for why
# naive row-scaling of the series before building lags is NOT valid WLS for
# an autoregressive model (it misattributes weight w_t to lag regressors that
# should carry w_{t-1}, w_{t-2}, ... instead).

def aic_bic_from_sigma(sigma_u: np.ndarray, n_eq: int, k: int, p: int) -> tuple[float, float]:
    """VAR AIC/BIC (Lütkepohl 2005 form) from a residual covariance matrix."""
    n_params = k * (k * p + 1)
    sign, logdet = np.linalg.slogdet(sigma_u)
    aic = logdet + 2 * n_params / n_eq
    bic = logdet + np.log(n_eq) * n_params / n_eq
    return aic, bic


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 3c - Model Estimation (Full Training Sample)")
    print("=" * 60)

    # Load inputs
    factors = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet")
    )[["beta0", "beta1", "beta2"]]

    # ── Macro variable specification (confirmed via 04_granger.py / 04b/04c) ──
    #
    # Systematic screening of 13 candidates x 6 transformations x 3 NS factors
    # x 9 horizons (04b single-step + 04c multi-horizon local-projection
    # Granger, training 2006-2022), followed by a full stationarity re-audit
    # (ADF+KPSS on the ACTUAL transform used, not assumed) once the candidate
    # set was picked. Final 5-variable selection (bivariate Granger, best lag
    # 1-4, official numbers from 04_granger.py Step 2):
    #
    #   DXY_z52W            -> beta0  p=0.0006 ***  USD 52W z-score; dollar
    #                                                deviation from trailing
    #                                                norm suppresses copper.
    #   inventory_lag2_d4W  -> beta1  p=0.097  (*)   4W inventory change,
    #                                                LAGGED 2 weeks so the
    #                                                regressor cannot reflect
    #                                                the same week's curve
    #                                                move (INVENTORY_LAG=2).
    #                                                Weak (10% level only, not
    #                                                5%) once properly lagged
    #                                                -- see correction note.
    #   US3M_d1W            -> beta0  p=0.0105 **    US 3M T-bill, 1-week
    #                        -> beta1  p=0.0060 ***  change. Significant for
    #                                                BOTH factors; captures
    #                                                short-rate repricing of
    #                                                carry cost.
    #   GPR_d26W            -> beta1  p=0.0009 ***   Caldara-Iacoviello (2022)
    #                                                geopolitical risk index,
    #                                                26W change. Sub-sample
    #                                                stability confirmed in
    #                                                BOTH training halves
    #                                                (2006-2014 p=0.0105**,
    #                                                2014-2022 p=0.0035***) --
    #                                                structural, not a
    #                                                Trump-era artefact.
    #   VIX (level)         -> beta0  p=0.0007 ***   Empirically the stronger
    #                        -> beta2  p=0.13 (n.s.) channel is price level
    #                                                (risk-off selling), not
    #                                                the theoretically-
    #                                                motivated curvature link
    #                                                (Section 2.4) -- both
    #                                                reported, not just the
    #                                                one that "worked".
    #
    # Excluded with documented justification (see 04b_macro_screening.py):
    #   AUD/USD   - collinear with DXY (r>0.7 in training)
    #   Brent     - commodity cycle endogeneity; captured by VIX + inventory
    #   CNY/USD   - β₂ effect only at level; stationarity ambiguous; β₂ secondary
    #   COT       - predicts β₀/β₂, not β₁; patchy across transforms (see 3.1);
    #               a weaker, less parsimonious case than GPR
    #   T10Y2Y, US10Y, US2Y - superseded by US3M (stronger, more direct carry)
    #   DFII10_z52W - real rate; Granger β₁ p=0.012 ** but superseded by US3M
    #                 and adding it reduces ESS/params further
    #   TEDRATE   - unavailable post-June 2023 (LIBOR discontinued)
    #   LCUACANC  - cancelled warrants ratio; Bloomberg data not available locally;
    #               documented as scope limitation in Section 3.1
    #
    # WHY EACH FINAL TRANSFORM IS WHAT IT IS:
    #   - GPR_d26W is a trailing 26-week change, computable in real time at
    #     t, exactly like DXY_z52W's 52-week window. The 6-month lookback is
    #     not a real-time obstacle.
    #   - US3M enters as US3M_d1W, not US3M_d26W. Both ADF and KPSS agree
    #     US3M_d26W is NON-STATIONARY in this sample (p=0.234, p=0.01) -- the
    #     weakest stationarity case of any variable tested. US3M_d1W is
    #     "conflicting" (the same category as DXY_z52W, T10Y2Y_z52W etc.,
    #     already handled via the Sims-Stock-Watson forecasting-consistency
    #     argument, Section 3.2) and has equal-or-better Granger significance.
    #   - Inventory enters as inventory_lag2_d4W, not a change computed from
    #     RAW (unlagged) inventory: the unlagged version does not honour
    #     INVENTORY_LAG=2 and partly reflects the contemporaneous simultaneity
    #     that lag exists to remove. Correctly lagged, significance drops
    #     from p=0.040 to p=0.097 (10% level only). Kept in the model despite
    #     the weaker result: VIF is clean (1.01), it is the empirical anchor
    #     for the Theory-of-Storage discussion (Section 2.4), and the joint-
    #     VAR/LP out-of-sample RMSE test (Section 4.4) is the decisive test
    #     for forecast value, not this bivariate pre-screen.
    #
    # Parsimony: 5 macro + 3 NS = 8 variables, p=2 lags → 17 params/equation.
    # This is a genuine limitation, not a settled question: the macro-relevance
    # ranking is itself time-varying (see the sub-sample stability results in
    # 3.1, where the inventory→β₁ and US3M→β₀/β₂ relationships that hold in
    # 2006–2014 weaken or vanish in 2014–2022, while GPR→β₁ strengthens). A
    # variable set fixed once at estimation time cannot track that drift; the
    # thesis's operational answer is exponential recency weighting (Model 4)
    # plus the annual-recalibration protocol in Section 6.6, not a claim that
    # this five-variable set is permanently correct - see Limitations (6.6).
    macro_levels  = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro.parquet"))
    macro_changes = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "macro_changes.parquet"))
    macro = pd.concat([
        macro_changes[MACRO_CHANGE_COLS],
        macro_levels[MACRO_LEVEL_COLS],
    ], axis=1)
    print(f"Macro columns in VAR: {list(macro.columns)}")

    lag_sel = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "lag_selection.parquet"))

    p = int(lag_sel["recommended"].iloc[0])
    print(f"Lag order p = {p} (from Phase 3a)")

    # Restrict to training period
    factors_train = factors.loc[:TRAIN_END]
    macro_train   = macro.loc[:TRAIN_END]

    # β₀ differencing decision - stationarity tests (script 03) confirmed β₀ is STATIONARY
    # in levels with λ=1.952 and 1–6M maturities. Use levels directly.
    # Previously True when 1–15M range and λ=15.23 produced a non-stationary level factor.
    # With the restricted 1–6M range, the level factor ADF p=0.042 → stationary.
    BETA0_DIFFERENCED = False

    endog = prepare_endog(factors_train, macro_train,
                          beta0_differenced=BETA0_DIFFERENCED)
    factor_cols = ["beta0", "beta1", "beta2"]
    print(f"Endog shape: {endog.shape} | {'β₀ differenced' if BETA0_DIFFERENCED else 'β₀ in levels'}")

    # ── Fit all models ─────────────────────────────────────────────────────────

    print("\nModel 0: Random walk (no fitting required - last observation as forecast)")
    # No fitting step for RW

    print(f"\nModel 1: AR({p}) per factor - equal weights")
    ar_models = fit_ar_per_factor(endog[factor_cols], p)
    for col, res in ar_models.items():
        print(f"  {col}: AIC={res.aic:.2f}, params={res.params.values.round(4)}")

    print(f"\nModel 2: VAR({p}) + macro - equal weights")
    var_eq = fit_var_equal(endog, p)
    print(f"  AIC={var_eq.aic:.2f}, BIC={var_eq.bic:.2f}")

    print(f"\nModel 3: VAR({p}) + macro + EW (λ={LAMBDA_EW})")
    var_ew = fit_var_ew(endog, p, LAMBDA_EW)
    k = len(var_ew["cols"])
    n_eq = len(var_ew["resid"])
    aic_ew, bic_ew = aic_bic_from_sigma(var_ew["sigma_u"], n_eq, k, p)
    half_life_weeks = np.log(0.5) / np.log(LAMBDA_EW)
    print(f"  Half-life: {half_life_weeks:.1f} weeks ≈ {half_life_weeks/52.18:.1f} years")
    print(f"  AIC={aic_ew:.2f}, BIC={bic_ew:.2f}  (weighted Σ_u basis - not directly comparable to OLS AIC/BIC)")

    # ── Save fitted models ─────────────────────────────────────────────────────
    model_store = {
        "lag_order":          p,
        "beta0_differenced":  BETA0_DIFFERENCED,
        "factor_cols":        factor_cols,
        "ar_models":          ar_models,
        "var_equal":          var_eq,
        "var_ew":             var_ew,
        "endog_train":        endog,
    }
    out_path = os.path.join(PROCESSED_DATA_DIR, "model_fits.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(model_store, f)
    print(f"\nSaved model fits: {out_path}")

    print("\nPhase 3c complete.")
    print("Next: run python code/06_diagnostics.py")


if __name__ == "__main__":
    main()
