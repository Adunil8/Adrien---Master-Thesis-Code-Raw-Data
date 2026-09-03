"""
utils.py - Shared helper functions used across all pipeline scripts.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ── Nelson-Siegel loading functions ───────────────────────────────────────────

def ns_loadings(tau: np.ndarray, lambda_ns: float) -> np.ndarray:
    """
    Compute Nelson-Siegel loading matrix for given maturities and decay parameter.

    Returns shape (len(tau), 3): columns are [L_loading, S_loading, C_loading].
    L (β₀) = 1 (level - long end)
    S (β₁) = (1 - e^{-λτ}) / (λτ)  (slope - short end)
    C (β₂) = (1 - e^{-λτ}) / (λτ) - e^{-λτ}  (curvature - hump)
    """
    tau = np.asarray(tau, dtype=float)
    lt = tau / lambda_ns
    l_load = np.ones_like(tau)
    s_load = (1 - np.exp(-lt)) / lt
    c_load = s_load - np.exp(-lt)
    return np.column_stack([l_load, s_load, c_load])


def ns_curve(betas: np.ndarray, tau: np.ndarray, lambda_ns: float) -> np.ndarray:
    """Reconstruct NS curve from factor vector (β₀, β₁, β₂) and maturities τ."""
    loadings = ns_loadings(tau, lambda_ns)
    return loadings @ np.asarray(betas)


# ── Trailing realised-volatility reference (stylised deal, Chapter 6) ─────────

def trailing_vol_reference(front_month: pd.Series, horizon_weeks: int,
                            lookback_weeks: int = 52) -> pd.Series:
    """
    Trailing realised-volatility reference on the front-month contract
    (LP1, "closest to cash"), scaled to a given horizon. Used both as the
    dollar-sizing basis and as the threshold itself for the stylised deal's
    margin-call probability (Section 3.9): a self-referential "will the
    market move more than its own recent normal range" question, rather
    than an externally-asserted percentage.

    Weekly log returns on `front_month` -> rolling `lookback_weeks`
    (default 52, trailing 12 months) standard deviation, re-estimated at
    every date, a genuine trailing window, not expanding -> annualised
    (x sqrt(52)) -> scaled to `horizon_weeks` (x sqrt(horizon_weeks/52)).

    horizon_weeks should match the question being asked: the deal's own
    tenor in weeks for the Chapter 6 illustration, so a 2-month deal and
    a 3-month deal read different thresholds off the same underlying
    volatility, and the same deal at two different dates reads different
    thresholds too, since sigma itself moves with market conditions.

    Returns a Series indexed by date, NaN for the first lookback_weeks+1
    dates where a full trailing window is not yet available.
    """
    px = front_month.dropna()
    log_ret = np.log(px / px.shift(1)).dropna()
    weekly_sigma = log_ret.rolling(lookback_weeks).std()
    annual_sigma = weekly_sigma * np.sqrt(52)
    return annual_sigma * np.sqrt(horizon_weeks / 52)


# ── Seasonal z-score regime classification ────────────────────────────────────

def seasonal_zscore(beta1_series: pd.Series, train_end: str) -> pd.Series:
    """
    Compute seasonally-adjusted z-score for β₁ (slope factor).

    For each date, compares β₁ to its distribution across the SAME calendar week
    in prior training years, correcting for documented LME copper seasonality.

    Parameters
    ----------
    beta1_series : pd.Series indexed by date, containing β₁ values
    train_end    : str, e.g. "2022-12-31" - statistics computed on training data only

    Returns
    -------
    pd.Series of z-scores, same index as beta1_series
    """
    train_mask = beta1_series.index <= pd.Timestamp(train_end)
    train_data = beta1_series[train_mask].copy()

    # Map each date to ISO week number (1–52)
    week_stats = (
        train_data
        .groupby(train_data.index.isocalendar().week)
        .agg(["mean", "std"])
    )

    z_scores = pd.Series(index=beta1_series.index, dtype=float, name="beta1_zscore")
    for date, val in beta1_series.items():
        week = date.isocalendar().week
        if week in week_stats.index:
            mu  = week_stats.loc[week, "mean"]
            sig = week_stats.loc[week, "std"]
            z_scores[date] = (val - mu) / sig if sig > 0 else 0.0

    return z_scores


def classify_regime(z_series: pd.Series, threshold: float = 1.0) -> pd.Series:
    """
    Classify curve regime from seasonally-adjusted z-scores.

    Returns: pd.Series of strings in {"backwardation", "contango", "neutral"}
    """
    regime = pd.Series("neutral", index=z_series.index, name="regime")
    regime[z_series >  threshold] = "backwardation"
    regime[z_series < -threshold] = "contango"
    return regime


# ── Exponential weights ────────────────────────────────────────────────────────

def exponential_weights(n: int, lambda_ew: float) -> np.ndarray:
    """
    Compute exponential weighting vector for n observations.

    weights[t] = λ^(T−1−t) for t = 0, ..., T−1 (most recent observation has weight 1).
    Normalised so sum = n (preserves effective sample-size intuition).
    """
    w = np.array([lambda_ew ** (n - 1 - t) for t in range(n)], dtype=float)
    return w * n / w.sum()


# ── Exponentially-weighted VAR (proper WLS, not row-scaling) ──────────────────
#
# Row-scaling the raw series by sqrt(w_t) before building lags is NOT valid WLS
# for an autoregressive model: once lags are built from the scaled series, the
# lag-1 regressor in equation t carries weight w_{t-1} (not w_t), lag-2 carries
# w_{t-2}, etc. - the weight gets attached to the wrong observation. The correct
# approach builds the lag design matrix from the ORIGINAL series first, then
# applies w_t to the correct equation row (the date of the dependent variable).

def build_var_design(endog: pd.DataFrame, p: int):
    """
    Build the VAR(p) lag design matrix from unscaled data.

    Returns
    -------
    X     : (T-p, 1+k*p) ndarray - row t = [1, y_{t-1}, ..., y_{t-p}]
    Y     : (T-p, k) ndarray     - row t = y_t
    dates : DatetimeIndex of length T-p - dates of the dependent variable y_t
    cols  : list of column names, in endog order
    """
    cols = list(endog.columns)
    arr  = endog.values
    T    = arr.shape[0]
    X_rows, Y_rows = [], []
    for t in range(p, T):
        row = [1.0]
        for lag in range(1, p + 1):
            row.extend(arr[t - lag])
        X_rows.append(row)
        Y_rows.append(arr[t])
    X = np.array(X_rows)
    Y = np.array(Y_rows)
    dates = endog.index[p:]
    return X, Y, dates, cols


def fit_var_ew(endog: pd.DataFrame, p: int, lambda_ew: float) -> dict:
    """
    Exponentially-weighted VAR(p) via per-equation WLS on the correct lag design.

    Weight w_t (RiskMetrics-style, half-life = ln(0.5)/ln(λ)) is applied to
    equation row t - i.e. to the date of the dependent variable y_t - so each
    equation is a standard WLS regression of y_t on [1, y_{t-1}, ..., y_{t-p}].

    Returns dict with:
      coefs    : (k, 1+k*p) ndarray - stacked [intercept, A1, ..., Ap] per equation
      resid    : (T-p, k) ndarray - WLS residuals (original scale)
      sigma_u  : (k, k) weighted residual covariance (for forecast error propagation)
      cols     : list of variable names
      p        : lag order
      last_obs : (p, k) ndarray - last p observations, in time order (forecast seed)
    """
    from statsmodels.regression.linear_model import WLS

    X, Y, dates, cols = build_var_design(endog, p)
    n_eq = len(Y)
    w    = exponential_weights(n_eq, lambda_ew)
    k    = Y.shape[1]

    coefs = np.zeros((k, X.shape[1]))
    resid = np.zeros_like(Y)
    for j in range(k):
        wls = WLS(Y[:, j], X, weights=w).fit()
        coefs[j, :] = wls.params
        resid[:, j] = wls.resid

    sigma_u = (resid * w[:, None]).T @ resid / w.sum()

    return {
        "coefs": coefs, "resid": resid, "sigma_u": sigma_u,
        "cols": cols, "p": p, "last_obs": endog.values[-p:],
    }


def forecast_var_ew(fit: dict, h: int) -> np.ndarray:
    """
    h-step-ahead forecast from a fitted exponentially-weighted VAR.

    Iterates the fitted coefficients forward from the last p observations
    (in the ORIGINAL, unscaled level space). Returns (h, k) ndarray.
    """
    coefs = fit["coefs"]
    p     = fit["p"]
    k     = coefs.shape[0]
    history = [row.copy() for row in fit["last_obs"]]   # length p, oldest→newest

    forecasts = np.zeros((h, k))
    for step in range(h):
        x = [1.0]
        for lag in range(1, p + 1):
            x.extend(history[-lag])
        y_hat = coefs @ np.array(x)
        forecasts[step] = y_hat
        history.append(y_hat)
    return forecasts


def forecast_cov_var_ew(fit: dict, h: int) -> np.ndarray:
    """
    h-step forecast error covariance matrix for a fitted EW-VAR, via the
    standard VAR companion-form MA recursion: Σ_h = Σ_{i=0}^{h-1} Φ_i Σ_u Φ_i'
    """
    coefs   = fit["coefs"]
    p       = fit["p"]
    k       = coefs.shape[0]
    sigma_u = fit["sigma_u"]

    A = coefs[:, 1:]                     # drop intercept column, shape (k, k*p)
    comp = np.zeros((k * p, k * p))
    comp[:k, :] = A
    if p > 1:
        comp[k:, :-k] = np.eye(k * (p - 1))

    sigma_h     = np.zeros((k, k))
    comp_power  = np.eye(k * p)
    for _ in range(h):
        phi_i = comp_power[:k, :k]
        sigma_h += phi_i @ sigma_u @ phi_i.T
        comp_power = comp_power @ comp
    return sigma_h


# ── Probabilistic forecast outputs ────────────────────────────────────────────

def prob_price_rise(mu_pct: float, sigma_pct: float, k_pct: float) -> float:
    """
    P(ΔF/F > +k%) for a SHORT futures position margin call trigger.
    mu_pct, sigma_pct, k_pct all expressed as decimals (e.g. 0.08 for 8%).
    """
    return 1.0 - norm.cdf((k_pct - mu_pct) / sigma_pct)


def prob_price_fall(mu_pct: float, sigma_pct: float, k_pct: float) -> float:
    """
    P(ΔF/F < -k%) for a LONG futures position margin call trigger.
    mu_pct, sigma_pct, k_pct all expressed as decimals (e.g. 0.08 for 8%).
    """
    return norm.cdf((-k_pct - mu_pct) / sigma_pct)


# ── Svensson loading functions ────────────────────────────────────────────────

def sv_loadings(tau: np.ndarray, lambda1: float, lambda2: float) -> np.ndarray:
    """
    Compute Svensson (1994) loading matrix.
    Extends Nelson-Siegel with a second curvature term at decay λ₂,
    allowing for a second hump at a different maturity.

    Returns shape (len(tau), 4): [L, S, C1, C2].
    L  (β₀) = 1
    S  (β₁) = (1 - e^{-τ/λ₁}) / (τ/λ₁)
    C1 (β₂) = (1 - e^{-τ/λ₁}) / (τ/λ₁) - e^{-τ/λ₁}
    C2 (β₃) = (1 - e^{-τ/λ₂}) / (τ/λ₂) - e^{-τ/λ₂}
    """
    tau = np.asarray(tau, dtype=float)
    lt1     = tau / lambda1
    l_load  = np.ones_like(tau)
    s_load  = (1 - np.exp(-lt1)) / lt1
    c1_load = s_load - np.exp(-lt1)
    lt2     = tau / lambda2
    c2_load = (1 - np.exp(-lt2)) / lt2 - np.exp(-lt2)
    return np.column_stack([l_load, s_load, c1_load, c2_load])


def sv_curve(betas: np.ndarray, tau: np.ndarray, lambda1: float, lambda2: float) -> np.ndarray:
    """Reconstruct Svensson curve from factor vector (β₀, β₁, β₂, β₃) and maturities τ."""
    return sv_loadings(tau, lambda1, lambda2) @ np.asarray(betas)


# ── Vectorised fit helpers (used in model comparison) ─────────────────────────

def fit_vectorized(
    curves_arr: np.ndarray,
    loadings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a linear factor model to all dates simultaneously via OLS.

    Parameters
    ----------
    curves_arr : (n_dates, n_tau)  - price or log-price matrix, no NaN
    loadings   : (n_tau, n_factors) - pre-computed loading matrix

    Returns
    -------
    betas      : (n_dates, n_factors)
    rmse_arr   : (n_dates,) per-date RMSE
    """
    pinv_X  = np.linalg.pinv(loadings)              # (n_factors, n_tau)
    betas   = curves_arr @ pinv_X.T                  # (n_dates, n_factors)
    fitted  = betas @ loadings.T                     # (n_dates, n_tau)
    resid   = curves_arr - fitted
    rmse_arr = np.sqrt(np.mean(resid ** 2, axis=1))  # (n_dates,)
    return betas, rmse_arr


# ── Data utilities ─────────────────────────────────────────────────────────────

def clean_weekly_friday(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to weekdays only and resample to weekly Friday close.
    Eliminates Bloomberg weekend-duplication artefacts.
    """
    df = df[df.index.dayofweek < 5]       # Mon=0 ... Fri=4; drop Sat/Sun
    df = df.resample("W-FRI").last()      # last available price on each Friday
    return df


def check_missing(df: pd.DataFrame, label: str = "") -> None:
    """Print a brief missing-data summary. Raise if >5% of any column is NaN."""
    pct_miss = df.isna().mean() * 100
    print(f"\n[{label}] Missing data (%):\n{pct_miss[pct_miss > 0].round(2)}")
    if (pct_miss > 5).any():
        raise ValueError(
            f"Column(s) {list(pct_miss[pct_miss > 5].index)} exceed 5% missing. "
            "Check raw data before proceeding."
        )


def brier_score(prob_forecast: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between probability forecasts and binary realisations."""
    return float(np.mean((prob_forecast - outcomes) ** 2))


# ── Local Projection (LP) helpers - shared by 07 and 08 ──────────────────────

def build_lp_matrices(
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    p: int,
    q: int,
    h: int,
    macro_cols: list | None = None,
) -> tuple:
    """
    Build OLS design matrix X and targets Y for the direct h-step LP regression.

    Column order of X:
      [intercept,
       β₀_lag0, β₁_lag0, β₂_lag0,  β₀_lag1, ..., β₂_lag{p-1},   (p AR lags)
       macro0_lag0, ..., macroK_lag{q-1}]                           (q macro lags)

    Y columns: [β₀_{t+h}, β₁_{t+h}, β₂_{t+h}]

    Returns (X, Y, index) or (None, None, None) if insufficient data.
    """
    factor_cols = ["beta0", "beta1", "beta2"]
    if macro_cols:
        combined = pd.concat(
            [factors[factor_cols], macro[macro_cols]], axis=1
        ).dropna()
    else:
        combined = factors[factor_cols].dropna()

    parts = {"intercept": pd.Series(1.0, index=combined.index, name="intercept")}
    for lag in range(p):
        for col in factor_cols:
            parts[f"{col}_lag{lag}"] = combined[col].shift(lag)
    if macro_cols:
        for lag in range(q):
            for col in macro_cols:
                parts[f"{col}_lag{lag}"] = combined[col].shift(lag)

    X_df = pd.DataFrame(parts)
    Y_df = combined[factor_cols].shift(-h)
    full = pd.concat(
        [X_df, Y_df.rename(columns={c: f"y_{c}" for c in factor_cols})],
        axis=1,
    ).dropna()

    if len(full) < p + q + h + 2:
        return None, None, None

    feat_cols = [c for c in full.columns if not c.startswith("y_")]
    tgt_cols  = [c for c in full.columns if c.startswith("y_")]
    return (
        full[feat_cols].values.astype(float),
        full[tgt_cols].values.astype(float),
        full.index,
    )


def fit_lp(
    X: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    OLS or WLS fit for the LP regression.

    Parameters
    ----------
    X       : (n, n_features)
    Y       : (n, 3) - three factor equations solved simultaneously
    weights : (n,) optional - exponential weights; normalised internally

    Returns
    -------
    coef : (n_features, 3) coefficient matrix
    """
    from numpy.linalg import lstsq
    if weights is not None:
        w = weights / weights.sum() * len(weights)
        W = np.sqrt(w[:, None])
        X, Y = X * W, Y * W
    coef, _, _, _ = lstsq(X, Y, rcond=None)
    return coef
