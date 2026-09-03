"""
02d_ns_fit_example.py - NS(log) cross-sectional fit example, at the actual
production decay rate lambda*=1.4677, on the actual production factors.

WHY A SEPARATE SCRIPT: 02b_ns_log.py and 02b_ns_raw.py already produce a
three-panel fit-example figure each (fig_nslog_fit_example.png,
fig_nsraw_fit_example.png), but both come from a separate robustness
exercise (Appendix IV, A4.3) that grid-searches its own RMSE-minimising
decay rate independently for each of four configurations (NS raw, NS log,
Svensson raw, Svensson log), to compare them on equal footing. That RMSE-
optimal lambda for NS(log) (7.5226) is NOT the lambda the actual production
model uses. The production model's decay rate, lambda*=1.4677, is instead
selected by matching backwardation/contango classification against the
observed front-vs-6M relationship (02c_lambda_classification_validation.py,
Section 3.2), a different criterion entirely. Presenting a figure fit at
lambda=7.52 as "the production model" would misrepresent what the thesis
actually runs. This script builds the correct figure instead, reading
lambda* and the beta time series directly from the production outputs
(ns_factors.parquet), not re-running any independent grid search.

Uses the same three fixed calendar dates as fig_nsraw_fit_example.png and
fig_nslog_fit_example.png (2006-01-06 backwardation, 2021-08-20 flat,
2024-07-12 contango, picked from the observed near-minus-far price spread,
model-free, not from any single fit's own beta1: see 02b_ns_raw.py's
plot_fit_example docstring for why idxmax/idxmin on a fit-specific beta1
is not used here), so all three figures are directly comparable on
identical dates.

Inputs : data/processed/ns_factors.parquet          (production beta_t, log units, lambda*=1.4677)
         data/processed/curves.parquet               (observed raw prices, m01-m06)
         data/processed/lambda_classification_validation.parquet (source of lambda*)
Outputs: report/figures/fig_ns_production_fit_example.png

Run: python code/02d_ns_fit_example.py
     (from thesis/ root with .venv active)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PROCESSED_DATA_DIR, FIGURES_DIR, MATURITIES_MONTHS
from utils import ns_curve

EXAMPLE_DATES = [
    ("2006-01-06", "Backwardation"),
    ("2021-08-20", "Neutral / Flat"),
    ("2024-07-12", "Contango"),
]


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    factors = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "ns_factors.parquet"))
    curves = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "curves.parquet"))
    lambda_val = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "lambda_classification_validation.parquet"))
    lambda_star = float(lambda_val.loc[lambda_val["agreement_pct"].idxmax(), "lambda"])
    print(f"Production lambda* = {lambda_star:.4f} (from lambda_classification_validation.parquet)")

    tau = np.array(MATURITIES_MONTHS, dtype=float)
    tau_fine = np.linspace(tau.min(), tau.max(), 300)
    cols = [f"m{i:02d}" for i in MATURITIES_MONTHS]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    colors = {"Backwardation": "#16a34a", "Neutral / Flat": "#555555", "Contango": "#dc2626"}

    for ax, (date_str, regime_label) in zip(axes, EXAMPLE_DATES):
        date = pd.Timestamp(date_str)
        betas = factors.loc[date, ["beta0", "beta1", "beta2"]].values
        fitted_log_fine = ns_curve(betas, tau_fine, lambda_star)
        fitted_price_fine = np.exp(fitted_log_fine)
        observed_price = curves.loc[date, cols].values.astype(float)

        color = colors[regime_label]
        ax.plot(tau_fine, fitted_price_fine, color=color, linewidth=2, label="NS(log) fit")
        ax.scatter(tau, observed_price, color="black", zorder=5, label="Observed")
        if regime_label == "Neutral / Flat":
            # Autoscaling on a ~$2/t range makes noise look like curvature.
            # Fixed range keeps this panel visually flat, matching what it is.
            ax.set_ylim(9020, 9080)
            ax.set_yticks([9030, 9040, 9050, 9060, 9070])
        ax.set_title(f"{regime_label}\n{date_str}\n"
                     f"$\\beta_0$={betas[0]:.3f}  $\\beta_1$={betas[1]:.3f}  $\\beta_2$={betas[2]:.3f}",
                     fontsize=9)
        ax.set_xlabel("Maturity (months)")
        ax.grid(lw=0.3, color="#dddddd")
        if ax is axes[0]:
            ax.set_ylabel("Price (USD/tonne)")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"NS(log) Production Fit - $\\lambda^*$={lambda_star:.4f}, Three Example Dates",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_ns_production_fit_example.png")
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
