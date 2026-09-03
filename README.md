# Forecasting LME Copper Futures Curve Dynamics

Code and data for the Master's thesis *"Forecasting LME Copper Futures Curve
Dynamics with a Macro-Augmented Dynamic Nelson-Siegel Model: Probabilistic
Price Signals for Trade Finance Risk Management"* (HEC Lausanne, MScF).

This repository contains everything needed to regenerate every table, figure,
and number reported in the thesis from the raw data.

## What's here

```
config.py          All parameters (dates, thresholds, macro variable spec).
                    Change a value here, re-run the pipeline, everything
                    downstream updates.
code/               The full pipeline, numbered by the phase it belongs to
                    (01 = data cleaning, 02 = Nelson-Siegel fit, 03 = factor
                    stationarity and lag selection, 04 = Granger causality,
                    05-08 = model estimation and forecast evaluation,
                    09 = the stylised trade finance deal, 12-15 = probability
                    calibration). run_all.py runs the whole chain in order.
data/raw/           Source data: Bloomberg exports, FRED macro series, CFTC
                    positioning reports, GPR index, LME cash settlement
                    prices.
data/processed/     Created by the pipeline. Empty until you run it.
```

## Data sources

- **LME copper futures curve (LP1-LP6) and LME on-warrant inventory**:
  Bloomberg Terminal exports. Included here for grading purposes.
- **FRED series** (DXY, VIX, US 3-month yield, and others): freely
  downloadable from the Federal Reserve Economic Data API.
- **CFTC Commitments of Traders reports**: public annual bulk files from
  cftc.gov.
- **Geopolitical Risk index**: Caldara & Iacoviello (2022), free download
  from policyuncertainty.com.
- **LME copper cash settlement prices**: free daily series from
  westmetall.com.

## Setup

Developed and verified end to end with Python 3.12. Package versions are
pinned in `requirements.txt` to the exact versions this pipeline was tested
with - newer releases of some of these (statsmodels in particular) have
changed function signatures in ways that break older code, so installing
unpinned latest versions is not guaranteed to work.

```bash
pip install -r requirements.txt
```

## Running the pipeline

From the repository root:

```bash
python code/run_all.py
```

This runs every script in `code/` in the order the pipeline requires (some
scripts read another script's output, so order matters) and writes every
intermediate `.parquet`/`.pkl` file to `data/processed/` and every figure to
`report/figures/`. A full run takes a few minutes.

If a script fails, `run_all.py` stops immediately and names the script that
failed rather than continuing with stale downstream output.

## Reproducing a specific result

Every script's docstring states which section, table, or figure of the
thesis it produces. To regenerate a single result rather than the whole
pipeline, find the relevant script and run it directly, e.g.:

```bash
python code/05_models.py          # the four forecasting models (Section 3.5)
python code/09_stylised_deal.py   # the stylised deal application (Chapter 6)
```

Most scripts depend on earlier ones having already been run at least once
(they read `data/processed/*.parquet` files that earlier scripts produce),
so the first run should be the full `run_all.py`.
