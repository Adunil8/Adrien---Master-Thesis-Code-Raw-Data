"""
run_all.py - Regenerate the full pipeline from raw data, in order.

Every script in PIPELINE below produces a table, figure, or number that the
thesis (main.qmd) actually cites. Run them in this order and every .parquet,
.pkl, and .png under data/processed/ and report/figures/ will match what the
thesis reports. The order matters: several scripts read another script's
output (e.g. 02c must run before 02, since 02 reads 02c's lambda-selection
result; 13c must run before 14b, which reads 13c's pre-period probabilities).
Running out of order will fail loudly (a missing-file error), not silently
produce wrong numbers.

Usage:
  python code/run_all.py                    # run everything from the start
  python code/run_all.py --start-at 05_models.py
                                              # skip everything before this
                                              # script (its inputs must
                                              # already exist on disk)
"""

import sys
import os
import subprocess
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable

PIPELINE = [
    "01_data_cleaning.py",
    "01b_download_macro.py",
    "01c_cot_data.py",
    "01d_lme_cash_cleaning.py",
    # 02c must run before 02: it produces lambda_classification_validation.parquet,
    # which 02_nelson_siegel.py reads to select its fallback lambda.
    "02c_lambda_classification_validation.py",
    "02_nelson_siegel.py",
    "02b_ns_raw.py",
    "02b_ns_log.py",
    "02b_sv_raw.py",
    "02b_sv_log.py",
    "02b_model_comparison.py",
    "02d_ns_fit_example.py",
    "03_stationarity_lags.py",
    "03b_lag_robustness.py",
    "04_granger.py",
    "04b_macro_screening.py",
    "04c_multi_horizon_granger.py",
    "00b_curve_and_macro_description.py",
    "05_models.py",
    "06_diagnostics.py",
    # 07b before 07: 07 optionally reads 07b's data-optimal lambda_EW and
    # falls back to the config default if that file is missing.
    "07b_lambda_robustness.py",
    "07_forecast_evaluation.py",
    "08_probability_outputs.py",
    "13_bootstrap_probability_outputs.py",
    # 15 must run before 13f: 13f reads calibration_diagnostics.parquet
    # (15's output) as the production-AUC baseline it compares its
    # walk-forward re-test against. 15 itself only needs 13's output, so
    # this ordering has no downstream cost.
    "15_calibration_diagnostics.py",
    "13c_calibration_preperiod.py",
    "13d_walkforward_stability_diagnostic.py",
    "13e_walkforward_residual_pool.py",
    "13f_walkforward_probability_and_auc.py",
    "14_platt_calibration.py",
    # 14b reads 13c's pre-period output directly; 14c reads 13f's and 14's output.
    "14b_platt_calibration_rolling.py",
    "14c_platt_on_walkforward_base.py",
    # 09c must run before 09 and 09d: both read 09c's own output
    # (deal_rolldown_probabilities_walkforward.parquet and
    # deal_rolldown_calibrated_walkforward.parquet).
    "09c_deal_rolldown_walkforward.py",
    "09_stylised_deal.py",
    "09a_deal_rolldown_probability.py",
    "09d_deal_platt_rolling.py",
    "09e_ch5_constant_maturity_trailing_vol.py",
    "10_granger_summary_table.py",
    "10_subperiod_analysis.py",
    "12_bootstrap_vs_gaussian_test.py",
]


def run_script(name: str) -> bool:
    path = os.path.join(CODE_DIR, name)
    if not os.path.exists(path):
        print(f"  [SKIP] {name} not found")
        return True
    print(f"\n{'='*70}\nRunning {name}\n{'='*70}")
    t0 = time.time()
    result = subprocess.run([PYTHON, path], cwd=os.path.dirname(CODE_DIR))
    dt = time.time() - t0
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    print(f"--- {name}: {status} ({dt:.1f}s) ---")
    return ok


def main():
    scripts = PIPELINE
    if "--start-at" in sys.argv:
        start_name = sys.argv[sys.argv.index("--start-at") + 1]
        if start_name not in PIPELINE:
            print(f"[ERROR] {start_name} is not in PIPELINE")
            sys.exit(1)
        scripts = PIPELINE[PIPELINE.index(start_name):]
        print(f"Resuming from {start_name} ({len(scripts)}/{len(PIPELINE)} scripts). "
              f"Everything before it must already have its output on disk.")
    else:
        print(f"Regenerating the full pipeline ({len(PIPELINE)} scripts)")

    failed = []
    for name in scripts:
        if not run_script(name):
            failed.append(name)
            print(f"\n[STOPPING] {name} failed - fix before continuing "
                  f"(downstream scripts depend on its output).")
            break

    print("\n" + "=" * 70)
    if failed:
        print(f"PIPELINE INCOMPLETE - stopped at: {failed[0]}")
        sys.exit(1)
    else:
        print("PIPELINE COMPLETE - all scripts ran successfully.")
        print("Every .parquet/.pkl/.png under data/processed/ and report/figures/")
        print("is now consistent with the current code and config.py.")


if __name__ == "__main__":
    main()
