# 2026-AP-RMFE: Code Record

This repository is the code and analysis record for an Imperial College Business School MSc Applied Project: *Improving Short-Horizon Market-Making with Order Book and Derivatives Signals: A Machine Learning Approach*. It is referenced from the appendix of the submitted report.

## Contents

- `AP_Analysis.ipynb`: the complete, executed analysis notebook. Reprocesses raw data when present, otherwise loads cached per-day outputs, and reproduces every result reported.
- `AP_Analysis.pdf`: a static, page-rendered export of the executed notebook, for browsing without a Jupyter environment.
- `2026-AP-RMFE.pdf`: the submitted report.
- `Figures/`: all figures from the notebook and report.
- `requirements.txt`: pinned package versions used for the executed run.
- `process_ofi.py`, `verify_wide_spreads.py`: the two preprocessing and verification scripts referenced in the report appendix. Their source is also embedded in the notebook.

## Data

Raw data (Databento SPY MBP-1 tick data, OptionMetrics IvyDB US, Cboe VIX) is not included. It is licensed through institutional Databento and WRDS subscriptions and cannot be redistributed. The notebook expects it under `Data/Raw/` and writes derived per-day features to `Data/Processed/`.

## Environment

Pinned versions are listed in `requirements.txt`. XGBoost must stay below version 3.0 (`xgboost<3`), since 3.x changes how `base_score` is serialised and breaks SHAP 0.49.x. Import `torch` before `numpy`/`scikit-learn` to avoid an OpenMP DLL conflict on Windows.
