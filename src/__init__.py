"""
src
---
Pipeline package for the analyst-forecast-bias project.

Seven standalone modules run in order to produce all outputs:

    python src/ingest.py      # Dolt CSV exports -> data/raw/
    python src/clean.py       # data/raw/ -> data/processed/panel.parquet
    python src/macro.py       # yfinance -> panel (adds sp500_return, vix_mean cols)
    python src/regime.py      # hmmlearn HMM -> data/processed/regimes.parquet
    python src/features.py    # consensus snapshots -> data/processed/features.parquet
    python src/models.py      # PyMC NUTS -> models/trace.nc + models/summary.csv
    python src/signals.py     # XGBoost -> models/xgb_model.json + xgb_*.csv

Public API
----------
The functions below are importable directly from the package.
load_and_prepare exists in both models and signals with different
signatures; access those via the module to avoid ambiguity:

    from src.models  import load_and_prepare   # panel + features -> model df
    from src.signals import load_and_prepare   # panel + features -> classifier df
"""

from .ingest import load_config, get_all_tickers, get_ticker_category_map, validate_coverage
from .clean import winsorize_by_category, assign_fiscal_quarter, clean_history, clean_estimate
from .macro import get_sp500_quarterly, get_vix_quarterly, build_macro_table, merge_onto_panel
from .regime import fit_hmm, label_states, attach_regimes
from .features import compute_slope, compute_features, build_features, validate_features
from .models import build_model, sample_model, check_convergence, extract_results
from .signals import split_data, train_model, evaluate, get_feature_importance

__all__ = [
    "load_config",
    "get_all_tickers",
    "get_ticker_category_map",
    "validate_coverage",
    "winsorize_by_category",
    "assign_fiscal_quarter",
    "clean_history",
    "clean_estimate",
    "get_sp500_quarterly",
    "get_vix_quarterly",
    "build_macro_table",
    "merge_onto_panel",
    "fit_hmm",
    "label_states",
    "attach_regimes",
    "compute_slope",
    "compute_features",
    "build_features",
    "validate_features",
    "build_model",
    "sample_model",
    "check_convergence",
    "extract_results",
    "split_data",
    "train_model",
    "evaluate",
    "get_feature_importance",
]
