"""Tests for src/models.py — load_and_prepare, build_model, build_time_effect_model, sample_model, and extract_results."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytest

from src.models import (
    load_and_prepare,
    build_model,
    build_time_effect_model,
    sample_model,
    check_convergence,
    extract_results,
    CATEGORY_ORDER,
)


@pytest.fixture
def mock_data_paths(tmp_path):
    """Create mock parquet files for panel and features."""
    panel_df = pd.DataFrame({
        "act_symbol": ["AAPL", "JPM", "XOM"],
        "period_end_date": pd.to_datetime(["2020-03-31", "2020-03-31", "2020-03-31"]),
        "prediction_cutoff": pd.to_datetime(["2020-03-31", "2020-03-31", "2020-03-31"]),
        "year": [2020, 2020, 2020],
        "fiscal_quarter": [1, 1, 1],
        "category": ["tech_cycle", "macro_rate_sensitive", "commodity_driven"],
        "forecast_error_winsorized": [0.1, 0.2, -0.05],
        "normalized_error_winsorized": [0.05, 0.10, -0.02],
        "sp500_return": [0.05, 0.02, -0.01],
        "vix_mean": [15.0, 18.0, 22.0],
        "sp500_return_pit": [0.05, 0.02, -0.01],
        "vix_mean_pit": [15.0, 18.0, 22.0],
    })
    
    features_df = pd.DataFrame({
        "act_symbol": ["AAPL", "JPM", "XOM"],
        "period_end_date": pd.to_datetime(["2020-03-31", "2020-03-31", "2020-03-31"]),
        "revision_slope": [0.01, -0.02, 0.00],
    })

    panel_path = tmp_path / "panel_macro.parquet"
    features_path = tmp_path / "features.parquet"

    panel_df.to_parquet(panel_path, index=False)
    features_df.to_parquet(features_path, index=False)

    return panel_path, features_path


def test_load_and_prepare(mock_data_paths):
    panel_path, features_path = mock_data_paths
    df = load_and_prepare(panel_path, features_path)

    assert isinstance(df, pd.DataFrame)
    assert "category_idx" in df.columns
    assert "ticker_idx" in df.columns
    assert "quarter_idx" in df.columns
    assert len(df) == 3
    assert df.loc[df["act_symbol"] == "AAPL", "category_idx"].values[0] == CATEGORY_ORDER.index("tech_cycle")


def test_load_and_prepare_missing_macro_columns(tmp_path):
    panel_df = pd.DataFrame({
        "act_symbol": ["AAPL"],
        "period_end_date": pd.to_datetime(["2020-03-31"]),
        "prediction_cutoff": pd.to_datetime(["2020-03-31"]),
        "year": [2020],
        "category": ["tech_cycle"],
        "forecast_error_winsorized": [0.1],
    })
    features_df = pd.DataFrame({
        "act_symbol": ["AAPL"],
        "period_end_date": pd.to_datetime(["2020-03-31"]),
    })

    panel_path = tmp_path / "panel.parquet"
    features_path = tmp_path / "features.parquet"
    panel_df.to_parquet(panel_path, index=False)
    features_df.to_parquet(features_path, index=False)

    with pytest.raises(ValueError, match="Column 'sp500_return' not found in panel"):
        load_and_prepare(panel_path, features_path)


def test_build_model_raw():
    # Make a tiny mock DataFrame with all required columns
    df = pd.DataFrame({
        "act_symbol": ["AAPL", "JPM", "XOM"],
        "category_idx": [0, 1, 2],
        "ticker_idx": [0, 1, 2],
        "sp500_pit_scaled": [0.1, -0.2, 0.3],
        "vix_pit_scaled": [-0.5, 0.2, 0.1],
        "forecast_error_winsorized": [0.05, 0.15, -0.02],
    })

    model = build_model(df, "forecast_error_winsorized")
    assert isinstance(model, pm.Model)
    
    # Check variables exist in the model
    varnames = [v.name for v in model.value_vars + model.free_RVs + model.deterministics]
    assert any("mu_global" in name for name in varnames)
    assert any("alpha_ticker_offset" in name for name in varnames)
    assert any("beta_sp500" in name for name in varnames)
    assert any("beta_vix" in name for name in varnames)
    assert any("sigma" in name for name in varnames)


def test_build_time_effect_model():
    # Make a tiny mock DataFrame with all required columns
    df = pd.DataFrame({
        "act_symbol": ["AAPL", "JPM", "XOM"],
        "category_idx": [0, 1, 2],
        "ticker_idx": [0, 1, 2],
        "quarter_idx": [0, 0, 0],
        "forecast_error_winsorized": [0.05, 0.15, -0.02],
    })

    model = build_time_effect_model(df, "forecast_error_winsorized")
    assert isinstance(model, pm.Model)
    
    # Check variables exist in the model
    varnames = [v.name for v in model.value_vars + model.free_RVs + model.deterministics]
    assert any("mu_global" in name for name in varnames)
    assert any("alpha_ticker_offset" in name for name in varnames)
    assert any("alpha_quarter_offset" in name for name in varnames)
    assert not any("beta_sp500" in name for name in varnames)
    assert any("sigma" in name for name in varnames)


def test_sample_model_and_diagnostics():
    df = pd.DataFrame({
        "act_symbol": ["AAPL", "JPM", "XOM", "AAPL", "JPM", "XOM"],
        "category_idx": [0, 1, 2, 0, 1, 2],
        "ticker_idx": [0, 1, 2, 0, 1, 2],
        "sp500_pit_scaled": [0.1, -0.2, 0.3, 0.1, -0.2, 0.3],
        "vix_pit_scaled": [-0.5, 0.2, 0.1, -0.5, 0.2, 0.1],
        "forecast_error_winsorized": [0.05, 0.15, -0.02, 0.04, 0.16, -0.03],
    })

    model = build_model(df, "forecast_error_winsorized")
    # Use very small draws/tune/chains to run fast in testing
    trace = sample_model(model, draws=5, tune=5, chains=2, target_accept=0.8, random_seed=123)

    assert isinstance(trace, az.InferenceData)
    
    converged = check_convergence(trace)
    assert isinstance(converged, bool)

    results = extract_results(trace, "Raw error")
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert "parameter" in results.columns
    assert "mean" in results.columns
    assert "hdi_3%" in results.columns
