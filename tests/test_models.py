"""Tests for src/models.py — load_and_prepare, build_model, sample_model, and extract_results."""

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
        "year": [2020, 2020, 2020],
        "category": ["tech_cycle", "macro_rate_sensitive", "commodity_driven"],
        "forecast_error_winsorized": [0.1, 0.2, -0.05],
        "sp500_return_z": [0.5, 0.5, 0.5],
        "vix_mean_z": [-0.2, -0.2, -0.2],
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
    # 2026 is filtered out, but our mock doesn't have it anyway
    assert len(df) == 3
    # Check category index values are codes pointing to CATEGORY_ORDER
    assert df.loc[df["act_symbol"] == "AAPL", "category_idx"].values[0] == CATEGORY_ORDER.index("tech_cycle")


def test_load_and_prepare_missing_macro_columns(tmp_path):
    panel_df = pd.DataFrame({
        "act_symbol": ["AAPL"],
        "period_end_date": pd.to_datetime(["2020-03-31"]),
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

    with pytest.raises(ValueError, match="Column 'sp500_return_z' not found in panel"):
        load_and_prepare(panel_path, features_path)


def test_build_model():
    # Make a tiny mock DataFrame with all required columns
    df = pd.DataFrame({
        "category_idx": [0, 1, 2],
        "sp500_return_z": [0.1, -0.2, 0.3],
        "vix_mean_z": [-0.5, 0.2, 0.1],
        "forecast_error_winsorized": [0.05, 0.15, -0.02],
    })

    model = build_model(df)
    assert isinstance(model, pm.Model)
    
    # Check variables exist in the model
    varnames = [v.name for v in model.value_vars + model.free_RVs]
    assert any("mu_global" in name for name in varnames)
    assert any("alpha_category" in name for name in varnames)
    assert any("beta_sp500" in name for name in varnames)
    assert any("beta_vix" in name for name in varnames)
    assert any("nu" in name for name in varnames)
    assert any("sigma" in name for name in varnames)


def test_sample_model_and_diagnostics():
    df = pd.DataFrame({
        "category_idx": [0, 1, 2, 0, 1, 2],
        "sp500_return_z": [0.1, -0.2, 0.3, 0.1, -0.2, 0.3],
        "vix_mean_z": [-0.5, 0.2, 0.1, -0.5, 0.2, 0.1],
        "forecast_error_winsorized": [0.05, 0.15, -0.02, 0.04, 0.16, -0.03],
    })

    model = build_model(df)
    # Use very small draws/tune/chains to run fast in testing
    trace = sample_model(model, draws=5, tune=5, chains=2, target_accept=0.8, random_seed=123)

    assert isinstance(trace, az.InferenceData)
    
    # Test check_convergence
    # Note: with only 5 draws/tune, check_convergence might return True or False depending on diagnostics,
    # but it should execute without raising an error.
    converged = check_convergence(trace)
    assert isinstance(converged, bool)

    # Test extract_results
    results = extract_results(trace)
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert "parameter" in results.columns
    assert "mean" in results.columns
    assert "hdi_3%" in results.columns
