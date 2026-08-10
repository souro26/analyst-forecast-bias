"""Tests for src/macro.py — build_macro_table and merge_onto_panel."""

import numpy as np
import pandas as pd
import pytest
from src.macro import build_macro_table, merge_onto_panel


@pytest.fixture
def mock_sp500_returns():
    """Mock S&P 500 quarterly returns DataFrame."""
    dates = pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"])
    return pd.DataFrame(
        {"sp500_return": [0.05, -0.02, 0.08, 0.03]},
        index=pd.Index(dates, name="quarter_end")
    )


@pytest.fixture
def mock_vix_means():
    """Mock VIX quarterly means DataFrame."""
    dates = pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"])
    return pd.DataFrame(
        {"vix_mean": [25.0, 30.0, 20.0, 18.0]},
        index=pd.Index(dates, name="quarter_end")
    )


def test_build_macro_table(mock_sp500_returns, mock_vix_means):
    macro = build_macro_table(mock_sp500_returns, mock_vix_means)

    assert isinstance(macro, pd.DataFrame)
    assert len(macro) == 4
    assert "sp500_return_z" in macro.columns
    assert "vix_mean_z" in macro.columns

    # Z-scores should have mean ~0 and standard deviation ~1
    np.testing.assert_almost_equal(macro["sp500_return_z"].mean(), 0.0, decimal=7)
    np.testing.assert_almost_equal(macro["vix_mean_z"].mean(), 0.0, decimal=7)
    np.testing.assert_almost_equal(macro["sp500_return_z"].std(ddof=1), 1.0, decimal=7)
    np.testing.assert_almost_equal(macro["vix_mean_z"].std(ddof=1), 1.0, decimal=7)


def test_merge_onto_panel(mock_sp500_returns, mock_vix_means):
    macro = build_macro_table(mock_sp500_returns, mock_vix_means)
    
    # Create a mock panel of company-quarter observations
    panel = pd.DataFrame({
        "act_symbol": ["AAPL", "JPM", "AAPL", "JPM"],
        "period_end_date": pd.to_datetime(["2020-03-15", "2020-03-20", "2020-06-15", "2020-06-25"]),
    })

    merged = merge_onto_panel(panel, macro)

    assert isinstance(merged, pd.DataFrame)
    assert len(merged) == 4
    assert "sp500_return" in merged.columns
    assert "vix_mean" in merged.columns
    assert "sp500_return_z" in merged.columns
    assert "vix_mean_z" in merged.columns

    # direction="nearest" check:
    # 2020-03-15 should merge with nearest quarter_end (2020-03-31)
    aapl_q1 = merged[(merged["act_symbol"] == "AAPL") & (merged["period_end_date"] == "2020-03-15")]
    assert aapl_q1["sp500_return"].values[0] == 0.05
    assert aapl_q1["vix_mean"].values[0] == 25.0

    # 2020-06-25 should merge with nearest quarter_end (2020-06-30)
    jpm_q2 = merged[(merged["act_symbol"] == "JPM") & (merged["period_end_date"] == "2020-06-25")]
    assert jpm_q2["sp500_return"].values[0] == -0.02
    assert jpm_q2["vix_mean"].values[0] == 30.0
