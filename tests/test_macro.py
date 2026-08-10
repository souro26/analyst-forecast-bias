"""Tests for src/macro.py — build_macro_table, merge_onto_panel, compute_pit_macro, and future leakage check."""

import numpy as np
import pandas as pd
import pytest
from src.macro import build_macro_table, merge_onto_panel, compute_pit_macro


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


def test_compute_pit_macro_no_future_leakage():
    # Setup daily indices
    dates = pd.date_range(start="2020-01-01", end="2020-06-01", freq="D")
    
    # Generate linear mock price series for GSPC and constant VIX
    gspc_close = pd.Series(np.arange(100, 100 + len(dates)), index=dates)
    vix_close = pd.Series([20.0] * len(dates), index=dates)

    # Panel observation with prediction_cutoff as May 1st 2020
    panel = pd.DataFrame({
        "act_symbol": ["AAPL"],
        "prediction_cutoff": pd.to_datetime(["2020-05-01"])
    })

    # Compute baseline PIT features
    panel_pit = compute_pit_macro(panel, gspc_close, vix_close)
    sp500_ret_base = panel_pit["sp500_return_pit"].values[0]
    vix_mean_base = panel_pit["vix_mean_pit"].values[0]

    # Add artificial future macro observations (e.g. huge spike on May 2nd)
    future_dates = pd.date_range(start="2020-05-02", end="2020-06-30", freq="D")
    gspc_future = pd.Series([500.0] * len(future_dates), index=future_dates)
    vix_future = pd.Series([100.0] * len(future_dates), index=future_dates)

    gspc_leaked = pd.concat([gspc_close, gspc_future])
    vix_leaked = pd.concat([vix_close, vix_future])

    # Re-compute PIT features using leaked series
    panel_pit_leaked = compute_pit_macro(panel, gspc_leaked, vix_leaked)
    sp500_ret_leaked = panel_pit_leaked["sp500_return_pit"].values[0]
    vix_mean_leaked = panel_pit_leaked["vix_mean_pit"].values[0]

    # PIT features must be identical! Future data should not affect them.
    assert np.isclose(sp500_ret_base, sp500_ret_leaked)
    assert np.isclose(vix_mean_base, vix_mean_leaked)
    assert not np.isnan(sp500_ret_base)
    assert not np.isnan(vix_mean_base)
