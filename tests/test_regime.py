"""Tests for src/regime.py — fit_hmm, label_states, and attach_regimes."""

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM
from src.regime import fit_hmm, label_states, attach_regimes


@pytest.fixture
def mock_returns():
    """Create a mock S&P 500 returns series spanning 36 months."""
    dates = pd.date_range(start="2020-01-01", periods=36, freq="MS")
    # Simulate three distinct regimes:
    # Bull: low volatility, positive mean
    # Sideways: low volatility, zero mean
    # Bear: high volatility, negative mean
    np.random.seed(42)
    bull_ret = np.random.normal(0.02, 0.01, 12)
    side_ret = np.random.normal(0.00, 0.01, 12)
    bear_ret = np.random.normal(-0.04, 0.05, 12)
    
    returns = np.concatenate([bull_ret, side_ret, bear_ret])
    return pd.DataFrame({"monthly_return": returns}, index=dates)


def test_fit_hmm(mock_returns):
    model, state_sequence = fit_hmm(mock_returns, n_states=3, n_iter=50, random_state=42)
    
    assert isinstance(model, GaussianHMM)
    assert isinstance(state_sequence, np.ndarray)
    assert len(state_sequence) == len(mock_returns)
    assert set(state_sequence).issubset({0, 1, 2})


def test_label_states(mock_returns):
    # Artificially assign state sequences to ensure deterministic ordering of means
    # State 0: low return (bear)
    # State 1: high return (bull)
    # State 2: medium return (sideways)
    state_sequence = np.array([1]*12 + [2]*12 + [0]*12)
    
    regime_df = label_states(mock_returns, state_sequence, n_states=3)
    
    assert isinstance(regime_df, pd.DataFrame)
    assert "regime" in regime_df.columns
    assert "state_raw" in regime_df.columns
    
    # State 1 has the highest return (from bull_ret in mock_returns) -> bull
    # State 2 has medium return -> sideways
    # State 0 has the lowest return -> bear
    assert regime_df.loc[regime_df["state_raw"] == 1, "regime"].iloc[0] == "bull"
    assert regime_df.loc[regime_df["state_raw"] == 2, "regime"].iloc[0] == "sideways"
    assert regime_df.loc[regime_df["state_raw"] == 0, "regime"].iloc[0] == "bear"


def test_attach_regimes():
    panel = pd.DataFrame({
        "act_symbol": ["AAPL", "JPM"],
        "period_end_date": pd.to_datetime(["2020-03-15", "2020-06-20"]),
    })
    
    regime_df = pd.DataFrame({
        "regime": ["bear", "sideways"],
        "year": [2020, 2020],
        "month": [3, 6],
    })

    result = attach_regimes(panel, regime_df)
    
    assert isinstance(result, pd.DataFrame)
    assert "regime" in result.columns
    assert result.loc[0, "regime"] == "bear"
    assert result.loc[1, "regime"] == "sideways"
