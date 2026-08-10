"""Tests for src/features.py — compute_slope, compute_features, and build_features."""

import numpy as np
import pandas as pd
import pytest
from src.features import compute_slope, compute_features, build_features


def test_compute_slope_basic():
    # Linear sequence: y = 2x + 1
    x = np.arange(5, dtype=float)
    y = 2.0 * x + 1.0
    slope = compute_slope(x, y)
    assert np.isclose(slope, 2.0)


def test_compute_slope_flat():
    x = np.arange(5, dtype=float)
    y = np.array([3.0] * 5)
    slope = compute_slope(x, y)
    assert np.isclose(slope, 0.0)


def test_compute_slope_insufficient_data():
    # Length < 2
    assert np.isnan(compute_slope(np.array([1.0]), np.array([2.0])))
    # Denominator zero
    assert np.isnan(compute_slope(np.array([1.0, 1.0]), np.array([2.0, 3.0])))


def test_compute_features_shape():
    # Setup mock data for a single ticker-quarter (8 weeks of snapshots)
    dates = pd.date_range(end="2020-03-31", periods=8, freq="W")
    group = pd.DataFrame({
        "date": dates,
        "consensus": [1.0, 1.02, 1.04, 1.03, 1.05, 1.07, 1.06, 1.10],
        "consensus_spread": [0.05, 0.04, 0.06, 0.05, 0.04, 0.03, 0.04, 0.02],
        "count": [10, 10, 11, 11, 12, 12, 12, 13],
    })

    feats = compute_features(group)

    assert isinstance(feats, dict)
    assert feats["weeks_of_data"] == 8
    assert feats["revision_slope"] > 0  # Since consensus generally trends up
    assert not np.isnan(feats["revision_acceleration"])
    assert not np.isnan(feats["direction_changes"])
    assert feats["direction_changes"] >= 0
    assert not np.isnan(feats["final_vs_initial"])
    assert feats["spread_trend"] < 0  # Disagreement tends to go down close to date
    assert feats["analyst_count_trend"] > 0


def test_build_features():
    # Setup mock estimate DataFrame
    dates = pd.date_range(end="2020-03-31", periods=10, freq="W")
    estimate = pd.DataFrame({
        "act_symbol": ["AAPL"] * 5 + ["JPM"] * 5,
        "period_end_date": pd.to_datetime(["2020-03-31"] * 10),
        "period": ["Current Quarter"] * 10,
        "date": list(dates[:5]) + list(dates[5:]),
        "consensus": [2.5, 2.55, 2.6, 2.62, 2.65, 1.2, 1.18, 1.15, 1.10, 1.05],
        "consensus_spread": [0.1] * 10,
        "count": [15] * 10,
    })

    features_df = build_features(estimate)

    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) == 2
    assert "act_symbol" in features_df.columns
    assert "period_end_date" in features_df.columns
    assert "revision_slope" in features_df.columns

    # AAPL has positive revision_slope, JPM has negative revision_slope
    aapl_slope = features_df.loc[features_df["act_symbol"] == "AAPL", "revision_slope"].values[0]
    jpm_slope = features_df.loc[features_df["act_symbol"] == "JPM", "revision_slope"].values[0]
    
    assert aapl_slope > 0
    assert jpm_slope < 0
