import numpy as np
import pandas as pd
import pytest
from src.features import (
    compute_slope,
    compute_features,
    build_features,
    restrict_to_prediction_events,
    validate_features
)


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

    cutoff = pd.Timestamp("2020-04-07")
    feats = compute_features(group, cutoff)

    assert isinstance(feats, dict)
    assert feats["weeks_of_data"] == 8
    assert feats["revision_slope"] > 0  # Since consensus generally trends up
    assert not np.isnan(feats["revision_acceleration"])
    assert not np.isnan(feats["direction_changes"])
    assert feats["direction_changes"] >= 0
    assert not np.isnan(feats["final_vs_initial"])
    assert feats["spread_trend"] < 0  # Disagreement tends to go down close to date
    assert feats["analyst_count_trend"] > 0


def test_build_features_pit_boundaries():
    # Setup mock estimate DataFrame with edge case dates around cutoff
    cutoff = pd.Timestamp("2020-03-31")
    
    # 16 weeks window: [2019-12-10, 2020-03-31)
    # 1. 2019-12-09: Outside 16-week window (too old) -> Exclude
    # 2. 2019-12-10: Exactly 112 days before cutoff -> Include
    # 3. 2020-01-15: Inside window -> Include
    # 4. 2020-03-30: Latest valid snapshot strictly before cutoff -> Include
    # 5. 2020-03-31: Exactly at cutoff -> Exclude
    # 6. 2020-04-01: After cutoff -> Exclude
    
    dates = pd.to_datetime([
        "2019-12-09",
        "2019-12-10",
        "2020-01-15",
        "2020-03-30",
        "2020-03-31",
        "2020-04-01"
    ])
    
    # Create estimate panel
    estimate = pd.DataFrame({
        "act_symbol": ["AAPL"] * 6,
        "period_end_date": [cutoff] * 6,
        "prediction_cutoff": [cutoff] * 6,
        "period": ["Current Quarter"] * 6,
        "date": dates,
        "consensus": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        "consensus_spread": [0.05] * 6,
        "count": [10] * 6,
    })

    # Filter mock dataframe to mimic features.py's internal behavior and check
    # that build_features output uses exactly the expected snapshots
    features_df = build_features(estimate)
    
    # Check that output exists
    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) == 1
    
    # Let's inspect final_vs_initial calculation:
    # Included snapshots:
    # 2019-12-10 (consensus=1.1)
    # 2020-01-15 (consensus=1.2)
    # 2020-03-30 (consensus=1.3)
    # Total weeks of data should be 3
    assert features_df.loc[0, "weeks_of_data"] == 3
    
    # 8-week limit: cutoff - 56 days = 2020-02-04
    # Latest snapshot on or before 2020-02-04 is 2020-01-15 (consensus=1.2)
    # final_consensus = 1.3 (at 2020-03-30)
    # final_vs_initial = 1.3 - 1.2 = 0.1
    assert np.isclose(features_df.loc[0, "final_vs_initial"], 0.1)


def test_build_features_sparse_and_duplicates():
    cutoff = pd.Timestamp("2020-03-31")
    
    # Duplicate dates and missing reference checks
    dates = pd.to_datetime([
        "2020-03-01",
        "2020-03-01",  # Duplicate date
        "2020-03-25"
    ])
    
    estimate = pd.DataFrame({
        "act_symbol": ["AAPL"] * 3,
        "period_end_date": [cutoff] * 3,
        "prediction_cutoff": [cutoff] * 3,
        "period": ["Current Quarter"] * 3,
        "date": dates,
        "consensus": [1.0, 1.1, 1.2],  # Duplicate on 2020-03-01: consensus should be 1.1 (latest)
        "consensus_spread": [0.05] * 3,
        "count": [10] * 3,
    })

    features_df = build_features(estimate)
    assert len(features_df) == 1
    # Duplicate resolved, weeks_of_data should be 2
    assert features_df.loc[0, "weeks_of_data"] == 2
    # No snapshot exists on or before 2020-02-04 (cutoff - 56 days), so final_vs_initial must be NaN
    assert np.isnan(features_df.loc[0, "final_vs_initial"])


def test_restrict_and_validate_features():
    panel = pd.DataFrame({
        "act_symbol": ["AAPL", "JPM"],
        "period_end_date": pd.to_datetime(["2020-03-31", "2020-03-31"]),
    })

    features = pd.DataFrame({
        "act_symbol": ["AAPL", "JPM", "XOM"],  # XOM is orphan
        "period_end_date": pd.to_datetime(["2020-03-31", "2020-03-31", "2020-03-31"]),
        "revision_slope": [0.1, -0.2, 0.05],
        "revision_acceleration": [0.01, -0.01, 0.0],
        "direction_changes": [1, 2, 0],
        "final_vs_initial": [0.05, -0.1, 0.02],
        "spread_trend": [-0.01, -0.02, -0.01],
        "analyst_count_trend": [0.1, 0.0, 0.1],
    })

    # Validate fails before restriction because of XOM orphan
    with pytest.raises(ValueError, match="Found 1 feature events without corresponding"):
        validate_features(features, panel)

    # Restrict to prediction events
    restricted = restrict_to_prediction_events(features, panel)
    assert len(restricted) == 2
    assert "XOM" not in restricted["act_symbol"].values

    # Validate passes after restriction
    validate_features(restricted, panel)
