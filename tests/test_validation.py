"""Tests for src/validation.py"""

import pytest
import pandas as pd
import numpy as np
from src.validation import (
    validate_prediction_cutoff,
    validate_point_in_time_features,
    validate_train_test_temporal_order
)

def test_validate_prediction_cutoff_valid():
    df = pd.DataFrame({
        "act_symbol": ["AAPL", "MSFT"],
        "period_end_date": pd.to_datetime(["2020-03-31", "2020-06-30"]),
        "prediction_cutoff": pd.to_datetime(["2020-03-31", "2020-06-30"])
    })
    validate_prediction_cutoff(df)  # should pass without raising

def test_validate_prediction_cutoff_missing_column():
    df = pd.DataFrame({
        "act_symbol": ["AAPL"],
        "period_end_date": pd.to_datetime(["2020-03-31"])
    })
    with pytest.raises(ValueError, match="Required column 'prediction_cutoff' is missing."):
        validate_prediction_cutoff(df)

def test_validate_prediction_cutoff_null_value():
    df = pd.DataFrame({
        "act_symbol": ["AAPL"],
        "period_end_date": [pd.NaT],
        "prediction_cutoff": pd.to_datetime(["2020-03-31"])
    })
    with pytest.raises(ValueError, match="contains 1 null value"):
        validate_prediction_cutoff(df)

def test_validate_prediction_cutoff_mismatch():
    df = pd.DataFrame({
        "act_symbol": ["AAPL"],
        "period_end_date": pd.to_datetime(["2020-03-31"]),
        "prediction_cutoff": pd.to_datetime(["2020-04-01"])
    })
    with pytest.raises(ValueError, match="Prediction cutoff must equal period_end_date"):
        validate_prediction_cutoff(df)


def test_validate_point_in_time_features_valid():
    df = pd.DataFrame({
        "act_symbol": ["AAPL", "AAPL"],
        "date": pd.to_datetime(["2020-03-29", "2020-03-30"]),
        "prediction_cutoff": pd.to_datetime(["2020-03-31", "2020-03-31"])
    })
    validate_point_in_time_features(df, "date", "prediction_cutoff")

def test_validate_point_in_time_features_future():
    df = pd.DataFrame({
        "act_symbol": ["AAPL"],
        "date": pd.to_datetime(["2020-04-01"]),
        "prediction_cutoff": pd.to_datetime(["2020-03-31"])
    })
    with pytest.raises(ValueError, match="TEMPORAL LEAKAGE DETECTED"):
        validate_point_in_time_features(df, "date", "prediction_cutoff")

def test_validate_point_in_time_features_exact_cutoff():
    df = pd.DataFrame({
        "act_symbol": ["AAPL"],
        "date": pd.to_datetime(["2020-03-31"]),
        "prediction_cutoff": pd.to_datetime(["2020-03-31"])
    })
    with pytest.raises(ValueError, match="TEMPORAL LEAKAGE DETECTED"):
        validate_point_in_time_features(df, "date", "prediction_cutoff")


def test_validate_train_test_temporal_order_valid():
    train = pd.DataFrame({"prediction_cutoff": pd.to_datetime(["2019-12-31"])})
    val = pd.DataFrame({"prediction_cutoff": pd.to_datetime(["2020-03-31"])})
    test = pd.DataFrame({"prediction_cutoff": pd.to_datetime(["2020-06-30"])})
    validate_train_test_temporal_order(train, val, test)

def test_validate_train_test_temporal_order_violation():
    train = pd.DataFrame({"prediction_cutoff": pd.to_datetime(["2020-03-31"])})
    val = pd.DataFrame({"prediction_cutoff": pd.to_datetime(["2020-02-28"])})
    test = pd.DataFrame({"prediction_cutoff": pd.to_datetime(["2020-06-30"])})
    with pytest.raises(ValueError, match="Temporal order violated"):
        validate_train_test_temporal_order(train, val, test)
