"""Tests for src/signals.py — SimpleStandardScaler, walk-forward splits, category mapping, and ablations."""

import numpy as np
import pandas as pd
import xgboost as xgb
import pytest
from src.signals import (
    SimpleStandardScaler,
    get_walk_forward_folds,
    CATEGORY_ORDER,
    FEATURE_COLS
)


@pytest.fixture
def mock_prepared_df():
    """Create a mock prepared DataFrame for walk-forward splits and scaling."""
    np.random.seed(42)
    n_rows = 200
    
    # 2017 to 2025
    years = np.random.choice(list(range(2017, 2026)), size=n_rows)
    
    df = pd.DataFrame({
        "act_symbol": np.random.choice(["AAPL", "JPM", "XOM", "GE"], size=n_rows),
        "year": years,
        "category": np.random.choice(CATEGORY_ORDER, size=n_rows),
        "beat": np.random.choice([0, 1], p=[0.2, 0.8], size=n_rows),
        "prediction_cutoff": pd.to_datetime(["2020-03-31"] * n_rows) # dummy date
    })

    # Set prediction cutoff based on year to be chronological
    df["prediction_cutoff"] = df["year"].apply(lambda y: pd.Timestamp(f"{y}-12-31"))

    # Add numeric features
    for col in FEATURE_COLS:
        if col == "category_encoded":
            df[col] = np.random.choice(list(range(len(CATEGORY_ORDER))), size=n_rows)
        else:
            df[col] = np.random.normal(10.0, 5.0, size=n_rows)
            
    return df


def test_simple_standard_scaler():
    df_train = pd.DataFrame({"feat": [1.0, 2.0, 3.0, np.nan, 4.0]})
    df_test = pd.DataFrame({"feat": [2.0, 5.0]})

    scaler = SimpleStandardScaler()
    scaler.fit(df_train, ["feat"])

    # Mean = 2.5, Std = 1.29099
    assert np.isclose(scaler.means["feat"], 2.5)
    assert np.isclose(scaler.stds["feat"], pd.Series([1.0, 2.0, 3.0, 4.0]).std())

    df_train_scaled = scaler.transform(df_train, ["feat"])
    df_test_scaled = scaler.transform(df_test, ["feat"])

    # Check that transform standardizes correctly
    assert np.isclose(df_train_scaled.loc[0, "feat"], (1.0 - 2.5) / scaler.stds["feat"])
    assert np.isnan(df_train_scaled.loc[3, "feat"]) # NaNs should be preserved
    assert np.isclose(df_test_scaled.loc[1, "feat"], (5.0 - 2.5) / scaler.stds["feat"])


def test_get_walk_forward_folds(mock_prepared_df):
    folds = get_walk_forward_folds(mock_prepared_df)
    
    # Test years are 2020 to 2025 (6 folds)
    assert len(folds) == 6
    
    for fold in folds:
        test_year = fold["test_year"]
        train_df = fold["train"]
        val_df = fold["val"]
        test_df = fold["test"]
        
        # Chronological order checks
        assert (train_df["year"] < test_year - 1).all()
        assert (val_df["year"] == test_year - 1).all()
        assert (test_df["year"] == test_year).all()

        # No overlap between train, val, and test sets
        train_years = set(train_df["year"].unique())
        val_years = set(val_df["year"].unique())
        test_years = set(test_df["year"].unique())
        
        assert len(train_years & val_years) == 0
        assert len(train_years & test_years) == 0
        assert len(val_years & test_years) == 0
