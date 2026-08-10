"""Tests for src/signals.py — split_data, train_model, evaluate, and get_feature_importance."""

import numpy as np
import pandas as pd
import xgboost as xgb
import pytest
from src.signals import split_data, train_model, evaluate, get_feature_importance, FEATURE_COLS


@pytest.fixture
def mock_prepared_df():
    """Create a mock prepared DataFrame for signals training and evaluation."""
    np.random.seed(42)
    n_rows = 100
    
    # 2017 to 2022 -> Train, 2023 to 2025 -> Test
    years = np.random.choice(list(range(2017, 2026)), size=n_rows)
    
    df = pd.DataFrame({
        "year": years,
        "category": np.random.choice(["tech_cycle", "commodity_driven"], size=n_rows),
        "beat": np.random.choice([0, 1], p=[0.2, 0.8], size=n_rows),
    })

    # Add features
    for col in FEATURE_COLS:
        if col == "category_encoded":
            df[col] = np.random.choice([0, 1], size=n_rows)
        else:
            df[col] = np.random.normal(0.0, 1.0, size=n_rows)
            
    return df


def test_split_data(mock_prepared_df):
    X_train, X_test, y_train, y_test, train, test = split_data(mock_prepared_df)

    # Verify time-based split bounds
    assert (train["year"] <= 2022).all()
    assert (test["year"] >= 2023).all()

    # Column checks
    assert list(X_train.columns) == FEATURE_COLS
    assert list(X_test.columns) == FEATURE_COLS
    assert y_train.name == "beat"
    assert y_test.name == "beat"
    
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_train_model(mock_prepared_df):
    X_train, X_test, y_train, y_test, _, _ = split_data(mock_prepared_df)

    model = train_model(X_train, y_train, X_test, y_test)

    assert isinstance(model, xgb.XGBClassifier)
    # Check default parameters we set inside train_model
    assert model.get_params()["max_depth"] == 4
    assert model.get_params()["learning_rate"] == 0.05
    assert model.get_params()["objective"] == "binary:logistic"


def test_evaluate(mock_prepared_df):
    X_train, X_test, y_train, y_test, _, test = split_data(mock_prepared_df)
    model = train_model(X_train, y_train, X_test, y_test)

    results_df = evaluate(model, X_test, y_test, test)

    assert isinstance(results_df, pd.DataFrame)
    assert "beat_proba" in results_df.columns
    assert "beat_pred" in results_df.columns
    assert len(results_df) == len(test)


def test_get_feature_importance(mock_prepared_df):
    X_train, X_test, y_train, y_test, _, _ = split_data(mock_prepared_df)
    model = train_model(X_train, y_train, X_test, y_test)

    imp_df = get_feature_importance(model)

    assert isinstance(imp_df, pd.DataFrame)
    assert "feature" in imp_df.columns
    assert "gain" in imp_df.columns
    assert "gain_normalized" in imp_df.columns
    # Ensure gain sums to 1 after normalization
    assert np.isclose(imp_df["gain_normalized"].sum(), 1.0)
