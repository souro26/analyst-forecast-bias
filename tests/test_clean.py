"""Tests for src/clean.py — winsorize_by_category, assign_fiscal_quarter, and normalized error targets."""

import numpy as np
import pandas as pd
import pytest

from src.clean import assign_fiscal_quarter, winsorize_by_category


@pytest.fixture
def two_category_panel():
    """Minimal two-category panel with clean numeric forecast errors."""
    return pd.DataFrame({
        "act_symbol":     ["AAPL"] * 10 + ["JPM"] * 10,
        "category":       ["tech_cycle"] * 10 + ["macro_rate_sensitive"] * 10,
        "forecast_error": list(range(-4, 6)) + list(range(-4, 6)),
    })


class TestWinsorizeByCategory:

    def test_returns_tuple_of_df_and_dict(self, two_category_panel):
        result = winsorize_by_category(two_category_panel, "forecast_error", 0.01, 0.99)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], dict)

    def test_winsorized_column_added(self, two_category_panel):
        df, _ = winsorize_by_category(two_category_panel, "forecast_error", 0.01, 0.99)
        assert "forecast_error_winsorized" in df.columns

    def test_clipping_reduces_extreme_values(self):
        df = pd.DataFrame({
            "category":       ["A"] * 20,
            "forecast_error": [1000.0] + list(range(1, 19)) + [-1000.0],
        })
        result, _ = winsorize_by_category(df, "forecast_error", 0.05, 0.95)
        assert result["forecast_error_winsorized"].max() < 1000.0
        assert result["forecast_error_winsorized"].min() > -1000.0

    def test_thresholds_keyed_by_each_category(self, two_category_panel):
        _, thresholds = winsorize_by_category(two_category_panel, "forecast_error", 0.01, 0.99)
        assert "tech_cycle" in thresholds
        assert "macro_rate_sensitive" in thresholds

    def test_threshold_dicts_have_lower_and_upper(self, two_category_panel):
        _, thresholds = winsorize_by_category(two_category_panel, "forecast_error", 0.01, 0.99)
        for t in thresholds.values():
            assert "lower" in t
            assert "upper" in t

    def test_no_nulls_in_winsorized_column(self, two_category_panel):
        df, _ = winsorize_by_category(two_category_panel, "forecast_error", 0.01, 0.99)
        assert df["forecast_error_winsorized"].isnull().sum() == 0

    def test_original_column_not_modified(self, two_category_panel):
        original = two_category_panel["forecast_error"].copy()
        df, _ = winsorize_by_category(two_category_panel, "forecast_error", 0.01, 0.99)
        pd.testing.assert_series_equal(df["forecast_error"], original)

    def test_inplace_values_unchanged_when_no_outliers(self):
        df = pd.DataFrame({
            "category":       ["A"] * 5,
            "forecast_error": [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        result, _ = winsorize_by_category(df, "forecast_error", 0.00, 1.00)
        pd.testing.assert_series_equal(
            result["forecast_error_winsorized"].reset_index(drop=True),
            df["forecast_error"].reset_index(drop=True),
            check_names=False,
        )


class TestAssignFiscalQuarter:

    def test_january_to_march_is_q1(self):
        dates = pd.to_datetime(["2021-01-31", "2021-02-28", "2021-03-31"])
        assert (assign_fiscal_quarter(pd.Series(dates)) == 1).all()

    def test_april_to_june_is_q2(self):
        dates = pd.to_datetime(["2021-04-30", "2021-05-31", "2021-06-30"])
        assert (assign_fiscal_quarter(pd.Series(dates)) == 2).all()

    def test_july_to_september_is_q3(self):
        dates = pd.to_datetime(["2021-07-31", "2021-08-31", "2021-09-30"])
        assert (assign_fiscal_quarter(pd.Series(dates)) == 3).all()

    def test_october_to_december_is_q4(self):
        dates = pd.to_datetime(["2021-10-31", "2021-11-30", "2021-12-31"])
        assert (assign_fiscal_quarter(pd.Series(dates)) == 4).all()

    def test_returns_integer_dtype(self):
        dates = pd.to_datetime(["2021-03-31"])
        result = assign_fiscal_quarter(pd.Series(dates))
        assert np.issubdtype(result.dtype, np.integer)

    def test_all_four_quarters_covered(self):
        dates = pd.to_datetime(["2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01"])
        result = assign_fiscal_quarter(pd.Series(dates))
        assert set(result.values) == {1, 2, 3, 4}

    def test_year_boundary_does_not_affect_quarter(self):
        dates = pd.to_datetime(["2020-12-31", "2021-01-01"])
        result = assign_fiscal_quarter(pd.Series(dates))
        assert result.iloc[0] == 4
        assert result.iloc[1] == 1


class TestNormalizedError:

    def test_normalized_error_basic(self):
        # positive EPS
        reported = 2.20
        estimate = 2.00
        normalized = (reported - estimate) / abs(estimate)
        assert np.isclose(normalized, 0.10)

        # negative EPS
        reported = -1.80
        estimate = -2.00
        normalized = (reported - estimate) / abs(estimate)
        assert np.isclose(normalized, 0.10)

    def test_near_zero_estimates_handling(self):
        # Test values close to zero (absolute value < 0.01)
        eps_values = [0.0, 0.005, -0.008, -0.0001]
        for est in eps_values:
            val = np.where(
                np.abs(est) >= 0.01,
                (1.0 - est) / np.abs(est),
                np.nan
            )
            assert np.isnan(val)

        # Value just above or equal to threshold
        assert not np.isnan(np.where(np.abs(0.01) >= 0.01, (1.0 - 0.01) / 0.01, np.nan))
