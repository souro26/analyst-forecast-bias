"""Tests for src/ingest.py — get_all_tickers, get_ticker_category_map, and validate_coverage."""

import pandas as pd
import pytest
from src.ingest import get_all_tickers, get_ticker_category_map, validate_coverage


@pytest.fixture
def mock_config():
    """Mock configuration dictionary."""
    return {
        "tickers": {
            "tech_cycle": ["AAPL", "NVDA"],
            "macro_rate_sensitive": ["JPM", "GS"],
        },
        "cleaning": {
            "date_start": "2017-01-01",
            "date_end": "2025-12-31",
        }
    }


def test_get_all_tickers(mock_config):
    tickers = get_all_tickers(mock_config)
    assert isinstance(tickers, list)
    assert sorted(tickers) == ["AAPL", "GS", "JPM", "NVDA"]


def test_get_ticker_category_map(mock_config):
    mapping = get_ticker_category_map(mock_config)
    assert isinstance(mapping, dict)
    assert mapping["AAPL"] == "tech_cycle"
    assert mapping["NVDA"] == "tech_cycle"
    assert mapping["JPM"] == "macro_rate_sensitive"
    assert mapping["GS"] == "macro_rate_sensitive"


def test_validate_coverage_all_present(caplog):
    """Test validate_coverage when all tickers are present."""
    tickers = ["AAPL", "JPM"]
    history = pd.DataFrame({"act_symbol": ["AAPL", "JPM"]})
    estimate = pd.DataFrame({"act_symbol": ["AAPL", "JPM"]})

    with caplog.at_level("INFO"):
        validate_coverage(tickers, history, estimate)

    assert "All tickers present in eps_history." in caplog.text
    assert "All tickers present in eps_estimate." in caplog.text
    assert "Missing from" not in caplog.text


def test_validate_coverage_missing_tickers(caplog):
    """Test validate_coverage when some tickers are missing."""
    tickers = ["AAPL", "JPM", "XOM"]
    history = pd.DataFrame({"act_symbol": ["AAPL", "JPM"]})
    estimate = pd.DataFrame({"act_symbol": ["AAPL"]})

    with caplog.at_level("WARNING"):
        validate_coverage(tickers, history, estimate)

    assert "Missing from eps_history  : ['XOM']" in caplog.text
    assert "Missing from eps_estimate : ['JPM', 'XOM']" in caplog.text
