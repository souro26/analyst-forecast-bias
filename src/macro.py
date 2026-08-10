"""
macro.py
--------
Downloads quarterly S&P 500 returns and average VIX per quarter
from yfinance and merges them onto panel.parquet as continuous
macro predictors for the hierarchical Bayesian model.

Why continuous predictors instead of discrete regime labels:
    With only 9 years of data, discrete bull/bear/sideways labels
    give the model 3-4 data points per regime — not enough to
    estimate regime effects. Continuous predictors give every
    observation its own macro context (2,455 data points), making
    the macro effect estimable and quantitative.

Outputs:
    - data/processed/panel.parquet   (updated with sp500_return, vix_mean columns)
    - data/processed/macro.parquet   (quarterly macro indicators, for reference)
    - logs/macro.log

Usage:
    python src/macro.py
"""

import logging
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

ROOT        = Path(__file__).resolve().parent.parent
PROCESSED   = ROOT / "data" / "processed"
LOG_DIR     = ROOT / "logs"

PANEL_PATH       = PROCESSED / "panel.parquet"
PANEL_MACRO_PATH = PROCESSED / "panel_macro.parquet"
MACRO_PATH       = PROCESSED / "macro.parquet"

LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "macro.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

def get_sp500_quarterly(start: str = "2016-10-01", end: str = "2026-07-01") -> pd.DataFrame:
    """
    Download monthly S&P 500 prices, resample to quarterly,
    compute quarter-over-quarter return.

    Returns DataFrame indexed by quarter-end date with column sp500_return.
    """
    log.info("Downloading S&P 500 monthly prices...")
    raw = yf.download(
        "^GSPC",
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )

    prices = raw["Close"].squeeze()
    prices.index = pd.to_datetime(prices.index)

    quarterly = prices.resample("QE").last()
    returns   = quarterly.pct_change().dropna()

    df = returns.to_frame(name="sp500_return")
    df.index.name = "quarter_end"

    log.info(f"  S&P 500 quarterly returns: {len(df)} quarters")
    log.info(f"  Range: {df.index[0].date()} to {df.index[-1].date()}")
    log.info(f"  Mean  : {df['sp500_return'].mean():.4f}")
    log.info(f"  Std   : {df['sp500_return'].std():.4f}")
    log.info(f"  Min   : {df['sp500_return'].min():.4f}  "
             f"Max: {df['sp500_return'].max():.4f}")

    return df


def get_vix_quarterly(start: str = "2016-10-01", end: str = "2026-07-01") -> pd.DataFrame:
    """
    Download daily VIX, resample to quarterly mean.
    VIX measures market uncertainty during the quarter —
    the forecasting environment analysts were operating in.

    Returns DataFrame indexed by quarter-end date with column vix_mean.
    """
    log.info("Downloading VIX daily prices...")
    raw = yf.download(
        "^VIX",
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    vix = raw["Close"].squeeze()
    vix.index = pd.to_datetime(vix.index)

    quarterly_vix = vix.resample("QE").mean()

    df = quarterly_vix.to_frame(name="vix_mean")
    df.index.name = "quarter_end"

    log.info(f"  VIX quarterly means: {len(df)} quarters")
    log.info(f"  Range: {df.index[0].date()} to {df.index[-1].date()}")
    log.info(f"  Mean  : {df['vix_mean'].mean():.2f}")
    log.info(f"  Std   : {df['vix_mean'].std():.2f}")
    log.info(f"  Min   : {df['vix_mean'].min():.2f}  "
             f"Max: {df['vix_mean'].max():.2f}")

    return df


def build_macro_table(sp500: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """
    Join S&P 500 returns and VIX into a single quarterly macro table.
    Standardize both variables (z-score) so model coefficients are
    on the same scale and interpretable as per-SD effects.
    """
    log.info("Building quarterly macro table...")

    macro = sp500.join(vix, how="inner")

    macro["sp500_return_z"] = (
        (macro["sp500_return"] - macro["sp500_return"].mean())
        / macro["sp500_return"].std()
    )
    macro["vix_mean_z"] = (
        (macro["vix_mean"] - macro["vix_mean"].mean())
        / macro["vix_mean"].std()
    )

    log.info(f"  Macro table: {len(macro)} quarters")
    log.info(f"  sp500_return_z: mean={macro['sp500_return_z'].mean():.4f}  "
             f"std={macro['sp500_return_z'].std():.4f}")
    log.info(f"  vix_mean_z:     mean={macro['vix_mean_z'].mean():.4f}  "
             f"std={macro['vix_mean_z'].std():.4f}")

    return macro


def merge_onto_panel(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Match each company-quarter in panel to its macro indicators.
    Uses merge_asof for nearest-date matching.
    """
    log.info("Merging macro indicators onto panel...")

    panel = panel.copy()

    macro_reset = macro[["sp500_return", "sp500_return_z",
                          "vix_mean", "vix_mean_z"]].reset_index()
    macro_reset = macro_reset.rename(columns={"quarter_end": "period_end_date"})
    macro_reset["period_end_date"] = pd.to_datetime(macro_reset["period_end_date"])

    panel["period_end_date"] = pd.to_datetime(panel["period_end_date"]).astype("datetime64[ns]")
    macro_reset["period_end_date"] = pd.to_datetime(macro_reset["period_end_date"]).astype("datetime64[ns]")

    panel = panel.sort_values("period_end_date")
    macro_reset = macro_reset.sort_values("period_end_date")

    panel = pd.merge_asof(
        panel,
        macro_reset,
        on="period_end_date",
        direction="nearest",
    )

    n_null_sp500 = panel["sp500_return"].isnull().sum()
    n_null_vix   = panel["vix_mean"].isnull().sum()
    log.info(f"  sp500_return nulls after merge: {n_null_sp500}")
    log.info(f"  vix_mean nulls after merge    : {n_null_vix}")

    log.info("  Macro indicator summary in panel:")
    log.info(f"    sp500_return  mean={panel['sp500_return'].mean():.4f}  "
             f"std={panel['sp500_return'].std():.4f}")
    log.info(f"    vix_mean      mean={panel['vix_mean'].mean():.2f}  "
             f"std={panel['vix_mean'].std():.2f}")

    return panel

def main() -> None:
    log.info("=" * 60)
    log.info(f"macro.py started at {datetime.now().isoformat()}")
    log.info("=" * 60)

    sp500 = get_sp500_quarterly()
    vix   = get_vix_quarterly()
    macro = build_macro_table(sp500, vix)
    macro.to_parquet(MACRO_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    log.info(f"Panel loaded: {len(panel):,} rows")

    panel = merge_onto_panel(panel, macro)

    panel.to_parquet(PANEL_MACRO_PATH, index=False)
    log.info(f"Written: {PANEL_MACRO_PATH}  (panel + sp500_return + vix_mean)")

    log.info("=" * 60)
    log.info(f"macro.py complete at {datetime.now().isoformat()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()