"""
macro.py
--------
Downloads S&P 500 returns and VIX prices via yfinance, and prepares:
1. Explanatory macro variables (full-quarter returns and averages) for descriptive/Bayesian analysis.
2. Predictive point-in-time (PIT) macro variables (112-day lookback, strictly before the cutoff) for signals.

Outputs:
    - data/processed/panel_macro.parquet   (updated with both explanatory and predictive macro variables)
    - data/processed/macro.parquet         (quarterly macro indicators, for reference)
    - data/processed/macro_audit.parquet   (contains metadata for validating macro PIT cutoff dates)
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

PANEL_PATH        = PROCESSED / "panel.parquet"
PANEL_MACRO_PATH  = PROCESSED / "panel_macro.parquet"
MACRO_PATH        = PROCESSED / "macro.parquet"
MACRO_AUDIT_PATH  = PROCESSED / "macro_audit.parquet"

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


def assign_fiscal_quarter(period_end_date: pd.Series) -> pd.Series:
    """Assign fiscal quarter based on period end date."""
    month = period_end_date.dt.month
    return pd.cut(
        month,
        bins = [0,3,6,9,12],
        labels = [1,2,3,4],
    ).astype(int)


def get_sp500_quarterly(start: str = "2016-10-01", end: str = "2026-07-01") -> pd.DataFrame:
    """Download and compute quarterly S&P 500 returns (explanatory)."""
    log.info("Downloading S&P 500 monthly prices...")
    raw = yf.download(
        "^GSPC",
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )

    prices = raw["Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices.index = pd.to_datetime(prices.index)

    quarterly = prices.resample("QE").last()
    returns   = quarterly.pct_change().dropna()

    df = returns.to_frame(name="sp500_return")
    df.index.name = "quarter_end"

    log.info(f"  S&P 500 quarterly returns: {len(df)} quarters")
    return df


def get_vix_quarterly(start: str = "2016-10-01", end: str = "2026-07-01") -> pd.DataFrame:
    """Download and compute quarterly VIX mean values (explanatory)."""
    log.info("Downloading VIX daily prices...")
    raw = yf.download(
        "^VIX",
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    vix = raw["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index)

    quarterly_vix = vix.resample("QE").mean()

    df = quarterly_vix.to_frame(name="vix_mean")
    df.index.name = "quarter_end"

    log.info(f"  VIX quarterly means: {len(df)} quarters")
    return df


def build_macro_table(sp500: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Build quarterly explanatory macro indicators.

    Raw quarterly macro variables are retained for descriptive and
    explanatory analysis. Predictive PIT variables are computed separately
    from daily data in compute_pit_macro().
    """
    log.info("Building quarterly macro table (explanatory)...")

    macro = sp500.join(vix, how="inner")

    macro["sp500_return_z"] = (
        (macro["sp500_return"] - macro["sp500_return"].mean())
        / macro["sp500_return"].std()
    )

    macro["vix_mean_z"] = (
        (macro["vix_mean"] - macro["vix_mean"].mean())
        / macro["vix_mean"].std()
    )

    return macro


def compute_pit_macro(
    panel: pd.DataFrame,
    gspc_close: pd.Series,
    vix_close: pd.Series
) -> pd.DataFrame:
    """
    Computes point-in-time (PIT) S&P 500 return and average VIX for each observation.

    Uses strictly lookback data before prediction_cutoff:
    [prediction_cutoff - 112 days, prediction_cutoff)

    The macro audit data is constructed separately in main().
    """
    panel = panel.copy()
    sp500_pit_vals = []
    vix_pit_vals = []

    gspc_close = gspc_close.sort_index()
    vix_close = vix_close.sort_index()

    for _, row in panel.iterrows():
        cutoff = pd.to_datetime(row["prediction_cutoff"])
        start_date = cutoff - pd.Timedelta(days=112)

        # S&P 500 PIT return
        gspc_before = gspc_close[gspc_close.index < cutoff]

        if len(gspc_before) > 0:
            p_end = gspc_before.iloc[-1]

            gspc_start_candidates = gspc_close[
                gspc_close.index < start_date
            ]

            if len(gspc_start_candidates) > 0:
                p_start = gspc_start_candidates.iloc[-1]
            else:
                p_start = gspc_before.iloc[0]

            sp500_return_pit = (p_end - p_start) / p_start
        else:
            sp500_return_pit = np.nan

        # VIX PIT mean
        vix_window = vix_close[
            (vix_close.index >= start_date) &
            (vix_close.index < cutoff)
        ]

        if len(vix_window) > 0:
            vix_mean_pit = vix_window.mean()
        else:
            vix_before = vix_close[vix_close.index < cutoff]

            if len(vix_before) > 0:
                vix_mean_pit = vix_before.iloc[-1]
            else:
                vix_mean_pit = np.nan

        sp500_pit_vals.append(sp500_return_pit)
        vix_pit_vals.append(vix_mean_pit)

    panel["sp500_return_pit"] = sp500_pit_vals
    panel["vix_mean_pit"] = vix_pit_vals

    return panel


def merge_onto_panel(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Merge quarterly explanatory macro indicators onto panel using explicit year/quarter alignment."""
    log.info("Merging explanatory macro indicators onto panel using explicit quarter alignment...")

    panel = panel.copy()

    macro_reset = macro[["sp500_return", "sp500_return_z", "vix_mean", "vix_mean_z"]].reset_index()
    macro_reset = macro_reset.rename(columns={"quarter_end": "period_end_date"})
    macro_reset["period_end_date"] = pd.to_datetime(macro_reset["period_end_date"])
    
    # Calculate year and fiscal_quarter for macro data
    macro_reset["year"] = macro_reset["period_end_date"].dt.year
    macro_reset["fiscal_quarter"] = assign_fiscal_quarter(macro_reset["period_end_date"])

    # Ensure year and fiscal_quarter are present in panel
    panel["period_end_date"] = pd.to_datetime(panel["period_end_date"]).astype("datetime64[ns]")
    panel["year"] = panel["period_end_date"].dt.year
    panel["fiscal_quarter"] = assign_fiscal_quarter(panel["period_end_date"])

    # Explicit merge on year and fiscal quarter
    panel = pd.merge(
        panel,
        macro_reset.drop(columns=["period_end_date"]),
        on=["year", "fiscal_quarter"],
        how="left"
    )

    n_null_sp500 = panel["sp500_return"].isnull().sum()
    n_null_vix   = panel["vix_mean"].isnull().sum()
    log.info(f"  sp500_return nulls after merge: {n_null_sp500}")
    log.info(f"  vix_mean nulls after merge    : {n_null_vix}")

    return panel


def main() -> None:
    log.info("=" * 60)
    log.info(f"macro.py started at {datetime.now().isoformat()}")
    log.info("=" * 60)

    # 1. Download/build quarterly explanatory macro indicators
    sp500_q = get_sp500_quarterly()
    vix_q   = get_vix_quarterly()
    macro = build_macro_table(sp500_q, vix_q)
    macro.to_parquet(MACRO_PATH)
    log.info(f"Written explanatory macro table: {MACRO_PATH}")

    # 2. Download daily prices for point-in-time calculations
    log.info("Downloading daily S&P 500 and VIX prices for PIT calculations...")
    gspc_daily = yf.download("^GSPC", start="2016-01-01", end="2026-07-01", interval="1d", auto_adjust=True, progress=False)
    vix_daily  = yf.download("^VIX", start="2016-01-01", end="2026-07-01", interval="1d", auto_adjust=True, progress=False)

    gspc_close = gspc_daily["Close"]
    if isinstance(gspc_close, pd.DataFrame):
        gspc_close = gspc_close.iloc[:, 0]
    vix_close = vix_daily["Close"]
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.iloc[:, 0]

    gspc_close.index = pd.to_datetime(gspc_close.index)
    vix_close.index = pd.to_datetime(vix_close.index)

    # Load panel
    panel = pd.read_parquet(PANEL_PATH)
    log.info(f"Panel loaded: {len(panel):,} rows")

    # 3. Merge explanatory macro indicators (explicit quarter alignment)
    panel = merge_onto_panel(panel, macro)

    # 4. Compute predictive PIT macro indicators
    log.info("Computing point-in-time predictive macro indicators...")
    panel = compute_pit_macro(panel, gspc_close, vix_close)

    # Build separate audit table from the same PIT cutoff logic
    audit_records = []

    for _, row in panel.iterrows():
        cutoff = pd.to_datetime(row["prediction_cutoff"])
        start_date = cutoff - pd.Timedelta(days=112)

        gspc_before = gspc_close[gspc_close.index < cutoff]
        vix_window = vix_close[
            (vix_close.index >= start_date) &
            (vix_close.index < cutoff)
        ]

        latest_gspc_date = (
            gspc_before.index[-1] if len(gspc_before) > 0 else pd.NaT
        )

        if len(vix_window) > 0:
            latest_vix_date = vix_window.index[-1]
        else:
            vix_before = vix_close[vix_close.index < cutoff]
            latest_vix_date = (
                vix_before.index[-1] if len(vix_before) > 0 else pd.NaT
            )

        audit_records.append({
            "act_symbol": row["act_symbol"],
            "period_end_date": row.get(
                "period_end_date",
                row["prediction_cutoff"]
            ),
            "prediction_cutoff": row["prediction_cutoff"],
            "latest_gspc_date": latest_gspc_date,
            "latest_vix_date": latest_vix_date,
        })

    audit_df = pd.DataFrame(audit_records)

    panel.to_parquet(PANEL_MACRO_PATH, index=False)
    log.info(f"Written panel with macro variables: {PANEL_MACRO_PATH}")

    audit_df.to_parquet(MACRO_AUDIT_PATH, index=False)
    log.info(f"Written macro PIT audit data: {MACRO_AUDIT_PATH}")

    log.info("=" * 60)
    log.info(f"macro.py complete at {datetime.now().isoformat()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()