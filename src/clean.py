"""
clean.py
--------
Cleans raw EPS data from data/raw/ and writes analysis-ready parquet files
to data/processed/.
 
Steps:
    1. Load eps_history.csv and eps_estimate.csv
    2. Drop null reported/estimate rows (documented)
    3. Compute forecast_error = reported - estimate
    4. Winsorize forecast_error by category (1st/99th percentile)
    5. Add time features (year, quarter, fiscal_quarter)
    6. Validate output integrity
    7. Write panel.parquet and estimate_panel.parquet
 
Usage:
    python src/clean.py
 
Outputs:
    - data/processed/panel.parquet
    - data/processed/estimate_panel.parquet
    - logs/clean.log
"""

import logging
import sys
import yaml
import numpy as np 
import pandas as pd
from pathlib import Path
from scipy.stats import mstats

ROOT         = Path(__file__).resolve().parent.parent
CONFIG_PATH  = ROOT / "configs" / "model_params.yml"
RAW_DIR      = ROOT / "data" / "raw"
PROCESSED    = ROOT / "data" / "processed"
LOG_DIR      = ROOT / "logs"
 
IN_HISTORY   = RAW_DIR / "eps_history.csv"
IN_ESTIMATE  = RAW_DIR / "eps_estimate.csv"
OUT_PANEL    = PROCESSED / "panel.parquet"
OUT_ESTIMATE = PROCESSED / "estimate_panel.parquet"

LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "clean.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def log_drop(reason: str, before: int, after: int) -> None:
    dropped = before - after
    log.info(f"  Dropped ({reason}): {dropped} rows  ({before} -> {after})")


def winsorize_by_category(
    df: pd.DataFrame,
    col: str,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    """Winsorize a column by category."""
    df = df.copy()
    df[f"{col}_winsorized"] = np.nan
    thresholds = {}
    
    for cat, group in df.groupby("category"):
        vals = group[col].dropna()
        lo = vals.quantile(lower)
        hi = vals.quantile(upper)
        thresholds[cat] = {"lower": round(lo, 4), "upper": round(hi, 4)}
        df.loc[group.index, f"{col}_winsorized"] = group[col].clip(lo, hi)

        log.info("Winsorization threshold by category:")
        for cat, t in sorted(thresholds.items()):
            log.info(f"    {cat:<30}  lower={t['lower']:>8.4f}  upper={t['upper']:>8.4f}")

        return df, thresholds


def assign_fiscal_quarter(period_end_date: pd.Series) -> pd.Series:
    """Assign fiscal quarter (1-4) based on the month of period_end_date."""
    month = period_end_date.dt.month
    return pd.cut(
        month,
        bins = [0,3,6,9,12],
        labels = [1,2,3,4],
    ).astype(int)


def clean_history(config: dict) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("Cleaning eps_history")
    log.info("=" * 60)
 
    df = pd.read_csv(IN_HISTORY, parse_dates=["period_end_date"])
    log.info(f"Loaded: {len(df):,} rows")

    n = len(df)
    null_reported = df[df["reported"].isnull()][["act_symbol", "period_end_date"]].values.tolist()
    null_estimate = df[df["estimate"].isnull()][["act_symbol", "period_end_date"]].values.tolist()
 
    log.info(f"  Null reported rows: {null_reported}")
    log.info(f"  Null estimate rows: {null_estimate}")
 
    df = df.dropna(subset=["reported", "estimate"])
    log_drop("null reported or estimate", n, len(df))

    df["forecast_error"] = df["reported"] - df["estimate"]
    log.info(f"  forecast_error computed: mean={df['forecast_error'].mean():.4f}, "
             f"std={df['forecast_error'].std():.4f}, "
             f"min={df['forecast_error'].min():.4f}, "
             f"max={df['forecast_error'].max():.4f}")

    lower = config["cleaning"]["winsorize_lower"]
    upper = config["cleaning"]["winsorize_upper"]
    log.info(f"  Winsorizing at [{lower}, {upper}] by category...")
    df, thresholds = winsorize_by_category(df, "forecast_error", lower, upper)

    clipped = (df["forecast_error"] != df["forecast_error_winsorized"]).sum()
    log.info(f"  Rows clipped by winsorization: {clipped}")

    df["year"]           = df["period_end_date"].dt.year
    df["calendar_month"] = df["period_end_date"].dt.month
    df["fiscal_quarter"] = assign_fiscal_quarter(df["period_end_date"])
    df["beat"] = (df["forecast_error"] > 0).astype(int)
 
    df["surprise_pct"] = np.where(
        df["estimate"].abs() > 0.01,
        df["forecast_error"] / df["estimate"].abs() * 100,
        np.nan,
    )
 
    df = df.sort_values(["act_symbol", "period_end_date"]).reset_index(drop=True)
 
    log.info(f"  Final panel shape: {df.shape}")
    log.info(f"  Columns: {df.columns.tolist()}")
    return df


def clean_estimate(config: dict) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("Cleaning eps_estimate")
    log.info("=" * 60)
 
    df = pd.read_csv(
        IN_ESTIMATE,
        parse_dates=["date", "period_end_date"],
        low_memory=False,
    )
    log.info(f"Loaded: {len(df):,} rows")
    
    n = len(df)
    df = df.dropna(subset=["consensus"])
    log_drop("null consensus", n, len(df))
    null_recent = df["recent"].isnull().sum()
    log.info(f"  Null 'recent' values retained: {null_recent} ({null_recent/len(df)*100:.1f}%) "
             f"— 'recent' is not a primary variable")
    df["year"]           = df["date"].dt.year
    df["week"]           = df["date"].dt.isocalendar().week.astype(int)
    df["fiscal_quarter"] = assign_fiscal_quarter(df["period_end_date"])
    df["consensus_spread"] = df["high"] - df["low"]
    df = df.sort_values(["act_symbol", "period_end_date", "date"]).reset_index(drop=True)
 
    log.info(f"  Final estimate panel shape: {df.shape}")
    log.info(f"  Columns: {df.columns.tolist()}")
    return df


def validate(panel: pd.DataFrame, estimate: pd.DataFrame, config: dict) -> None:
    log.info("=" * 60)
    log.info("Validation")
    log.info("=" * 60)
 
    tickers = [s for syms in config["tickers"].values() for s in syms]
    missing_panel    = [t for t in tickers if t not in panel["act_symbol"].values]
    missing_estimate = [t for t in tickers if t not in estimate["act_symbol"].values]
 
    if missing_panel:
        log.warning(f"  Tickers missing from panel: {missing_panel}")
    else:
        log.info("  All 72 tickers present in panel.")
 
    if missing_estimate:
        log.warning(f"  Tickers missing from estimate_panel: {missing_estimate}")
    else:
        log.info("  All 72 tickers present in estimate_panel.")
    critical_panel = ["reported", "estimate", "forecast_error", "forecast_error_winsorized", "category"]
    for col in critical_panel:
        n_null = panel[col].isnull().sum()
        if n_null > 0:
            log.warning(f"  panel['{col}'] has {n_null} nulls after cleaning.")
        else:
            log.info(f"  panel['{col}']: no nulls.")
 
    critical_estimate = ["consensus", "category", "period_end_date"]
    for col in critical_estimate:
        n_null = estimate[col].isnull().sum()
        if n_null > 0:
            log.warning(f"  estimate_panel['{col}'] has {n_null} nulls after cleaning.")
        else:
            log.info(f"  estimate_panel['{col}']: no nulls.")
 
    log.info(f"  panel date range    : {panel['period_end_date'].min().date()} "
             f"to {panel['period_end_date'].max().date()}")
    log.info(f"  estimate date range : {estimate['date'].min().date()} "
             f"to {estimate['date'].max().date()}")
 
    log.info("  Row counts by category (panel):")
    for cat, count in panel.groupby("category").size().items():
        log.info(f"    {cat:<30}  {count:>4} rows")
 
    log.info("=" * 60)


def main() -> None:
    from datetime import datetime
    log.info("=" * 60)
    log.info(f"clean.py started at {datetime.now().isoformat()}")
    log.info("=" * 60)
    config = load_config()
    PROCESSED.mkdir(exist_ok=True)
    panel    = clean_history(config)
    estimate = clean_estimate(config)
    validate(panel, estimate, config)
    panel.to_parquet(OUT_PANEL, index=False)
    log.info(f"Written: {OUT_PANEL}  ({len(panel):,} rows)")
    estimate.to_parquet(OUT_ESTIMATE, index=False)
    log.info(f"Written: {OUT_ESTIMATE}  ({len(estimate):,} rows)")
    log.info(f"clean.py complete at {datetime.now().isoformat()}")
 
 
if __name__ == "__main__":
    main()