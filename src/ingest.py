"""
ingest.py
---------
Filters the raw Dolt table dumps (eps_estimate_full.csv, eps_history_full.csv)
to 72 tickers and date range, then writes clean CSVs to data/raw/.

Prerequisites:
    - data/raw/eps_estimate_full.csv  (from: dolt table export eps_estimate)
    - data/raw/eps_history_full.csv   (from: dolt table export eps_history)

Outputs:
    - data/raw/eps_history.csv
    - data/raw/eps_estimate.csv
    - logs/ingest.log
"""

import logging
import sys
import yaml
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH   = ROOT / "configs" / "model_params.yml"
RAW_DIR       = ROOT / "data" / "raw"
LOG_DIR       = ROOT / "logs"

FULL_HISTORY  = RAW_DIR / "eps_history_full.csv"
FULL_ESTIMATE = RAW_DIR / "eps_estimate_full.csv"
OUT_HISTORY   = RAW_DIR / "eps_history.csv"
OUT_ESTIMATE  = RAW_DIR / "eps_estimate.csv"

LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "ingest.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_all_tickers(config: dict) -> list[str]:
    tickers = []
    for symbols in config["tickers"].values():
        tickers.extend(symbols)
    return sorted(set(tickers))


def get_ticker_category_map(config: dict) -> dict[str, str]:
    mapping = {}
    for category, symbols in config["tickers"].items():
        for symbol in symbols:
            mapping[symbol] = category
    return mapping


def validate_coverage(
    tickers: list[str],
    history: pd.DataFrame,
    estimate: pd.DataFrame,
) -> None:
    history_tickers  = set(history["act_symbol"].unique())
    estimate_tickers = set(estimate["act_symbol"].unique())

    missing_history  = [t for t in tickers if t not in history_tickers]
    missing_estimate = [t for t in tickers if t not in estimate_tickers]

    log.info("=" * 60)
    log.info("COVERAGE REPORT")
    log.info("=" * 60)
    log.info(f"Tickers requested : {len(tickers)}")
    log.info(f"eps_history       : {len(history_tickers)} tickers, {len(history)} rows")
    log.info(f"eps_estimate      : {len(estimate_tickers)} tickers, {len(estimate)} rows")

    if missing_history:
        log.warning(f"Missing from eps_history  : {missing_history}")
    else:
        log.info("All tickers present in eps_history.")

    if missing_estimate:
        log.warning(f"Missing from eps_estimate : {missing_estimate}")
    else:
        log.info("All tickers present in eps_estimate.")

    log.info("-" * 60)
    log.info("Row counts per ticker (eps_history):")
    counts = history.groupby("act_symbol").size()
    for ticker in sorted(tickers):
        count = counts.get(ticker, 0)
        flag  = "  *** MISSING ***" if count == 0 else ""
        log.info(f"  {ticker:<6}  {count:>4} rows{flag}")

    log.info("-" * 60)
    log.info("Row counts per ticker (eps_estimate):")
    counts = estimate.groupby("act_symbol").size()
    for ticker in sorted(tickers):
        count = counts.get(ticker, 0)
        flag  = "  *** MISSING ***" if count == 0 else ""
        log.info(f"  {ticker:<6}  {count:>4} rows{flag}")

    log.info("=" * 60)


def main() -> None:
    log.info("=" * 60)
    log.info(f"ingest.py started at {datetime.now().isoformat()}")
    log.info("=" * 60)

    config   = load_config()
    tickers  = get_all_tickers(config)
    cat_map  = get_ticker_category_map(config)
    date_start = config["cleaning"]["date_start"]
    date_end   = config["cleaning"]["date_end"]
    log.info(f"Loaded {len(tickers)} tickers from config.")
    log.info(f"Date range: {date_start} to {date_end}")

    for path in [FULL_HISTORY, FULL_ESTIMATE]:
        if not path.exists():
            log.error(
                f"Raw dump not found: {path}\n"
                "Run from data/external/earnings/:\n"
                "  dolt table export eps_history ../../../data/raw/eps_history_full.csv\n"
                "  dolt table export eps_estimate ../../../data/raw/eps_estimate_full.csv"
            )
            sys.exit(1)

    log.info(f"Loading {FULL_HISTORY.name} ...")
    history_full = pd.read_csv(FULL_HISTORY, parse_dates=["period_end_date"])
    log.info(f"  {len(history_full):,} rows loaded.")

    log.info(f"Loading {FULL_ESTIMATE.name} ...")
    estimate_full = pd.read_csv(
        FULL_ESTIMATE,
        parse_dates=["date", "period_end_date"],
        low_memory=False,
    )
    log.info(f"  {len(estimate_full):,} rows loaded.")

    history  = history_full[history_full["act_symbol"].isin(tickers)].copy()
    estimate = estimate_full[estimate_full["act_symbol"].isin(tickers)].copy()
    log.info(f"After ticker filter — history: {len(history):,} rows, estimate: {len(estimate):,} rows.")

    history  = history[
        (history["period_end_date"] >= date_start) &
        (history["period_end_date"] <= date_end)
    ].copy()

    estimate = estimate[
        (estimate["date"] >= date_start) &
        (estimate["date"] <= date_end)
    ].copy()
    log.info(f"After date filter  — history: {len(history):,} rows, estimate: {len(estimate):,} rows.")

    estimate = estimate[
        estimate["period"].isin(["Current Quarter", "Current Year"])
    ].copy()
    log.info(f"After period filter — estimate: {len(estimate):,} rows.")

    history["category"]  = history["act_symbol"].map(cat_map)
    estimate["category"] = estimate["act_symbol"].map(cat_map)

    history  = history.sort_values(["act_symbol", "period_end_date"]).reset_index(drop=True)
    estimate = estimate.sort_values(["act_symbol", "date"]).reset_index(drop=True)

    validate_coverage(tickers, history, estimate)

    history.to_csv(OUT_HISTORY, index=False)
    log.info(f"Written: {OUT_HISTORY}  ({len(history):,} rows)")

    estimate.to_csv(OUT_ESTIMATE, index=False)
    log.info(f"Written: {OUT_ESTIMATE}  ({len(estimate):,} rows)")

    log.info(f"ingest.py complete at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()