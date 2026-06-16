"""
ingest.py
---------
Exports eps_history and eps_estimate tables from the local Dolt database
(data/external/earnings/) into data/raw/ as CSV files, filtered to the
72 tickers defined in configs/model_params.yml.
 
Usage:
    python src/ingest.py
 
Requirements:
    - Dolt CLI installed and in PATH
    - data/external/earnings/ cloned via:
        cd data/external && dolt clone post-no-preference/earnings
 
Outputs:
    - data/raw/eps_history.csv
    - data/raw/eps_estimate.csv
    - logs/ingest.log
"""

import subprocess
import csv
import logging
import sys
import yaml
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "model_params.yml"
DOLT_DB_PATH = ROOT / "data" / "external" / "earnings"
RAW_DIR = ROOT / "data" / "raw"
LOG_DIR = ROOT / "logs"

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
    """Flatten the category → ticker mapping into a single sorted list."""
    tickers = []
    for category, symbols in config["tickers"].items():
        for symbol in symbols:
            tickers.append(symbol)
    return sorted(set(tickers))


def ticker_sql_list(tickers: list[str]) -> str:
    """Return a SQL-safe IN (...) string for the ticker list."""
    escaped = ", ".join(f"'{t}'" for t in tickers)
    return f"({escaped})"


def run_dolt_query(query: str) -> list[dict]:
    """Run a SQL query against the local Dolt database and return results as a list of dicts."""
    cmd = [
        "dolt", "sql",
        "--result-format", "csv",
        "-q", query,
    ]
    result = subprocess.run(
        cmd,
        cwd=DOLT_DB_PATH,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Dolt query failed:\n{result.stderr}\nQuery: {query}"
        )
 
    lines = result.stdout.strip().splitlines()
    if not lines:
        return []
 
    reader = csv.DictReader(lines)
    return list(reader)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        log.warning(f"No rows to write to {path}.")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def row_counts_by_ticker(rows: list[dict], ticker_col: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        sym = row[ticker_col]
        counts[sym] = counts.get(sym, 0) + 1
    return counts


def export_eps_history(tickers: list[str], config: dict) -> list[dict]:
    """Export eps_history for all tickers, filtered by date range in config."""
    date_start = config["cleaning"]["date_start"]
    date_end = config["cleaning"]["date_end"]
    ticker_list = ticker_sql_list(tickers)
 
    query = f"""
        SELECT act_symbol, period_end_date, reported, estimate
        FROM eps_history
        WHERE act_symbol IN {ticker_list}
          AND period_end_date >= '{date_start}'
          AND period_end_date <= '{date_end}'
        ORDER BY act_symbol, period_end_date;
    """
 
    log.info("Querying eps_history from Dolt...")
    rows = run_dolt_query(query)
    log.info(f"eps_history: {len(rows)} rows returned.")
    return rows


def export_eps_estimate(tickers: list[str], config: dict) -> list[dict]:
    """Export eps_estimate for all tickers, filtered by date range in config."""
    date_start = config["cleaning"]["date_start"]
    date_end = config["cleaning"]["date_end"]
    ticker_list = ticker_sql_list(tickers)
 
    query = f"""
        SELECT date, act_symbol, period, period_end_date,
               consensus, recent, count, high, low, year_ago
        FROM eps_estimate
        WHERE act_symbol IN {ticker_list}
          AND date >= '{date_start}'
          AND date <= '{date_end}'
          AND period IN ('Current Quarter', 'Current Year')
        ORDER BY act_symbol, date;
    """
 
    log.info("Querying eps_estimate from Dolt...")
    rows = run_dolt_query(query)
    log.info(f"eps_estimate: {len(rows)} rows returned.")
    return rows
 

def validate_coverage(
    tickers: list[str],
    history_rows: list[dict],
    estimate_rows: list[dict],
) -> None:
    """Log coverage stats: which tickers have data, which are missing, and row counts per ticker for both tables."""
    history_count = row_counts_by_ticker(history_rows, "act_symbol")
    estimate_count = row_counts_by_ticker(estimate_rows, "act_symbol")

    missing_history = [t for t in tickers if t not in history_count]
    missing_estimate = [t for t in tickers if t not in estimate_count]

    log.info("=" * 60)
    log.info("COVERAGE REPORT")
    log.info("=" * 60)
    log.info(f"Tickers requested : {len(tickers)}")
    log.info(
        f"eps_history       : {len(history_count)} tickers, "
        f"{sum(history_count.values())} total rows"
    )
    log.info(
        f"eps_estimate      : {len(estimate_count)} tickers, "
        f"{sum(estimate_count.values())} total rows"
    )

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
    for ticker in sorted(tickers):
        count = history_count.get(ticker, 0)
        flag = "  *** MISSING ***" if count == 0 else ""
        log.info(f"  {ticker:<6}  {count:>4} rows{flag}")
 
    log.info("-" * 60)
    log.info("Row counts per ticker (eps_estimate / Current Quarter only):")
    for ticker in sorted(tickers):
        count = estimate_count.get(ticker, 0)
        flag = "  *** MISSING ***" if count == 0 else ""
        log.info(f"  {ticker:<6}  {count:>4} rows{flag}")
 
    log.info("=" * 60)


def main() -> None:
    log.info("=" * 60)
    log.info(f"ingest.py started at {datetime.now().isoformat()}")
    log.info("=" * 60)

    config = load_config()
    tickers = get_all_tickers(config)
    log.info(f"Loaded {len(tickers)} tickers from {CONFIG_PATH}.")

    if not DOLT_DB_PATH.exists():
        log.error(
            f"Dolt database not found at {DOLT_DB_PATH}. "
            "Run: cd data/external && dolt clone post-no-preference/earnings"
        )
        sys.exit(1)
    
    RAW_DIR.mkdir(exist_ok=True)
 
    history_rows = export_eps_history(tickers, config)
    estimate_rows = export_eps_estimate(tickers, config)
    validate_coverage(tickers, history_rows, estimate_rows)

    history_path = RAW_DIR / "eps_history.csv"
    estimate_path = RAW_DIR / "eps_estimate.csv"
 
    write_csv(history_rows, history_path)
    log.info(f"Written: {history_path}")
 
    write_csv(estimate_rows, estimate_path)
    log.info(f"Written: {estimate_path}")
 
    log.info(f"ingest.py complete at {datetime.now().isoformat()}")
 
 
if __name__ == "__main__":
    main()


    