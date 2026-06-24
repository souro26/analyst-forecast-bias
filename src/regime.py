"""
regime.py
---------
Infers market regimes from S&P 500 monthly returns using a 3-state
Hidden Markov Model (bull / sideways / bear). Attaches regime labels
to panel.parquet as a new column.

Steps:
    1. Download S&P 500 monthly returns via yfinance
    2. Fit 3-state Gaussian HMM (hmmlearn)
    3. Interpret and validate state labels
    4. Map monthly regimes to company-quarters in panel.parquet
    5. Write data/processed/regimes.parquet and updated panel.parquet

Fallback:
    If HMM convergence fails, uses FRED USREC recession indicator
    (binary: recession / expansion).

Usage:
    python src/regime.py

Outputs:
    - data/processed/regimes.parquet   (monthly regime labels)
    - data/processed/panel.parquet     (updated with regime column)
    - logs/regime.log
"""

import logging
import sys
import warnings
import numpy as np 
import pandas as pd
import yfinance as yf
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
LOG_DIR = ROOT / "logs"

OUT_REGIMES = PROCESSED / "regimes.parquet"
PANEL_PATH = PROCESSED / "panel.parquet"

LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "regime.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

def get_sp500_returns(start: str = "2016-01-01", end: str = "2026-06-01") -> pd.DataFrame:
    """Download monthly S&P 500 prices and compute month-over-month returns."""
    log.info("Downloading S&P 500 monthly prices via yfinance...")
    raw = yf.download("^GSPC", start=start, end=end, interval="1mo", auto_adjust=True, progress=False)

    prices = raw["Close"].squeeze()
    prices.index = pd.to_datetime(prices.index).to_period("M").to_timestamp()
    returns = prices.pct_change().dropna()
    
    log.info(f"  S&P 500 returns: {len(returns)} months  "
             f"({returns.index[0].date()} to {returns.index[-1].date()})")
    log.info(f"  Mean monthly return : {returns.mean():.4f}")
    log.info(f"  Std monthly return  : {returns.std():.4f}")
    log.info(f"  Min                 : {returns.min():.4f}  "
             f"Max: {returns.max():.4f}")
    
    return returns.to_frame(name="monthly_return")

def fit_hmm(returns: pd.DataFrame, n_states: int = 3, n_iter: int = 100, random_state: int = 42) -> tuple:
    """Fit a Gaussian HMM on monthly returns."""
    log.info(f"Fitting {n_states}-state Gaussian HMM  (n_iter={n_iter})...")
    
    X = returns["monthly_return"].values.reshape(-1, 1)
    
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)
    
    if not model.monitor_.converged:
        log.warning("  HMM did not converge — consider increasing n_iter or using FRED fallback.")
    else:
        log.info("  HMM converged.")
    
    state_sequence = model.predict(X)
    
    # Log what each raw state looks like
    log.info("  Raw state statistics (before labelling):")
    for s in range(n_states):
        mask = state_sequence == s
        state_returns = returns["monthly_return"].values[mask]
        log.info(f"    State {s}: mean={state_returns.mean():.4f}  "
                 f"std={state_returns.std():.4f}  n={mask.sum()}")
    
    return model, state_sequence

def label_states(returns: pd.DataFrame, state_sequence: np.ndarray, n_states: int = 3) -> pd.DataFrame:
    """ Assign bull/sideways/bear labels to HMM states based on mean return."""
    log.info("Labelling HMM states ...")

    state_means = {}
    for s in range(n_states):
        mask = state_sequence == s
        state_means[s] = returns["monthly_return"].values[mask].mean()

    ranked = sorted(state_means.keys(), key = lambda s: state_means[s], reverse = True)
    state_map = {
        ranked[0]: "bull",
        ranked[1]: "sideways",
        ranked[2]: "bear",
    }

    log.info("State label assignments:")
    for raw_state, label in state_map.items():
        log.info(f"    State {raw_state} → {label}  "
                f"(mean monthly return = {state_means[raw_state]:.4f})")

    regime_df = returns.copy()
    regime_df["state_raw"] = state_sequence
    regime_df["regime"] = regime_df["state_raw"].map(state_map)
    regime_df["year"] = regime_df.index.year
    regime_df["month"] = regime_df.index.month

    log.info("Validation against known regime periods:")
    
    checks = {
        "2020-03": "bear",
        "2020-04": "bear",
        "2021-06": "bull",
        "2022-06": "bear",
        "2023-06": "bull",
    }
    for ym, expected in checks.items():
        yr, mo = int(ym.split("-")[0]), int(ym.split("-")[1])
        row = regime_df[(regime_df["year"] == yr) & (regime_df["month"] == mo)]
        if len(row) ==0:
             log.warning(f"    {ym}: not found in regime series")
             continue
        actual = row["regime"].values[0]
        status = "OK" if actual == expected else "MISMATCH"
        log.info(f"    {ym}: expected={expected}  actual={actual}  [{status}]")
    
    return regime_df

def get_fred_fallback() -> pd.DataFrame:
    """Fallback regime labels from FRED USREC recession indicator."""
    log.info("Using FRED USREC fallback...")

    try:
        from fredapi import Fred 
        fred = Fred(api_key=os.getenv("FRED_API_KEY"))
        usrec= fred.get_series("USREC", observation_Start= "2016-01-01")

        regime_df = usrec.to_frame(name="usrec")
        regime_df.index = pd.to_datetime(regime_df.index).to_period("M").to_timestamp()
        regime_df["regime"] = regime_df["usrec"].map({1: "bear", 0: "bull"})
        regime_df["year"]   = regime_df.index.year
        regime_df["month"]  = regime_df.index.month
        
        log.info(f"  FRED USREC loaded: {len(regime_df)} months")
        log.info(f"  Bear months : {(regime_df['regime'] == 'bear').sum()}")
        log.info(f"  Bull months : {(regime_df['regime'] == 'bull').sum()}")
        
        return regime_df
    
    except Exception as e:
        log.error(f"  FRED fallback failed: {e}")
        log.error("  No regime labels available — stopping.")
        raise

def attach_regimes(panel: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    """Map monthly regime labels onto company-quarters in panel."""
    log.info("Attaching regime labels to panel...")

    regime_lookup = (
        regime_df[["year", "month", "regime"]]
        .drop_duplicates()
        .set_index(["year", "month"])["regime"]
    )

    panel = panel.copy()
    panel["year_month_key"] = list(
        zip(panel["period_end_date"].dt.year,
            panel["period_end_date"].dt.month)
    )

    panel["regime"] = panel["year_month_key"].map(
        lambda ym: regime_lookup.get(ym, np.nan)
    )
    panel = panel.drop(columns=["year_month_key"])

    n_null = panel["regime"].isnull().sum()
    if n_null > 0:
        log.warning(f"  {n_null} rows could not be assigned a regime label.")
    else:
        log.info("  All rows assigned a regime label.")

    log.info("  Regime distribution in panel:")
    for regime, count in panel["regime"].value_counts().items():
        pct = count / len(panel) * 100
        log.info(f"    {regime:<10} {count:>4} rows  ({pct:.1f}%)")

    return panel
                
def main() -> None:
    log.info(f"regime.py started at {datetime.now().isoformat()}")
    returns = get_sp500_returns()
    model, state_sequence = fit_hmm(returns)
    if model.monitor_.converged:
        regime_df = label_states(returns, state_sequence)
    else:
        log.warning("HMM did not converge - switching to FRED fallback.")
        regime_df = get_fred_fallback()

    panel = pd.read_parquet(PANEL_PATH)
    log.info(f"Panel loaded: {len(panel):,} rows")
    panel = attach_regimes(panel, regime_df)

    regimes_out = regime_df[["year", "month", "regime"]].copy()
    regimes_out.index.name = "date"
    regimes_out.to_parquet(OUT_REGIMES)
    log.info(f"Written: {OUT_REGIMES}")

    panel.to_parquet(PANEL_PATH, index=False)
    log.info(f"Updated: {PANEL_PATH}  (now includes regime column)")

    log.info("=" * 60)
    log.info(f"regime.py complete at {datetime.now().isoformat()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
