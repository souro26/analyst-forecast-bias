"""
regime.py
---------
Optional exploratory market-regime analysis.
Infers market regimes from S&P 500 monthly returns using a 3-state
Hidden Markov Model (bull / sideways / bear).

NOTE: The HMM is not a dependency of the primary Bayesian or XGBoost models.
It is used only for exploratory regime analysis.

Outputs:
    - data/processed/regimes.parquet   (monthly regime labels)
    - data/processed/panel_exploratory_regimes.parquet (panel merged with regimes, for exploration)
    - logs/regime.log

Usage:
    python src/regime.py
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
OUT_EXPLORATORY_PANEL = PROCESSED / "panel_exploratory_regimes.parquet"

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
    
    return model, state_sequence

def label_states(returns: pd.DataFrame, state_sequence: np.ndarray, n_states: int = 3) -> pd.DataFrame:
    """Label HMM states as bull, sideways, or bear."""
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
    
    return regime_df

def get_fred_fallback() -> pd.DataFrame:
    """Get fallback regime labels from FRED recession indicator."""
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

    # Save regimes output
    regimes_out = regime_df[["year", "month", "regime"]].copy()
    regimes_out.index.name = "date"
    regimes_out.to_parquet(OUT_REGIMES)
    log.info(f"Written: {OUT_REGIMES}")

    # Read and merge with panel for exploratory purposes
    if PANEL_PATH.exists():
        panel = pd.read_parquet(PANEL_PATH)
        log.info(f"Panel loaded: {len(panel):,} rows")
        panel = attach_regimes(panel, regime_df)
        panel.to_parquet(OUT_EXPLORATORY_PANEL, index=False)
        log.info(f"Written exploratory panel with regimes: {OUT_EXPLORATORY_PANEL}")
    else:
        log.warning("Clean panel not found; skipping exploratory panel creation.")

    log.info("=" * 60)
    log.info(f"regime.py complete at {datetime.now().isoformat()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
