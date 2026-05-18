import requests
import time
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FMP_API_KEY")

TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "META", "NVDA",
    # Financials
    "JPM", "GS", "BAC", "WFC", "MS",
    # Healthcare
    "JNJ", "PFE", "UNH", "MRK", "ABT",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Consumer Discretionary
    "AMZN", "HD", "MCD", "NKE", "SBUX",
    # Consumer Staples
    "WMT", "PG", "KO", "PEP", "CL",
    # Industrials
    "BA", "CAT", "GE", "MMM", "HON",
    # Utilities
    "NEE", "DUK", "SO", "AEP", "EXC",
    # Real Estate
    "AMT", "PLD", "CCI", "EQIX", "PSA",
    # Materials
    "LIN", "APD", "ECL", "DD", "NEM",
]

def get_company_data(ticker, api_key):
    """Fetch earnings history and company profile for one ticker."""

    earnings_url = (
        f"https://financialmodelingprep.com/stable/earnings"
        f"?symbol={ticker}&apikey={api_key}"
    )
    earnings_resp = requests.get(earnings_url)
    if not earnings_resp.text.strip():
        time.sleep(5)
        earnings_resp = requests.get(earnings_url)
    earnings = earnings_resp.json()

    profile_url = (
        f"https://financialmodelingprep.com/stable/profile"
        f"?symbol={ticker}&apikey={api_key}"
    )
    profile_resp = requests.get(profile_url)
    if not profile_resp.text.strip():
        time.sleep(5)
        profile_resp = requests.get(profile_url)
    profile = profile_resp.json()

    sector = None
    industry = None
    if profile and isinstance(profile, list):
        sector = profile[0].get("sector")
        industry = profile[0].get("industry")

    rows = []
    for row in earnings:
        row["sector"] = sector
        row["industry"] = industry
        rows.append(row)

    return rows

def save_raw(data, filename):
    """Save list of dicts to data/raw/ as CSV."""
    os.makedirs("data/raw", exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(f"data/raw/{filename}", index=False)
    print(f"Saved {len(df)} rows to data/raw/{filename}")

def main():
    all_earnings = []

    for i, ticker in enumerate(TICKERS):
        print(f"Fetching {ticker} ({i+1}/{len(TICKERS)})...")
        try:
            rows = get_company_data(ticker, API_KEY)
            all_earnings.extend(rows)
        except Exception as e:
            print(f"Failed {ticker}: {e}")

        time.sleep(2)

    save_raw(all_earnings, "earnings_raw.csv")
    print("Done.")

if __name__ == "__main__":
    main()





