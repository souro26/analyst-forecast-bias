import pandas as pd
import numpy as np 
import os

def load_raw(filepath):
    """Load raw earnings csv"""
    df = pd.read_csv(filepath)
    return df

def clean(df):
    """Clean the dataframe"""
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'].dt.year >= 1993) & (df['date'].dt.year <= 2025)]
    df = df.dropna(subset = ['epsActual', 'epsEstimated'])
    df['forecast_error'] = df['epsActual'] - df['epsEstimated']

    lower = df['forecast_error'].quantile(0.01)
    upper = df['forecast_error'].quantile(0.99)
    df['forecast_error'] = df['forecast_error'].clip(lower, upper)

    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter

    df = df.reset_index(drop = True)

    return df

def save_processed(df, filepath):
    """Save cleaned panel to parquet."""
    os.makedirs(os.path.dirname(filepath), exist_ok = True)
    df.to_parquet(filepath, index= False)
    print(f"Saved {len(df)} rows to {filepath}")

def main():
    raw = load_raw("data/raw/earnings_raw.csv")
    print(f"Loaded {len(raw)} raw rows")
    
    cleaned = clean(raw)
    print(f"After cleaning: {len(cleaned)} rows")
    print(cleaned[['symbol', 'date', 'epsActual', 'epsEstimated', 'forecast_error']].head())
    
    save_processed(cleaned, "data/processed/panel.parquet")
    print("Done.")

if __name__ == "__main__":
    main()
