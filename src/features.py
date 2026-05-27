import pandas as pd
import numpy as np 
import os
from fredapi import Fred 
from dotenv import load_dotenv

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

def load_panel(filepath):
    df = pd.read_parquet(filepath)
    return df

def add_recession_indicator(df):
    """Fetch USREC from FRED and tag each row as recession or expansion."""
    fred = Fred(api_key = FRED_API_KEY)
    usrec = fred.get_series('USREC', observation_start='1993-01-01')
    usrec = usrec.resample('QE').max()
    usrec.index = usrec.index.to_period('Q')

    df['period'] = df['date'].dt.to_period('Q')
    df['recession'] = df['period'].map(usrec).fillna(0).astype(int)
    df = df.drop(columns=['period'])
    return df

def add_sector_index(df):
    """Sector saved as integer for bayesian models."""
    sectors = sorted(df['sector'].unique())
    sector_map = {s: i for i,s in enumerate(sectors)}
    df['sector_idx'] = df['sector'].map(sector_map)
    return df, sector_map

def add_lag_features(df):
    """Add lag and rolling features per company."""
    df = df.sort_values(['symbol', 'date'])

    df['lag1_error'] = df.groupby('symbol')['forecast_error'].shift(1)
    df['lag2_error'] = df.groupby('symbol')['forecast_error'].shift(2)

    df['rolling4_mean_error'] = (
        df.groupby('symbol')['forecast_error']
        .transform(lambda x: x.shift(1).rolling(4).mean())
    )
    df['rolling4_std_error'] = (
        df.groupby('symbol')['forecast_error']
        .transform(lambda x: x.shift(1).rolling(4).std())
    )
    return df

def add_eps_growth(df):
    """Was EPS growing going into this quarter."""
    df = df.sort_values(['symbol', 'date'])
    df['eps_growth'] = df.groupby('symbol')['epsActual'].pct_change()
    return df

def add_beat_streak(df):
    """How many consecutive quarters has this company beaten estimates."""
    df = df.sort_values(['symbol', 'date'])

    def streak(x):
        streaks = []
        count = 0
        for val in x:
            if val > 0:
                count += 1
            else:
                count = 0
            streaks.append(count)
        return pd.Series(streaks, index=x.index)

    df['beat_streak'] = df.groupby('symbol')['forecast_error'].transform(streak)
    df['beat_streak'] = df.groupby('symbol')['beat_streak'].shift(1)
    return df

def add_surprise_pct(df):
    """Percentage surprise relative to absolute estimate."""
    df['surprise_pct'] = np.where(
        df['epsEstimated'].abs() > 0.01,
        df['forecast_error'] / df['epsEstimated'].abs(),
        np.nan
    )
    return df

def add_beat_target(df):
    """Binary target: 1 if beat estimates, 0 if missed."""
    df['beat'] = (df['forecast_error'] > 0).astype(int)
    return df

def main():
    df = load_panel("data/processed/panel.parquet")
    print(f"Loaded {len(df)} rows")

    df = add_recession_indicator(df)
    print(f"Recession quarters tagged: {df['recession'].sum()}")

    df, sector_map = add_sector_index(df)
    print(f"Sector map: {sector_map}")

    df = add_lag_features(df)
    df = add_eps_growth(df)
    df = add_beat_streak(df)
    df = add_surprise_pct(df)
    df = add_beat_target(df)

    df = df.reset_index(drop=True)

    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df[['symbol', 'date', 'forecast_error', 'recession',
              'sector_idx', 'lag1_error', 'beat_streak', 'beat']].head(10))

    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet("data/processed/features.parquet", index=False)
    print(f"\nSaved {len(df)} rows to data/processed/features.parquet")


if __name__ == "__main__":
    main()
