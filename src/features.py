"""
features.py
-----------
Builds revision path features from estimate_panel.parquet for Finding 3
and the Part B XGBoost classifier.
 
For each (ticker, quarter), collapses weekly consensus snapshots in the
16 weeks before cutoff into a single row of engineered features.
 
Features:
    1. revision_slope          - linear trend of consensus over 16 weeks
    2. revision_acceleration   - second derivative
    3. direction_changes       - how many times did consensus reverse direction?
    4. final_vs_initial        - consensus at cutoff minus consensus 8 weeks prior (within window)
    5. spread_trend            - slope of analyst disagreement (high-low) over time
    6. analyst_count_trend     - slope of analyst coverage count over time
    7. weeks_of_data           - how many weekly snapshots exist for this quarter
 
Usage:
    python src/features.py
"""

import logging
import sys
import numpy as np 
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.validation import validate_point_in_time_features

ROOT        = Path(__file__).resolve().parent.parent
PROCESSED   = ROOT / "data" / "processed"
LOG_DIR     = ROOT / "logs"
 
IN_ESTIMATE = PROCESSED / "estimate_panel.parquet"
IN_PANEL    = PROCESSED / "panel.parquet"
OUT_FEATURES = PROCESSED / "features.parquet"
  
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "features.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def compute_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Fit a linear slope to (x, y) via least squares."""
    if len(x) < 2:
        return np.nan
    x = x - x.mean()
    denom = (x**2).sum()
    if denom == 0:
        return np.nan
    return (x * y).sum() / denom


def compute_features(group: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """Compute features for a single ticker-quarter using only valid PIT snapshots."""
    # Deduplicate by date (take latest if duplicates exist)
    group = group.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    
    n = len(group)
    if n == 0:
        return {
            "revision_slope": np.nan,
            "revision_acceleration": np.nan,
            "direction_changes": np.nan,
            "final_vs_initial": np.nan,
            "spread_trend": np.nan,
            "analyst_count_trend": np.nan,
            "weeks_of_data": 0,
        }

    consensus = group["consensus"].values
    spread    = group["consensus_spread"].values
    count     = group["count"].values

    weeks_of_data = n 

    # 1. Revision slope
    x = np.arange(n, dtype=float)
    revision_slope = compute_slope(x, consensus)

    # 2 & 3. Revision acceleration and Direction changes
    if n >= 3:
        diffs = np.diff(consensus)
        x2 = np.arange(len(diffs), dtype=float)
        revision_acceleration = compute_slope(x2, diffs)
        
        signs = np.sign(diffs[diffs != 0])  # ignore flat weeks
        if len(signs) >= 2:
            direction_changes = int((np.diff(signs) != 0).sum())
        else:
            direction_changes = 0
    else:
        revision_acceleration = np.nan
        direction_changes = np.nan

    # 4. Final vs Initial (8-week comparison within 16-week window)
    final_consensus = consensus[-1]
    
    # initial reference: latest snapshot on or before prediction_cutoff - 56 days
    ref_limit = cutoff - pd.Timedelta(days=56)
    ref_candidates = group[group["date"] <= ref_limit]
    if len(ref_candidates) > 0:
        initial_consensus = ref_candidates.iloc[-1]["consensus"]
        final_vs_initial = final_consensus - initial_consensus
    else:
        final_vs_initial = np.nan

    # 5. Spread trend
    spread_clean = spread[~np.isnan(spread)]
    if len(spread_clean) >= 2:
        x_spread = np.arange(len(spread_clean), dtype=float)
        spread_trend = compute_slope(x_spread, spread_clean)
    else:
        spread_trend = np.nan
    
    # 6. Analyst count trend
    count_clean = count[~np.isnan(count.astype(float))].astype(float)
    if len(count_clean) >= 2:
        x_count = np.arange(len(count_clean), dtype=float)
        analyst_count_trend = compute_slope(x_count, count_clean)
    else:
        analyst_count_trend = np.nan

    return {
        "revision_slope": revision_slope,
        "revision_acceleration": revision_acceleration,
        "direction_changes": direction_changes,
        "final_vs_initial": final_vs_initial,
        "spread_trend": spread_trend,
        "analyst_count_trend": analyst_count_trend,
        "weeks_of_data": weeks_of_data,
    }


def build_features(estimate: pd.DataFrame) -> pd.DataFrame:
    """Group by ticker-quarter and apply compute_features using strict 16-week PIT window."""
    log.info("Filtering Current Quarter consensus snapshots...")
    cq = estimate[estimate["period"] == "Current Quarter"].copy()
    
    log.info(f"  Current Quarter snapshots: {len(cq):,} rows")
    
    cq["date"] = pd.to_datetime(cq["date"])
    cq["prediction_cutoff"] = pd.to_datetime(cq["prediction_cutoff"])
    cq["period_end_date"] = pd.to_datetime(cq["period_end_date"])

    start_dates = cq["prediction_cutoff"] - pd.Timedelta(days=112)
    cq_filtered = cq[(cq["date"] >= start_dates) & (cq["date"] < cq["prediction_cutoff"])].copy()
    log.info(f"  Snapshots within 16-week pre-cutoff window: {len(cq_filtered):,} rows")

    validate_point_in_time_features(cq_filtered, "date", "prediction_cutoff")

    records = []
    groups = cq_filtered.groupby(["act_symbol", "period_end_date"])
    log.info(f"  Unique ticker-quarters to process: {groups.ngroups:,}")

    for (ticker, period_end), group in groups:
        cutoff = pd.to_datetime(period_end)
        feats = compute_features(group, cutoff)
        feats["act_symbol"] = ticker
        feats["period_end_date"] = period_end
        records.append(feats)

    features_df = pd.DataFrame(records)
    if features_df.empty:
        features_df = pd.DataFrame(columns=[
            "act_symbol", "period_end_date",
            "revision_slope", "revision_acceleration",
            "direction_changes", "final_vs_initial",
            "spread_trend", "analyst_count_trend",
            "weeks_of_data"
        ])
    else:
        features_df = features_df[[
            "act_symbol", "period_end_date",
            "revision_slope", "revision_acceleration",
            "direction_changes", "final_vs_initial",
            "spread_trend", "analyst_count_trend",
            "weeks_of_data"
        ]]
 
    log.info(f"  Features shape: {features_df.shape}")
    return features_df


def restrict_to_prediction_events(
    features: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict features to prediction events for which a realized target
    exists in the cleaned panel."""
    panel_keys = panel[["act_symbol", "period_end_date"]].drop_duplicates().copy()
    panel_keys["period_end_date"] = pd.to_datetime(panel_keys["period_end_date"])

    features = features.copy()
    features["period_end_date"] = pd.to_datetime(features["period_end_date"])

    before = len(features)

    features = features.merge(
        panel_keys,
        on=["act_symbol", "period_end_date"],
        how="inner",
        validate="one_to_one",
    )

    removed = before - len(features)

    log.info(
        f"  Restricted features to cleaned prediction-event universe: "
        f"{len(features):,} rows"
    )
    log.info(
        f"  Removed orphan feature events without realized targets: "
        f"{removed:,} rows"
    )

    return features


def validate_features(features: pd.DataFrame, panel: pd.DataFrame) -> None:
    """Check merge coverage and feature null rates."""
    log.info("Validating features...")
 
    panel_keys   = set(zip(panel["act_symbol"], pd.to_datetime(panel["period_end_date"])))
    feature_keys = set(zip(features["act_symbol"], pd.to_datetime(features["period_end_date"])))
 
    covered     = panel_keys & feature_keys
    not_covered = panel_keys - feature_keys
 
    feature_only = feature_keys - panel_keys

    log.info(
        f"  Feature ticker-quarters without panel target: {len(feature_only):,}"
    )

    if feature_only:
        log.error("  Feature table contains events outside the prediction-event universe:")
        for sym, dt in sorted(feature_only):
            log.error(f"    {sym:6s}  {dt.date()}")
        raise ValueError(
            f"Found {len(feature_only)} feature events without corresponding "
            "prediction targets in panel."
        )

    log.info(f"  Panel ticker-quarters          : {len(panel_keys):,}")
    log.info(f"  Feature ticker-quarters        : {len(feature_keys):,}")
    log.info(f"  Panel rows with features       : {len(covered):,}")
    log.info(f"  Panel rows WITHOUT features    : {len(not_covered):,}")
 
    if not_covered:
        log.warning("  Ticker-quarters in panel with no features:")
        for sym, dt in sorted(not_covered):
            log.warning(f"    {sym:6s}  {dt.date()}")
 
    log.info("  Null rates by feature:")
    feat_cols = ["revision_slope", "revision_acceleration", "direction_changes",
                 "final_vs_initial", "spread_trend", "analyst_count_trend"]
    for col in feat_cols:
        n_null = features[col].isnull().sum()
        pct    = n_null / len(features) * 100
        log.info(f"    {col:<25}  {n_null:>4} nulls  ({pct:.1f}%)")
 
    log.info("  Feature summary statistics:")
    log.info(f"\n{features[feat_cols].describe().round(4).to_string()}")


def main() -> None:
    log.info("=" * 60)
    log.info(f"features.py started at {datetime.now().isoformat()}")
    log.info("=" * 60)
 
    estimate = pd.read_parquet(IN_ESTIMATE)
    panel    = pd.read_parquet(IN_PANEL)
    log.info(f"Loaded estimate_panel: {len(estimate):,} rows")
    log.info(f"Loaded panel         : {len(panel):,} rows")
 
    features = build_features(estimate)

    features = restrict_to_prediction_events(
        features,
        panel,
    )

    validate_features(features, panel)
 
    features.to_parquet(OUT_FEATURES, index=False)
    log.info(f"Written: {OUT_FEATURES}  ({len(features):,} rows)")
 
    log.info("=" * 60)
    log.info(f"features.py complete at {datetime.now().isoformat()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()