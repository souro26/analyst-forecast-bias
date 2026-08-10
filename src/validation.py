"""
validation.py
-------------
Temporal data validation layer for financial research.
Ensures point-in-time invariants and prevents look-ahead leakage.
"""

import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

def validate_prediction_cutoff(df: pd.DataFrame) -> None:
    """
    Verify that the dataset contains necessary columns, prediction_cutoff equals
    period_end_date, and dates are valid.
    """
    required_cols = ["act_symbol", "period_end_date", "prediction_cutoff"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing.")

    # Check for nulls in critical columns
    for col in required_cols:
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            raise ValueError(f"Column '{col}' contains {null_count} null value(s).")

    # Convert to datetime if not already
    period_ends = pd.to_datetime(df["period_end_date"])
    cutoffs = pd.to_datetime(df["prediction_cutoff"])

    # Verify prediction_cutoff == period_end_date
    mismatches = (period_ends != cutoffs)
    if mismatches.any():
        num_mismatches = mismatches.sum()
        offending = df[mismatches][["act_symbol", "period_end_date", "prediction_cutoff"]].head()
        raise ValueError(
            f"Prediction cutoff must equal period_end_date. Found {num_mismatches} mismatches. "
            f"Examples:\n{offending}"
        )

    log.info("validate_prediction_cutoff: PASSED. All columns present and cutoff matches period end.")


def validate_point_in_time_features(
    df: pd.DataFrame,
    timestamp_col: str,
    cutoff_col: str,
    context_cols: list[str] = None
) -> None:
    """
    Validate that every observation used for a prediction is strictly before the prediction cutoff.
    If violations exist, fail loudly with an audit summary.
    """
    if df.empty:
        return

    t_vals = pd.to_datetime(df[timestamp_col])
    c_vals = pd.to_datetime(df[cutoff_col])

    violations = (t_vals >= c_vals)
    if violations.any():
        num_violations = violations.sum()
        pct = (num_violations / len(df)) * 100
        max_leakage = (t_vals - c_vals).max()
        
        # Grab context columns for error reporting
        if context_cols is None:
            context_cols = ["act_symbol", "period_end_date"]

        # Only include context columns that actually exist in the dataframe.
        offending_cols = [
            col for col in context_cols + [timestamp_col, cutoff_col]
            if col in df.columns
        ]

        offending_sample = df.loc[violations, offending_cols].head(10)
        
        msg = (
            f"TEMPORAL LEAKAGE DETECTED! {num_violations} violations found ({pct:.2f}% of data).\n"
            f"Maximum future leakage duration: {max_leakage}\n"
            f"Offending examples:\n{offending_sample.to_string()}"
        )
        log.error(msg)
        raise ValueError(msg)

    log.info(f"validate_point_in_time_features: PASSED for {timestamp_col} < {cutoff_col}.")


def validate_train_test_temporal_order(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """
    Validate that training, validation, and testing sets are strictly ordered in time.
    """
    if train_df.empty or test_df.empty:
        return

    train_max = pd.to_datetime(train_df["prediction_cutoff"]).max()
    test_min = pd.to_datetime(test_df["prediction_cutoff"]).min()

    if val_df is not None and not val_df.empty:
        val_min = pd.to_datetime(val_df["prediction_cutoff"]).min()
        val_max = pd.to_datetime(val_df["prediction_cutoff"]).max()

        if train_max >= val_min:
            raise ValueError(
                f"Temporal order violated: Train max date ({train_max.date()}) "
                f"is not strictly before Val min date ({val_min.date()})."
            )
        if val_max >= test_min:
            raise ValueError(
                f"Temporal order violated: Val max date ({val_max.date()}) "
                f"is not strictly before Test min date ({test_min.date()})."
            )
    else:
        if train_max >= test_min:
            raise ValueError(
                f"Temporal order violated: Train max date ({train_max.date()}) "
                f"is not strictly before Test min date ({test_min.date()})."
            )

    log.info("validate_train_test_temporal_order: PASSED.")


def generate_temporal_audit_report(
    model_df: pd.DataFrame,
    estimate_df: pd.DataFrame,
    macro_audit_df: pd.DataFrame,
    folds: list[dict],
    out_path: str
) -> None:
    """
    Generate a temporal audit report.

    The raw estimate panel contains historical snapshots that may occur
    after a prediction cutoff. Those observations are expected to exist
    in the source data and are NOT, by themselves, evidence of leakage.

    Leakage is determined by the feature-construction PIT validation in
    features.py. This audit therefore reports raw-source observations
    separately from actual model-pipeline temporal violations.
    """

    # ------------------------------------------------------------
    # 1. Actual prediction-event universe
    # ------------------------------------------------------------
    prediction_events = model_df[
        ["act_symbol", "period_end_date"]
    ].drop_duplicates()

    total_prediction_events = len(prediction_events)

    events_with_missing_cutoff = int(
        estimate_df["prediction_cutoff"].isnull().sum()
    )

    # ------------------------------------------------------------
    # 2. Raw estimate-panel observations
    #
    # These are SOURCE DATA observations. They are intentionally
    # reported separately because future snapshots can exist in the
    # raw panel without being used as model features.
    # ------------------------------------------------------------
    est_dates = pd.to_datetime(estimate_df["date"])
    est_cutoffs = pd.to_datetime(estimate_df["prediction_cutoff"])

    raw_future_feature_observations = int(
        (est_dates > est_cutoffs).sum()
    )

    raw_cutoff_equal_observations = int(
        (est_dates == est_cutoffs).sum()
    )

    # ------------------------------------------------------------
    # 3. Macro PIT audit
    #
    # macro_audit.parquet records the latest market observation
    # actually available strictly before each prediction cutoff.
    # ------------------------------------------------------------
    future_macro_observations = 0

    if macro_audit_df is not None and not macro_audit_df.empty:
        gspc_dates = pd.to_datetime(
            macro_audit_df["latest_gspc_date"]
        )
        vix_dates = pd.to_datetime(
            macro_audit_df["latest_vix_date"]
        )
        cutoffs = pd.to_datetime(
            macro_audit_df["prediction_cutoff"]
        )

        gspc_violations = gspc_dates >= cutoffs
        vix_violations = vix_dates >= cutoffs

        future_macro_observations = int(
            (gspc_violations | vix_violations).sum()
        )

    # ------------------------------------------------------------
    # 4. Walk-forward temporal ordering
    # ------------------------------------------------------------
    train_test_temporal_violations = 0

    for fold_idx, fold in enumerate(folds):
        try:
            validate_train_test_temporal_order(
                fold["train"],
                fold.get("val"),
                fold["test"]
            )
        except ValueError as e:
            log.warning(
                f"Fold {fold_idx} temporal order violation: {e}"
            )
            train_test_temporal_violations += 1

    # ------------------------------------------------------------
    # 5. Audit report
    # ------------------------------------------------------------
    report = {
        # Actual source prediction-event universe.
        "total_prediction_events": total_prediction_events,

        # Missing cutoff values in the source estimate panel.
        "events_with_missing_cutoff": events_with_missing_cutoff,

        # IMPORTANT:
        # These are raw-source observations, NOT model features.
        # They are retained for transparency and must not be
        # interpreted as model leakage.
        "raw_future_estimate_observations":
            raw_future_feature_observations,

        "raw_cutoff_equal_estimate_observations":
            raw_cutoff_equal_observations,

        # Actual PIT macro violations.
        "future_macro_observations":
            future_macro_observations,

        # Feature-level PIT validation is performed separately by
        # validate_point_in_time_features() in features.py.
        "feature_pit_validation":
            "performed_in_features.py",

        # Walk-forward validation.
        "train_test_temporal_violations":
            train_test_temporal_violations,
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(exist_ok=True, parents=True)

    with open(out_file, "w") as f:
        json.dump(report, f, indent=4)

    log.info(
        f"Temporal audit report written to {out_path}:\n"
        f"{json.dumps(report, indent=4)}"
    )
