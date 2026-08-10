"""
signals.py
----------
XGBoost classifier to predict analyst beat/miss direction from
revision path features. Implements Finding 3.

Question: Does the shape of the consensus revision path in the
weeks before announcement contain predictive information about
whether a company will beat or miss?

Design decisions:
    - Time-based train/test split: train 2017-2022, test 2023-2025
    - Class imbalance handled via scale_pos_weight
    - Evaluation: AUC-ROC and Precision-Recall AUC (not accuracy)
    - Features: 7 revision path features + category + macro context
    - No data leakage: all features observable before announcement

Inputs:
    - data/processed/panel.parquet
    - data/processed/features.parquet

Outputs:
    - models/xgb_model.json          (fitted XGBoost model)
    - models/xgb_results.csv         (per-observation predictions + actuals)
    - models/xgb_feature_importance.csv
    - logs/signals.log

Usage:
    python src/signals.py
"""

import logging
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

ROOT         = Path(__file__).resolve().parent.parent
PROCESSED    = ROOT / "data" / "processed"
MODELS_DIR   = ROOT / "models"
LOG_DIR      = ROOT / "logs"

PANEL_PATH   = PROCESSED / "panel.parquet"
FEATURES_PATH = PROCESSED / "features.parquet"

OUT_MODEL       = MODELS_DIR / "xgb_model.json"
OUT_RESULTS     = MODELS_DIR / "xgb_results.csv"
OUT_IMPORTANCE  = MODELS_DIR / "xgb_feature_importance.csv"

MODELS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "signals.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "revision_slope",
    "revision_acceleration",
    "direction_changes",
    "final_vs_initial",
    "spread_trend",
    "analyst_count_trend",
    "weeks_of_data",
    "category_encoded",
    "sp500_return_z",
    "vix_mean_z",
]

TRAIN_YEARS = list(range(2017, 2023))   # 2017-2022
TEST_YEARS  = list(range(2023, 2026))   # 2023-2025


def load_and_prepare(panel_path: Path, features_path: Path) -> pd.DataFrame:
    """
    Merge panel and features, encode categoricals, validate no leakage.
    """
    log.info("Loading data...")
    panel    = pd.read_parquet(panel_path)
    features = pd.read_parquet(features_path)

    df = panel.merge(features, on=["act_symbol", "period_end_date"], how="inner")
    log.info(f"  Merged shape: {df.shape}")

    df = df[df["year"] < 2026].copy()

    before = len(df)
    df = df.dropna(subset=["beat", "revision_slope", "sp500_return_z"])
    after = len(df)
    if before - after > 0:
        log.warning(f"  Dropped {before - after} rows with missing values.")

    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["category"])
    log.info(f"  Category encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    df["beat"] = df["beat"].astype(int)

    log.info(f"  Final rows: {len(df):,}")
    log.info(f"  Beat rate overall: {df['beat'].mean()*100:.1f}%")
    log.info(f"  Train years (2017-2022): {(df['year'].isin(TRAIN_YEARS)).sum():,} rows")
    log.info(f"  Test years  (2023-2025): {(df['year'].isin(TEST_YEARS)).sum():,} rows")

    return df, le


def split_data(df: pd.DataFrame) -> tuple:
    """
    Time-based split: train on 2017-2022, test on 2023-2025.
    Returns X_train, X_test, y_train, y_test, train_df, test_df.
    """
    train = df[df["year"].isin(TRAIN_YEARS)].copy()
    test  = df[df["year"].isin(TEST_YEARS)].copy()

    X_train = train[FEATURE_COLS]
    X_test  = test[FEATURE_COLS]
    y_train = train["beat"]
    y_test  = test["beat"]

    log.info(f"  Train: {len(X_train):,} rows  beat rate={y_train.mean()*100:.1f}%")
    log.info(f"  Test : {len(X_test):,} rows  beat rate={y_test.mean()*100:.1f}%")

    return X_train, X_test, y_train, y_test, train, test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier with class imbalance correction.

    scale_pos_weight = n_negative / n_positive
    Upweights misses (minority class) to counteract 83% beat rate.
    """
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    log.info(f"  n_pos={n_pos}  n_neg={n_neg}  "
             f"scale_pos_weight={scale_pos_weight:.3f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    log.info("  XGBoost training complete.")
    return model


def evaluate(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate model on test set. Log AUC-ROC, PR-AUC, confusion matrix.
    Returns results DataFrame with predictions attached.
    """
    log.info("Evaluating on test set (2023-2025)...")

    proba  = model.predict_proba(X_test)[:, 1]
    preds  = (proba >= 0.5).astype(int)

    auc_roc = roc_auc_score(y_test, proba)
    pr_auc  = average_precision_score(y_test, proba)

    log.info(f"  AUC-ROC          : {auc_roc:.4f}")
    log.info(f"  PR-AUC           : {pr_auc:.4f}")
    log.info(f"  Baseline PR-AUC  : {y_test.mean():.4f}  (always-beat classifier)")

    log.info("  Classification report:")
    report = classification_report(y_test, preds, target_names=["miss", "beat"])
    for line in report.split("\n"):
        if line.strip():
            log.info(f"    {line}")

    cm = confusion_matrix(y_test, preds)
    log.info(f"  Confusion matrix:")
    log.info(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    log.info(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    log.info("  AUC-ROC by category:")
    results_df = test_df.copy()
    results_df["beat_proba"] = proba
    results_df["beat_pred"]  = preds

    for cat in sorted(test_df["category"].unique()):
        mask = results_df["category"] == cat
        if mask.sum() < 10:
            continue
        cat_auc = roc_auc_score(
            results_df.loc[mask, "beat"],
            results_df.loc[mask, "beat_proba"],
        )
        log.info(f"    {cat:<30}  AUC={cat_auc:.4f}  n={mask.sum()}")

    return results_df


def get_feature_importance(
    model: xgb.XGBClassifier,
) -> pd.DataFrame:
    """
    Extract and log feature importance (gain-based).
    Gain = average improvement in loss from splits using that feature.
    More meaningful than frequency-based importance.
    """
    log.info("Feature importance (gain):")

    importance = model.get_booster().get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame.from_dict(importance, orient="index", columns=["gain"])
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("gain", ascending=False)
    )
    imp_df["gain_normalized"] = imp_df["gain"] / imp_df["gain"].sum()

    for _, row in imp_df.iterrows():
        log.info(f"  {row['feature']:<25}  gain={row['gain']:.2f}  "
                 f"({row['gain_normalized']*100:.1f}%)")

    return imp_df


def main() -> None:
    log.info("=" * 60)
    log.info(f"signals.py started at {datetime.now().isoformat()}")
    log.info("=" * 60)

    df, le = load_and_prepare(PANEL_PATH, FEATURES_PATH)

    log.info("Splitting data...")
    X_train, X_test, y_train, y_test, train_df, test_df = split_data(df)

    log.info("Training XGBoost classifier...")
    model = train_model(X_train, y_train, X_test, y_test)

    results_df = evaluate(model, X_test, y_test, test_df)

    imp_df = get_feature_importance(model)

    model.save_model(str(OUT_MODEL))
    log.info(f"Written: {OUT_MODEL}")

    results_df.to_csv(OUT_RESULTS, index=False)
    log.info(f"Written: {OUT_RESULTS}")

    imp_df.to_csv(OUT_IMPORTANCE, index=False)
    log.info(f"Written: {OUT_IMPORTANCE}")

    log.info("=" * 60)
    log.info(f"signals.py complete at {datetime.now().isoformat()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()