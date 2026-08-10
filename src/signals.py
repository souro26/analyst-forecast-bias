"""
signals.py
----------
XGBoost classifier pipeline with strict walk-forward validation, inner chronological validation,
point-in-time macro scaling, model ablations, clustered category AUC confidence intervals,
and permutation feature importance.

Outputs:
    - models/xgb_model.json
    - models/xgb_results.csv
    - models/xgb_feature_importance.csv
    - models/xgb_ablation_results.csv
    - models/xgb_category_auc.csv
    - reports/temporal_audit.json
    - logs/signals.log
"""

import logging
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from src.validation import (
    validate_prediction_cutoff,
    validate_point_in_time_features,
    validate_train_test_temporal_order,
    generate_temporal_audit_report
)

ROOT         = Path(__file__).resolve().parent.parent
PROCESSED    = ROOT / "data" / "processed"
MODELS_DIR   = ROOT / "models"
LOG_DIR      = ROOT / "logs"
REPORTS_DIR  = ROOT / "reports"

PANEL_MACRO_PATH = PROCESSED / "panel_macro.parquet"
FEATURES_PATH    = PROCESSED / "features.parquet"
ESTIMATE_PATH    = PROCESSED / "estimate_panel.parquet"
MACRO_AUDIT_PATH = PROCESSED / "macro_audit.parquet"

OUT_MODEL       = MODELS_DIR / "xgb_model.json"
OUT_RESULTS     = MODELS_DIR / "xgb_results.csv"
OUT_IMPORTANCE  = MODELS_DIR / "xgb_feature_importance.csv"
OUT_ABLATION    = MODELS_DIR / "xgb_ablation_results.csv"
OUT_CAT_AUC     = MODELS_DIR / "xgb_category_auc.csv"
OUT_AUDIT       = REPORTS_DIR / "temporal_audit.json"

MODELS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

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

CATEGORY_ORDER = [
    "commodity_driven",
    "macro_rate_sensitive",
    "tech_cycle",
    "economically_cyclical",
    "regulatory_idiosyncratic",
    "defensive_baseline",
]

REVISION_FEATS = [
    "revision_slope",
    "revision_acceleration",
    "direction_changes",
    "final_vs_initial",
    "spread_trend",
    "analyst_count_trend",
    "weeks_of_data",
]

FEATURE_COLS = REVISION_FEATS + ["category_encoded", "sp500_return_pit", "vix_mean_pit"]


class SimpleStandardScaler:
    """Sklearn-like scaler fitted only on training data, robust to NaNs."""
    def __init__(self):
        self.means = {}
        self.stds = {}
        
    def fit(self, df: pd.DataFrame, cols: list[str]) -> None:
        for col in cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                self.means[col] = vals.mean()
                self.stds[col] = vals.std()
            else:
                self.means[col] = 0.0
                self.stds[col] = 1.0
                
    def transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df_trans = df.copy()
        for col in cols:
            m = self.means.get(col, 0.0)
            s = self.stds.get(col, 1.0)
            if s == 0.0 or np.isnan(s):
                s = 1.0
            df_trans[col] = (df_trans[col] - m) / s
        return df_trans


def load_and_prepare() -> pd.DataFrame:
    """Merge panel and features, encode categories deterministically, run validation."""
    log.info("Loading and merging data...")
    panel = pd.read_parquet(PANEL_MACRO_PATH)
    features = pd.read_parquet(FEATURES_PATH)

    df = panel.merge(features, on=["act_symbol", "period_end_date"], how="inner")
    log.info(f"  Merged shape: {df.shape}")

    # Explicitly define prediction_cutoff
    df["prediction_cutoff"] = df["period_end_date"]

    # Validate inputs
    validate_prediction_cutoff(df)
    
    # Exclude partial year 2026
    df = df[df["year"] < 2026].copy()

    # Drop rows missing crucial modeling variables
    before = len(df)
    df = df.dropna(subset=["beat", "revision_slope", "sp500_return_pit", "vix_mean_pit"])
    log.info(f"  Dropped {before - len(df)} rows with missing values.")

    # Deterministic Category Encoding based on CATEGORY_ORDER
    cat_to_idx = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    df["category_encoded"] = df["category"].map(cat_to_idx)
    log.info(f"  Deterministic category encoding applied: {cat_to_idx}")

    df["beat"] = df["beat"].astype(int)
    return df


def get_walk_forward_folds(df: pd.DataFrame) -> list[dict]:
    """
    Generate chronological walk-forward validation folds.
    Test years: 2020, 2021, 2022, 2023, 2024, 2025.
    For test year T:
        train: 2017 to T-2
        val: T-1
        test: T
    """
    folds = []
    test_years = sorted(df[df["year"] >= 2020]["year"].unique())
    
    for t_year in test_years:
        train_years = list(range(2017, t_year - 1))
        val_years = [t_year - 1]
        test_years_fold = [t_year]

        train_df = df[df["year"].isin(train_years)].copy()
        val_df = df[df["year"].isin(val_years)].copy()
        test_df = df[df["year"].isin(test_years_fold)].copy()

        folds.append({
            "test_year": t_year,
            "train": train_df,
            "val": val_df,
            "test": test_df
        })
    return folds


def run_ablation_experiments(folds: list[dict]) -> pd.DataFrame:
    """Run walk-forward evaluation across 8 feature configurations with uncertainty quantification."""
    log.info("Running ablation experiments...")
    
    ablation_specs = {
        "Majority baseline": [],
        "Category only": ["category_encoded"],
        "Revision only": REVISION_FEATS,
        "Macro only": ["sp500_return_pit", "vix_mean_pit"],
        "Category + Revision": ["category_encoded"] + REVISION_FEATS,
        "Revision + Macro": REVISION_FEATS + ["sp500_return_pit", "vix_mean_pit"],
        "Category + Macro": ["category_encoded", "sp500_return_pit", "vix_mean_pit"],
        "Category + Revision + Macro": FEATURE_COLS
    }

    ablation_results = []

    for name, feats in ablation_specs.items():
        log.info(f"  Evaluating configuration: {name}")
        fold_scores = []
        
        for fold in folds:
            train_df = fold["train"]
            val_df = fold["val"]
            test_df = fold["test"]

            if name == "Majority baseline":
                auc_roc = 0.5
                pr_auc = test_df["beat"].mean()
            else:
                scaler = SimpleStandardScaler()
                scale_cols = [c for c in ["sp500_return_pit", "vix_mean_pit"] if c in feats]
                
                if scale_cols:
                    scaler.fit(train_df, scale_cols)
                    train_scaled = scaler.transform(train_df, scale_cols)
                    val_scaled = scaler.transform(val_df, scale_cols)
                    test_scaled = scaler.transform(test_df, scale_cols)
                else:
                    train_scaled, val_scaled, test_scaled = train_df, val_df, test_df
                
                X_train = train_scaled[feats]
                y_train = train_scaled["beat"]
                X_val = val_scaled[feats]
                y_val = val_scaled["beat"]
                X_test = test_scaled[feats]
                y_test = test_scaled["beat"]

                n_pos = y_train.sum()
                n_neg = len(y_train) - n_pos
                scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

                model = xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    objective="binary:logistic",
                    random_state=42,
                    verbosity=0,
                    early_stopping_rounds=15
                )

                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )

                proba = model.predict_proba(X_test)[:, 1]
                
                if len(y_test.unique()) < 2:
                    auc_roc = np.nan
                    pr_auc = np.nan
                else:
                    auc_roc = roc_auc_score(y_test, proba)
                    pr_auc = average_precision_score(y_test, proba)

            fold_scores.append({
                "ROC-AUC": auc_roc,
                "PR-AUC": pr_auc,
                "prev": test_df["beat"].mean()
            })

        score_df = pd.DataFrame(fold_scores).dropna()
        n_folds = len(score_df)
        
        mean_roc = score_df["ROC-AUC"].mean()
        std_roc = score_df["ROC-AUC"].std()
        mean_pr = score_df["PR-AUC"].mean()

        # Calculate 95% Confidence Interval for the mean fold ROC-AUC
        sem_roc = std_roc / np.sqrt(n_folds) if n_folds > 0 else 0.0
        ci_lower = mean_roc - 1.96 * sem_roc
        ci_upper = mean_roc + 1.96 * sem_roc

        ablation_results.append({
            "Model": name,
            "Mean ROC-AUC": round(mean_roc, 4),
            "Mean PR-AUC": round(mean_pr, 4),
            "Std ROC-AUC": round(std_roc, 4),
            "95% CI Lower": round(ci_lower, 4),
            "95% CI Upper": round(ci_upper, 4)
        })

    ablation_df = pd.DataFrame(ablation_results)
    log.info("\n" + ablation_df.to_string(index=False))
    return ablation_df


def train_and_eval_full_model(folds: list[dict]) -> tuple[pd.DataFrame, list[xgb.XGBClassifier], pd.DataFrame]:
    """Train full Category + Revision + Macro model across folds, collect test predictions & models."""
    log.info("Evaluating full model fold-by-fold...")
    
    test_predictions = []
    models = []
    fold_reports = []

    for idx, fold in enumerate(folds):
        train_df = fold["train"]
        val_df = fold["val"]
        test_df = fold["test"]

        # Ensure temporal chronological order
        validate_train_test_temporal_order(train_df, val_df, test_df)

        # PIT macro standardization
        scaler = SimpleStandardScaler()
        scaler.fit(train_df, ["sp500_return_pit", "vix_mean_pit"])
        
        train_scaled = scaler.transform(train_df, ["sp500_return_pit", "vix_mean_pit"])
        val_scaled = scaler.transform(val_df, ["sp500_return_pit", "vix_mean_pit"])
        test_scaled = scaler.transform(test_df, ["sp500_return_pit", "vix_mean_pit"])

        X_train = train_scaled[FEATURE_COLS]
        y_train = train_scaled["beat"]
        X_val = val_scaled[FEATURE_COLS]
        y_val = val_scaled["beat"]
        X_test = test_scaled[FEATURE_COLS]
        y_test = test_scaled["beat"]

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            random_state=42,
            verbosity=0,
            early_stopping_rounds=15
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        fold_test_df = test_df.copy()
        fold_test_df["beat_proba"] = proba
        fold_test_df["beat_pred"] = preds
        test_predictions.append(fold_test_df)
        models.append(model)

        # Metrics
        auc_roc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        prev = y_test.mean()

        fold_reports.append({
            "Fold": idx + 1,
            "Train": f"{train_df['year'].min()}–{train_df['year'].max()}",
            "Validation": f"{val_df['year'].min()}",
            "Test": f"{test_df['year'].min()}",
            "N": len(test_df),
            "ROC-AUC": round(auc_roc, 4),
            "PR-AUC": round(pr_auc, 4),
            "Baseline PR-AUC": round(prev, 4)
        })

    pooled_test_df = pd.concat(test_predictions, ignore_index=True)
    report_df = pd.DataFrame(fold_reports)
    
    log.info("\n" + report_df.to_string(index=False))
    return pooled_test_df, models, report_df


def compute_permutation_importance(models: list[xgb.XGBClassifier], folds: list[dict]) -> pd.DataFrame:
    """Calculate Permutation Feature Importance averaged across walk-forward folds."""
    log.info("Calculating permutation feature importance...")
    
    importances = {col: [] for col in FEATURE_COLS}

    for idx, fold in enumerate(folds):
        model = models[idx]
        train_df = fold["train"]
        test_df = fold["test"]

        # Standardize macro features
        scaler = SimpleStandardScaler()
        scaler.fit(train_df, ["sp500_return_pit", "vix_mean_pit"])
        test_scaled = scaler.transform(test_df, ["sp500_return_pit", "vix_mean_pit"])

        X_test = test_scaled[FEATURE_COLS]
        y_test = test_scaled["beat"]

        # Baseline ROC-AUC
        base_proba = model.predict_proba(X_test)[:, 1]
        base_auc = roc_auc_score(y_test, base_proba)

        for col in FEATURE_COLS:
            X_test_perm = X_test.copy()
            X_test_perm[col] = np.random.permutation(X_test_perm[col].values)
            
            perm_proba = model.predict_proba(X_test_perm)[:, 1]
            perm_auc = roc_auc_score(y_test, perm_proba)
            
            importances[col].append(base_auc - perm_auc)

    # Compile summary
    imp_records = []
    for col in FEATURE_COLS:
        imp_records.append({
            "feature": col,
            "permutation_importance_mean": round(np.mean(importances[col]), 6),
            "permutation_importance_std": round(np.std(importances[col]), 6)
        })

    imp_df = pd.DataFrame(imp_records).sort_values("permutation_importance_mean", ascending=False)
    
    # Get standard gain-based importance from the final model to compare
    final_model = models[-1]
    gain_scores = final_model.get_booster().get_score(importance_type="gain")
    gain_df = pd.DataFrame.from_dict(gain_scores, orient="index", columns=["gain"]).reset_index().rename(columns={"index": "feature"})
    
    merged_imp = imp_df.merge(gain_df, on="feature", how="left").fillna(0.0)
    log.info("\n" + merged_imp.to_string(index=False))
    return merged_imp


def compute_category_auc_with_clustered_bootstrap(pooled_test_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate category-level AUC and 95% clustered bootstrap confidence intervals."""
    log.info("Calculating category-level AUCs with clustered bootstrap CIs...")

    results = []

    for cat in sorted(pooled_test_df["category"].unique()):
        cat_df = pooled_test_df[pooled_test_df["category"] == cat].copy()
        n = len(cat_df)
        
        if n < 30 or cat_df["beat"].nunique() < 2:
            log.info(f"  {cat:<30}  AUC=insufficient sample (N={n})")
            results.append({
                "category": cat,
                "n": n,
                "beat_rate": round(cat_df["beat"].mean(), 4) if n > 0 else 0.0,
                "auc": None,
                "lower_95": None,
                "upper_95": None
            })
            continue

        actual_auc = roc_auc_score(cat_df["beat"], cat_df["beat_proba"])
        beat_rate = cat_df["beat"].mean()

        # Clustered Bootstrap by ticker
        tickers = cat_df["act_symbol"].unique()
        boot_aucs = []
        
        np.random.seed(42)
        for b in range(1000):
            sampled_tickers = np.random.choice(tickers, size=len(tickers), replace=True)
            resampled_parts = [cat_df[cat_df["act_symbol"] == t] for t in sampled_tickers]
            boot_df = pd.concat(resampled_parts, ignore_index=True)
            
            if boot_df["beat"].nunique() >= 2:
                auc = roc_auc_score(boot_df["beat"], boot_df["beat_proba"])
                boot_aucs.append(auc)

        if len(boot_aucs) > 0:
            lower_ci = np.percentile(boot_aucs, 2.5)
            upper_ci = np.percentile(boot_aucs, 97.5)
            log.info(f"  {cat:<30} AUC={actual_auc:.4f} 95% CI=[{lower_ci:.4f}, {upper_ci:.4f}] N={n}")
        else:
            lower_ci, upper_ci = np.nan, np.nan
            log.info(f"  {cat:<30} AUC={actual_auc:.4f} CI=failed N={n}")

        results.append({
            "category": cat,
            "n": n,
            "beat_rate": round(beat_rate, 4),
            "auc": round(actual_auc, 4),
            "lower_95": round(lower_ci, 4) if not np.isnan(lower_ci) else None,
            "upper_95": round(upper_ci, 4) if not np.isnan(upper_ci) else None
        })

    return pd.DataFrame(results)


def main() -> None:
    log.info("=" * 60)
    log.info(f"signals.py started at {datetime.now().isoformat()}")
    log.info("=" * 60)

    # 1. Load data & run temporal validations
    df = load_and_prepare()

    # 2. Setup walk-forward folds
    folds = get_walk_forward_folds(df)
    log.info(f"Generated {len(folds)} walk-forward folds.")

    # 3. Generate and save Temporal Audit Report (with actual PIT macro audit inspection)
    estimate_panel = pd.read_parquet(ESTIMATE_PATH)
    macro_audit = None
    if MACRO_AUDIT_PATH.exists():
        macro_audit = pd.read_parquet(MACRO_AUDIT_PATH)
    generate_temporal_audit_report(df, estimate_panel, macro_audit, folds, str(OUT_AUDIT))

    # 4. Run Ablation Experiments (includes 95% CI of fold scores)
    ablation_df = run_ablation_experiments(folds)
    ablation_df.to_csv(OUT_ABLATION, index=False)
    log.info(f"Written: {OUT_ABLATION}")

    # 5. Train & Evaluate Full model
    pooled_test_df, models, report_df = train_and_eval_full_model(folds)

    # 6. Compute and Save Clustered Bootstrap CIs for Categories
    cat_auc_df = compute_category_auc_with_clustered_bootstrap(pooled_test_df)
    cat_auc_df.to_csv(OUT_CAT_AUC, index=False)
    log.info(f"Written: {OUT_CAT_AUC}")

    # 7. Compute Permutation Feature Importance
    imp_df = compute_permutation_importance(models, folds)
    
    # Save outputs
    models[-1].save_model(str(OUT_MODEL))
    log.info(f"Written: {OUT_MODEL}")

    pooled_test_df.to_csv(OUT_RESULTS, index=False)
    log.info(f"Written: {OUT_RESULTS}")

    imp_df.to_csv(OUT_IMPORTANCE, index=False)
    log.info(f"Written: {OUT_IMPORTANCE}")

    log.info("=" * 60)
    log.info(f"signals.py complete at {datetime.now().isoformat()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()