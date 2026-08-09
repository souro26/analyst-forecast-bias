"""
models.py
---------
Hierarchical Bayesian model of analyst forecast bias.

Structure:
    Level 1 - grand mean (mu_global)
    Level 2 - category effects (alpha_category)
    Level 3 - continuous macro effects (beta_sp500, beta_vix)
    Likelihood - Student-t (justified by EDA QQ plots)

Why continuous macro predictors instead of discrete regime labels:
    With only 9 years of data, discrete bull/bear/sideways labels
    give the model 3-4 data points per regime — not enough to
    estimate regime effects. Continuous predictors (quarterly S&P
    return and average VIX) give every observation its own macro
    context (2,455 data points), making macro effects estimable
    and quantitative.

Model:
    mu_global         ~ Normal(0, 0.5)
    sigma_category    ~ HalfNormal(0.3)
    alpha_category[c] ~ Normal(0, sigma_category)

    beta_sp500        ~ Normal(0, 0.3)   # effect of quarterly S&P return (z-scored)
    beta_vix          ~ Normal(0, 0.3)   # effect of quarterly VIX (z-scored)

    mu[i] = mu_global
           + alpha_category[cat_idx[i]]
           + beta_sp500 * sp500_return_z[i]
           + beta_vix   * vix_mean_z[i]

    nu    ~ Gamma(2, 0.1)
    sigma ~ HalfNormal(0.3)
    y[i]  ~ StudentT(nu, mu[i], sigma)

Inputs:
    - data/processed/panel.parquet       (includes sp500_return_z, vix_mean_z)
    - data/processed/features.parquet

Outputs:
    - models/trace.nc
    - models/summary.csv
    - logs/models.log

Usage:
    python src/models.py
"""

import logging
import sys
import warnings
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path
from datetime import datetime

ROOT          = Path(__file__).resolve().parent.parent
PROCESSED     = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
LOG_DIR       = ROOT / "logs"

PANEL_PATH    = PROCESSED / "panel.parquet"
FEATURES_PATH = PROCESSED / "features.parquet"
TRACE_PATH    = MODELS_DIR / "trace.nc"
SUMMARY_PATH  = MODELS_DIR / "summary.csv"

MODELS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "models.log"),
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


def load_and_prepare(panel_path: Path, features_path: Path) -> pd.DataFrame:
    """
    Load panel and features, merge, validate macro columns present.
    Excludes 2026 (partial year). Drops rows missing target.
    """
    log.info("Loading data...")
    panel    = pd.read_parquet(panel_path)
    features = pd.read_parquet(features_path)

    df = panel.merge(features, on=["act_symbol", "period_end_date"], how="left")
    log.info(f"  Merged shape: {df.shape}")

    for col in ["sp500_return_z", "vix_mean_z"]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in panel. "
                "Run macro.py before models.py."
            )

    df = df[df["year"] < 2026].copy()
    log.info(f"  After excluding 2026: {len(df):,} rows")

    before = len(df)
    df = df.dropna(subset=["forecast_error_winsorized"])
    after = len(df)
    if before - after > 0:
        log.warning(f"  Dropped {before - after} rows missing target.")

    df["category_idx"] = pd.Categorical(
        df["category"], categories=CATEGORY_ORDER
    ).codes

    log.info(f"  Final modelling rows: {len(df):,}")
    log.info("  Category index mapping:")
    for i, c in enumerate(CATEGORY_ORDER):
        n = (df["category_idx"] == i).sum()
        log.info(f"    {i}  {c:<30}  n={n}")

    log.info("  Macro predictor summary:")
    log.info(f"    sp500_return_z  mean={df['sp500_return_z'].mean():.4f}  "
             f"std={df['sp500_return_z'].std():.4f}  "
             f"min={df['sp500_return_z'].min():.4f}  "
             f"max={df['sp500_return_z'].max():.4f}")
    log.info(f"    vix_mean_z      mean={df['vix_mean_z'].mean():.4f}  "
             f"std={df['vix_mean_z'].std():.4f}  "
             f"min={df['vix_mean_z'].min():.4f}  "
             f"max={df['vix_mean_z'].max():.4f}")

    return df


def build_model(df: pd.DataFrame) -> pm.Model:
    """
    Build the hierarchical Bayesian model with continuous macro predictors.

    beta_sp500: effect of a 1-SD increase in quarterly S&P 500 return
                on forecast error. Positive = analysts underestimate more
                in strong market quarters.
    beta_vix:   effect of a 1-SD increase in quarterly VIX on forecast
                error. Positive = analysts underestimate more in high
                uncertainty quarters.
    """
    n_categories = len(CATEGORY_ORDER)

    cat_idx    = df["category_idx"].values
    sp500_z    = df["sp500_return_z"].values.astype(float)
    vix_z      = df["vix_mean_z"].values.astype(float)
    y          = df["forecast_error_winsorized"].values.astype(float)

    log.info("Building hierarchical model...")
    log.info(f"  n_categories={n_categories}  n_obs={len(y)}")

    with pm.Model() as model:
        mu_global      = pm.Normal("mu_global", mu=0.0, sigma=0.5)
        sigma_category = pm.HalfNormal("sigma_category", sigma=0.3)

        alpha_category = pm.Normal(
            "alpha_category",
            mu=0.0,
            sigma=sigma_category,
            shape=n_categories,
        )

        beta_sp500 = pm.Normal("beta_sp500", mu=0.0, sigma=0.3)
        beta_vix   = pm.Normal("beta_vix",   mu=0.0, sigma=0.3)

        mu = (
            mu_global
            + alpha_category[cat_idx]
            + beta_sp500 * sp500_z
            + beta_vix   * vix_z
        )

        nu    = pm.Gamma("nu", alpha=2, beta=0.1)
        sigma = pm.HalfNormal("sigma", sigma=0.3)

        pm.StudentT(
            "y",
            nu=nu,
            mu=mu,
            sigma=sigma,
            observed=y,
        )

    log.info("  Model built successfully.")
    return model


def sample_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
) -> az.InferenceData:
    """
    Sample from the posterior using NUTS.
    """
    log.info("Sampling posterior...")
    log.info(f"  draws={draws}  tune={tune}  chains={chains}  "
             f"target_accept={target_accept}")

    with model:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=True,
                return_inferencedata=True,
            )

    log.info("  Sampling complete.")
    return trace


def check_convergence(trace: az.InferenceData) -> bool:
    """
    Check R-hat and ESS for all parameters.
    """
    log.info("Checking convergence diagnostics...")

    summary = az.summary(trace, var_names=[
        "mu_global", "sigma_category", "alpha_category",
        "beta_sp500", "beta_vix", "nu", "sigma"
    ])

    max_rhat = summary["r_hat"].max()
    min_ess  = summary["ess_bulk"].min()

    log.info(f"  Max R-hat : {max_rhat:.4f}  (threshold: 1.01)")
    log.info(f"  Min ESS   : {min_ess:.0f}    (threshold: 400)")

    bad_rhat = summary[summary["r_hat"] > 1.01]
    bad_ess  = summary[summary["ess_bulk"] < 400]

    if len(bad_rhat) > 0:
        log.warning(f"  {len(bad_rhat)} parameters with R-hat > 1.01:")
        for param in bad_rhat.index:
            log.warning(f"    {param}  R-hat={bad_rhat.loc[param, 'r_hat']:.4f}")

    if len(bad_ess) > 0:
        log.warning(f"  {len(bad_ess)} parameters with ESS < 400:")
        for param in bad_ess.index:
            log.warning(f"    {param}  ESS={bad_ess.loc[param, 'ess_bulk']:.0f}")

    converged = (max_rhat <= 1.01) and (min_ess >= 400)
    log.info(f"  Convergence: {'PASSED' if converged else 'FAILED'}")
    return converged


def extract_results(trace: az.InferenceData) -> pd.DataFrame:
    """
    Extract posterior means and 94% HDI for all key parameters.
    """
    log.info("Extracting posterior summaries...")

    records = []

    s   = az.summary(trace, var_names=["mu_global"])
    row = s.iloc[0]
    records.append({
        "parameter": "mu_global", "type": "global",
        "category": None,
        "mean": row["mean"], "sd": row["sd"],
        "hdi_3%": row["hdi_3%"], "hdi_97%": row["hdi_97%"],
        "r_hat": row["r_hat"],
    })

    alpha_summary = az.summary(trace, var_names=["alpha_category"])
    for i, cat in enumerate(CATEGORY_ORDER):
        row = alpha_summary.iloc[i]
        records.append({
            "parameter": f"alpha_category[{cat}]", "type": "category_effect",
            "category": cat,
            "mean": row["mean"], "sd": row["sd"],
            "hdi_3%": row["hdi_3%"], "hdi_97%": row["hdi_97%"],
            "r_hat": row["r_hat"],
        })

    for var in ["beta_sp500", "beta_vix"]:
        s   = az.summary(trace, var_names=[var])
        row = s.iloc[0]
        records.append({
            "parameter": var, "type": "macro_effect",
            "category": None,
            "mean": row["mean"], "sd": row["sd"],
            "hdi_3%": row["hdi_3%"], "hdi_97%": row["hdi_97%"],
            "r_hat": row["r_hat"],
        })

    for var in ["nu", "sigma"]:
        s   = az.summary(trace, var_names=[var])
        row = s.iloc[0]
        records.append({
            "parameter": var, "type": "global",
            "category": None,
            "mean": row["mean"], "sd": row["sd"],
            "hdi_3%": row["hdi_3%"], "hdi_97%": row["hdi_97%"],
            "r_hat": row["r_hat"],
        })

    results_df = pd.DataFrame(records)

    log.info("  Global parameters:")
    for var in ["mu_global", "nu", "sigma"]:
        row = results_df[results_df["parameter"] == var].iloc[0]
        log.info(f"    {var:<20}  mean={row['mean']:.4f}  "
                 f"94% HDI=[{row['hdi_3%']:.4f}, {row['hdi_97%']:.4f}]")

    log.info("  Macro effects — core Finding 1:")
    for var in ["beta_sp500", "beta_vix"]:
        row = results_df[results_df["parameter"] == var].iloc[0]
        excludes_zero = (row["hdi_3%"] > 0) or (row["hdi_97%"] < 0)
        log.info(f"    {var:<15}  mean={row['mean']:.4f}  "
                 f"94% HDI=[{row['hdi_3%']:.4f}, {row['hdi_97%']:.4f}]  "
                 f"{'** excludes zero **' if excludes_zero else 'includes zero'}")

    log.info("  Category effects (alpha_category):")
    for cat in CATEGORY_ORDER:
        row = results_df[results_df["parameter"] == f"alpha_category[{cat}]"].iloc[0]
        excludes_zero = (row["hdi_3%"] > 0) or (row["hdi_97%"] < 0)
        log.info(f"    {cat:<30}  mean={row['mean']:.4f}  "
                 f"94% HDI=[{row['hdi_3%']:.4f}, {row['hdi_97%']:.4f}]  "
                 f"{'** excludes zero **' if excludes_zero else 'includes zero'}")

    return results_df


def main() -> None:
    log.info("=" * 60)
    log.info(f"models.py started at {datetime.now().isoformat()}")
    log.info("=" * 60)

    df    = load_and_prepare(PANEL_PATH, FEATURES_PATH)
    model = build_model(df)
    trace = sample_model(model)

    converged = check_convergence(trace)
    if not converged:
        log.warning("Convergence issues — check trace plots before interpreting.")

    results_df = extract_results(trace)

    trace.to_netcdf(str(TRACE_PATH))
    log.info(f"Written: {TRACE_PATH}")

    results_df.to_csv(SUMMARY_PATH, index=False)
    log.info(f"Written: {SUMMARY_PATH}")

    log.info("=" * 60)
    log.info(f"models.py complete at {datetime.now().isoformat()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()