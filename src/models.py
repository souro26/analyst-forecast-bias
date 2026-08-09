"""
models.py
---------
Hierarchical Bayesian model of analyst forecast bias.

Structure:
    Level 1 - grand mean (mu_global)
    Level 2 - category effects (alpha_category)
    Level 3 - year effects grouped by regime (mu_regime, year_effect)
    Likelihood - Student-t (justified by EDA QQ plots)

The key insight: regime signal operates at the year level, not the
observation level. Adding a year random effect with regime-level
hyperpriors lets the model find regime structure without being
drowned by within-quarter observation noise.

Model:
    mu_global         ~ Normal(0, 0.5)
    sigma_category    ~ HalfNormal(0.3)
    alpha_category[c] ~ Normal(0, sigma_category)

    mu_regime[r]      ~ Normal(0, 0.3)       # mean effect per regime
    sigma_year        ~ HalfNormal(0.2)       # year-to-year noise within regime
    year_effect[y]    ~ Normal(mu_regime[regime_of_year[y]], sigma_year)

    mu[i] = mu_global
           + alpha_category[cat_idx[i]]
           + year_effect[year_idx[i]]

    nu    ~ Gamma(2, 0.1)
    sigma ~ HalfNormal(0.3)
    y[i]  ~ StudentT(nu, mu[i], sigma)

Inputs:
    - data/processed/panel.parquet
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

REGIME_ORDER = ["bull", "sideways", "bear"]

YEAR_REGIME_MAP = {
    2017: "sideways",
    2018: "bear",
    2019: "bull",
    2020: "bear",
    2021: "sideways",
    2022: "bear",
    2023: "bull",
    2024: "bull",
    2025: "bull",
}


def load_and_prepare(panel_path: Path, features_path: Path) -> pd.DataFrame:
    """
    Load panel and features, merge, encode categoricals as integers.
    Excludes 2026 (partial year). Drops rows missing regime or target.
    """
    log.info("Loading data...")
    panel    = pd.read_parquet(panel_path)
    features = pd.read_parquet(features_path)

    df = panel.merge(features, on=["act_symbol", "period_end_date"], how="left")
    log.info(f"  Merged shape: {df.shape}")

    df = df[df["year"] < 2026].copy()
    log.info(f"  After excluding 2026: {len(df):,} rows")

    before = len(df)
    df = df.dropna(subset=["regime", "forecast_error_winsorized"])
    after = len(df)
    if before - after > 0:
        log.warning(f"  Dropped {before - after} rows with missing regime or target.")

    df["category_idx"] = pd.Categorical(
        df["category"], categories=CATEGORY_ORDER
    ).codes

    year_order = sorted(df["year"].unique().tolist())
    df["year_idx"] = df["year"].map({y: i for i, y in enumerate(year_order)})

    df["regime_of_year"] = df["year"].map(YEAR_REGIME_MAP)
    df["regime_idx"] = pd.Categorical(
        df["regime_of_year"], categories=REGIME_ORDER
    ).codes

    n_years = len(year_order)
    year_to_regime_idx = np.array([
        REGIME_ORDER.index(YEAR_REGIME_MAP[y]) for y in year_order
    ])

    log.info(f"  Final modelling rows: {len(df):,}")
    log.info("  Category index mapping:")
    for i, c in enumerate(CATEGORY_ORDER):
        n = (df["category_idx"] == i).sum()
        log.info(f"    {i}  {c:<30}  n={n}")

    log.info("  Year index mapping:")
    for i, y in enumerate(year_order):
        regime = YEAR_REGIME_MAP[y]
        n = (df["year_idx"] == i).sum()
        log.info(f"    {i}  {y}  regime={regime:<10}  n={n}")

    df.attrs["year_order"]         = year_order
    df.attrs["year_to_regime_idx"] = year_to_regime_idx.tolist()
    df.attrs["n_years"]            = n_years

    return df


def build_model(df: pd.DataFrame) -> pm.Model:
    """
    Build the hierarchical Bayesian model with year-level regime effects.
    """
    n_categories       = len(CATEGORY_ORDER)
    n_regimes          = len(REGIME_ORDER)
    n_years            = df.attrs["n_years"]
    year_to_regime_idx = np.array(df.attrs["year_to_regime_idx"])

    cat_idx  = df["category_idx"].values
    year_idx = df["year_idx"].values
    y        = df["forecast_error_winsorized"].values.astype(float)

    log.info("Building hierarchical model...")
    log.info(f"  n_categories={n_categories}  n_regimes={n_regimes}  "
             f"n_years={n_years}  n_obs={len(y)}")

    with pm.Model() as model:

        mu_global      = pm.Normal("mu_global", mu=0.0, sigma=0.5)
        sigma_category = pm.HalfNormal("sigma_category", sigma=0.3)

        alpha_category = pm.Normal(
            "alpha_category",
            mu=0.0,
            sigma=sigma_category,
            shape=n_categories,
        )

        mu_regime  = pm.Normal("mu_regime", mu=0.0, sigma=0.3, shape=n_regimes)
        sigma_year = pm.HalfNormal("sigma_year", sigma=0.2)

        year_effect_raw = pm.Normal(
            "year_effect_raw",
            mu=0.0,
            sigma=1.0,
            shape=n_years,
        )
        year_effect = pm.Deterministic(
            "year_effect",
            mu_regime[year_to_regime_idx] + year_effect_raw * sigma_year,
        )

        mu = (
            mu_global
            + alpha_category[cat_idx]
            + year_effect[year_idx]
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
    log.info(f"  draws={draws}  tune={tune}  chains={chains}  target_accept={target_accept}")

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
        "mu_regime", "sigma_year", "year_effect",
        "nu", "sigma"
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


def extract_results(trace: az.InferenceData, df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract posterior means and 94% HDI for key parameters.
    """
    log.info("Extracting posterior summaries...")

    year_order = df.attrs["year_order"]
    records    = []

    s   = az.summary(trace, var_names=["mu_global"])
    row = s.iloc[0]
    records.append({
        "parameter": "mu_global", "type": "global",
        "category": None, "year": None, "regime": None,
        "mean": row["mean"], "sd": row["sd"],
        "hdi_3%": row["hdi_3%"], "hdi_97%": row["hdi_97%"],
        "r_hat": row["r_hat"],
    })

    alpha_summary = az.summary(trace, var_names=["alpha_category"])
    for i, cat in enumerate(CATEGORY_ORDER):
        row = alpha_summary.iloc[i]
        records.append({
            "parameter": f"alpha_category[{cat}]", "type": "category_effect",
            "category": cat, "year": None, "regime": None,
            "mean": row["mean"], "sd": row["sd"],
            "hdi_3%": row["hdi_3%"], "hdi_97%": row["hdi_97%"],
            "r_hat": row["r_hat"],
        })

    regime_summary = az.summary(trace, var_names=["mu_regime"])
    for i, reg in enumerate(REGIME_ORDER):
        row = regime_summary.iloc[i]
        records.append({
            "parameter": f"mu_regime[{reg}]", "type": "regime_mean",
            "category": None, "year": None, "regime": reg,
            "mean": row["mean"], "sd": row["sd"],
            "hdi_3%": row["hdi_3%"], "hdi_97%": row["hdi_97%"],
            "r_hat": row["r_hat"],
        })

    year_summary = az.summary(trace, var_names=["year_effect"])
    for i, yr in enumerate(year_order):
        row = year_summary.iloc[i]
        records.append({
            "parameter": f"year_effect[{yr}]", "type": "year_effect",
            "category": None, "year": yr,
            "regime": YEAR_REGIME_MAP[yr],
            "mean": row["mean"], "sd": row["sd"],
            "hdi_3%": row["hdi_3%"], "hdi_97%": row["hdi_97%"],
            "r_hat": row["r_hat"],
        })

    for var in ["nu", "sigma"]:
        s   = az.summary(trace, var_names=[var])
        row = s.iloc[0]
        records.append({
            "parameter": var, "type": "global",
            "category": None, "year": None, "regime": None,
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

    log.info("  Regime means (mu_regime) — core Finding 1:")
    for reg in REGIME_ORDER:
        row = results_df[results_df["parameter"] == f"mu_regime[{reg}]"].iloc[0]
        log.info(f"    {reg:<10}  mean={row['mean']:.4f}  "
                 f"94% HDI=[{row['hdi_3%']:.4f}, {row['hdi_97%']:.4f}]")

    log.info("  Category effects (alpha_category):")
    for cat in CATEGORY_ORDER:
        row = results_df[results_df["parameter"] == f"alpha_category[{cat}]"].iloc[0]
        log.info(f"    {cat:<30}  mean={row['mean']:.4f}  "
                 f"94% HDI=[{row['hdi_3%']:.4f}, {row['hdi_97%']:.4f}]")

    log.info("  Year effects:")
    for yr in year_order:
        row = results_df[results_df["parameter"] == f"year_effect[{yr}]"].iloc[0]
        log.info(f"    {yr}  regime={YEAR_REGIME_MAP[yr]:<10}  "
                 f"mean={row['mean']:.4f}  "
                 f"94% HDI=[{row['hdi_3%']:.4f}, {row['hdi_97%']:.4f}]")

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

    results_df = extract_results(trace, df)

    trace.to_netcdf(str(TRACE_PATH))
    log.info(f"Written: {TRACE_PATH}")

    results_df.to_csv(SUMMARY_PATH, index=False)
    log.info(f"Written: {SUMMARY_PATH}")

    log.info("=" * 60)
    log.info(f"models.py complete at {datetime.now().isoformat()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()