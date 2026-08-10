"""
models.py
---------
Hierarchical Bayesian panel model of analyst forecast bias.
Acknowledges panel structure: repeated company observations and common macro conditions.

Model A:
    mu_global         ~ Normal(0, 0.5)
    sigma_category    ~ HalfNormal(0.3)
    alpha_category[c] ~ Normal(0, sigma_category)
    
    sigma_ticker      ~ HalfNormal(0.3)
    alpha_ticker[t]   ~ Normal(alpha_category[category[t]], sigma_ticker) [Non-centered implementation]

    beta_sp500        ~ Normal(0, 0.3)
    beta_vix          ~ Normal(0, 0.3)

    mu[i] = mu_global + alpha_ticker[ticker_idx[i]] + beta_sp500 * sp500_z[i] + beta_vix * vix_z[i]

    nu    ~ Gamma(2, 0.1)
    sigma ~ HalfNormal(0.3)
    y[i]  ~ StudentT(nu, mu[i], sigma)

Model B (Robustness Time-Effect Model):
    Same as Model A but replaces continuous macro variables with explicit quarter/time random intercepts:
    mu[i] = mu_global + alpha_ticker[ticker_idx[i]] + alpha_quarter[quarter_idx[i]]

Outputs:
    - models/trace.nc & models/summary.csv (Raw forecast error - Model A)
    - models/trace_normalized.nc & models/summary_normalized.csv (Scale-normalized error - Model A)
    - models/trace_time_effect.nc & models/summary_time_effect.csv (Time-effect model - Model B)
    - logs/models.log
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

PANEL_PATH    = PROCESSED / "panel_macro.parquet"
FEATURES_PATH = PROCESSED / "features.parquet"

TRACE_PATH    = MODELS_DIR / "trace.nc"
SUMMARY_PATH  = MODELS_DIR / "summary.csv"

TRACE_NORM_PATH   = MODELS_DIR / "trace_normalized.nc"
SUMMARY_NORM_PATH = MODELS_DIR / "summary_normalized.csv"

TRACE_TIME_PATH   = MODELS_DIR / "trace_time_effect.nc"
SUMMARY_TIME_PATH = MODELS_DIR / "summary_time_effect.csv"

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
    Excludes 2026 (partial year). Drops rows missing targets.
    """
    log.info("Loading data...")
    panel    = pd.read_parquet(panel_path)
    features = pd.read_parquet(features_path)

    df = panel.merge(features, on=["act_symbol", "period_end_date"], how="left")
    log.info(f"  Merged shape: {df.shape}")

    # Confirm macro columns present
    for col in ["sp500_return_z", "vix_mean_z"]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in panel. "
                "Run macro.py before models.py."
            )

    # Exclude partial year 2026
    df = df[df["year"] < 2026].copy()
    log.info(f"  After excluding 2026: {len(df):,} rows")

    # Encode category as integer index
    df["category_idx"] = pd.Categorical(
        df["category"], categories=CATEGORY_ORDER
    ).codes

    # Create mapping from ticker to index
    tickers_unique = df["act_symbol"].unique()
    ticker_to_idx = {t: i for i, t in enumerate(tickers_unique)}
    df["ticker_idx"] = df["act_symbol"].map(ticker_to_idx)

    # Create mapping from quarter key to index
    df["quarter_key"] = df["year"].astype(str) + "Q" + df["fiscal_quarter"].astype(str)
    quarters_unique = df["quarter_key"].unique()
    quarter_to_idx = {q: i for i, q in enumerate(quarters_unique)}
    df["quarter_idx"] = df["quarter_key"].map(quarter_to_idx)

    return df


def build_model(df: pd.DataFrame, target_col: str) -> pm.Model:
    """
    Build the hierarchical Bayesian panel model with nested ticker-level effects (Model A).
    Uses non-centered parameterization for stable convergence.
    """
    n_categories = len(CATEGORY_ORDER)
    n_tickers = len(df["ticker_idx"].unique())

    ticker_idx = df["ticker_idx"].values
    sp500_z    = df["sp500_return_z"].values.astype(float)
    vix_z      = df["vix_mean_z"].values.astype(float)
    
    # Target column check
    y = df[target_col].values.astype(float)
    valid_mask = ~np.isnan(y)
    y = y[valid_mask]
    ticker_idx = ticker_idx[valid_mask]
    sp500_z = sp500_z[valid_mask]
    vix_z = vix_z[valid_mask]

    # Map each unique ticker to its corresponding category index
    ticker_cat_idx = df.drop_duplicates("ticker_idx").sort_values("ticker_idx")["category_idx"].values

    log.info(f"Building hierarchical Model A (Macro) for target '{target_col}'...")
    log.info(f"  n_categories={n_categories}  n_tickers={n_tickers}  n_obs={len(y)}")

    with pm.Model() as model:
        # Global bias intercept
        mu_global      = pm.Normal("mu_global", mu=0.0, sigma=0.5)
        
        # Category-level hyperpriors
        sigma_category = pm.HalfNormal("sigma_category", sigma=0.3)
        alpha_category = pm.Normal(
            "alpha_category",
            mu=0.0,
            sigma=sigma_category,
            shape=n_categories,
        )

        # Ticker-level partial pooling (Non-centered parameterization)
        sigma_ticker   = pm.HalfNormal("sigma_ticker", sigma=0.3)
        alpha_ticker_offset = pm.Normal("alpha_ticker_offset", mu=0.0, sigma=1.0, shape=n_tickers)
        alpha_ticker   = pm.Deterministic(
            "alpha_ticker",
            alpha_category[ticker_cat_idx] + alpha_ticker_offset * sigma_ticker
        )

        # Continuous macro effects (explanatory)
        beta_sp500 = pm.Normal("beta_sp500", mu=0.0, sigma=0.3)
        beta_vix   = pm.Normal("beta_vix",   mu=0.0, sigma=0.3)

        # Linear predictor
        mu = (
            mu_global
            + alpha_ticker[ticker_idx]
            + beta_sp500 * sp500_z
            + beta_vix   * vix_z
        )

        # Student-t likelihood to handle heavy tails robustly
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


def build_time_effect_model(df: pd.DataFrame, target_col: str) -> pm.Model:
    """
    Build the hierarchical Bayesian robustness model with explicit quarter/time effects (Model B).
    Uses non-centered parameterization for stable convergence.
    """
    n_categories = len(CATEGORY_ORDER)
    n_tickers = len(df["ticker_idx"].unique())
    n_quarters = len(df["quarter_idx"].unique())

    ticker_idx = df["ticker_idx"].values
    quarter_idx = df["quarter_idx"].values
    
    y = df[target_col].values.astype(float)
    valid_mask = ~np.isnan(y)
    y = y[valid_mask]
    ticker_idx = ticker_idx[valid_mask]
    quarter_idx = quarter_idx[valid_mask]

    ticker_cat_idx = df.drop_duplicates("ticker_idx").sort_values("ticker_idx")["category_idx"].values

    log.info(f"Building hierarchical Model B (Time-Effect) for target '{target_col}'...")
    log.info(f"  n_categories={n_categories}  n_tickers={n_tickers}  n_quarters={n_quarters}  n_obs={len(y)}")

    with pm.Model() as model:
        # Global intercept
        mu_global      = pm.Normal("mu_global", mu=0.0, sigma=0.5)
        
        # Category-level hyperpriors
        sigma_category = pm.HalfNormal("sigma_category", sigma=0.3)
        alpha_category = pm.Normal(
            "alpha_category",
            mu=0.0,
            sigma=sigma_category,
            shape=n_categories,
        )

        # Ticker-level partial pooling (Non-centered parameterization)
        sigma_ticker   = pm.HalfNormal("sigma_ticker", sigma=0.3)
        alpha_ticker_offset = pm.Normal("alpha_ticker_offset", mu=0.0, sigma=1.0, shape=n_tickers)
        alpha_ticker   = pm.Deterministic(
            "alpha_ticker",
            alpha_category[ticker_cat_idx] + alpha_ticker_offset * sigma_ticker
        )

        # Quarter-level time effects (Non-centered parameterization)
        sigma_quarter = pm.HalfNormal("sigma_quarter", sigma=0.3)
        alpha_quarter_offset = pm.Normal("alpha_quarter_offset", mu=0.0, sigma=1.0, shape=n_quarters)
        alpha_quarter = pm.Deterministic(
            "alpha_quarter",
            alpha_quarter_offset * sigma_quarter
        )

        # Linear predictor (No macro variables)
        mu = (
            mu_global
            + alpha_ticker[ticker_idx]
            + alpha_quarter[quarter_idx]
        )

        # Student-t likelihood to handle heavy tails robustly
        nu    = pm.Gamma("nu", alpha=2, beta=0.1)
        sigma = pm.HalfNormal("sigma", sigma=0.3)

        pm.StudentT(
            "y",
            nu=nu,
            mu=mu,
            sigma=sigma,
            observed=y,
        )

    log.info("  Model B built successfully.")
    return model


def sample_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
) -> az.InferenceData:
    """Sample from the posterior using NUTS."""
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
    """Check convergence diagnostics: R-hat, ESS, and divergences."""
    log.info("Checking convergence diagnostics...")

    available_vars = list(trace.posterior.data_vars.keys())
    vars_to_check = ["mu_global", "sigma_category", "sigma_ticker", "nu", "sigma"]
    for v in ["sigma_quarter", "beta_sp500", "beta_vix"]:
        if v in available_vars:
            vars_to_check.append(v)

    # Calculate summary stats for main parameters
    summary = az.summary(trace, var_names=vars_to_check)

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

    # Check divergences
    divergences = int(trace.sample_stats["diverging"].sum())
    log.info(f"  Divergence count: {divergences}")

    converged = (max_rhat <= 1.01) and (min_ess >= 400) and (divergences == 0)
    log.info(f"  Convergence: {'PASSED' if converged else 'FAILED/WARNING'}")
    return converged


def extract_results(trace: az.InferenceData, target_name: str) -> pd.DataFrame:
    """Extract posterior statistics for key parameters."""
    log.info(f"Extracting posterior summaries for {target_name}...")

    records = []
    available_vars = list(trace.posterior.data_vars.keys())

    vars_to_extract = ["mu_global", "sigma_category", "sigma_ticker", "sigma_quarter", "beta_sp500", "beta_vix", "nu", "sigma"]
    for var in vars_to_extract:
        if var in available_vars:
            s = az.summary(trace, var_names=[var])
            row = s.iloc[0]
            records.append({
                "parameter": var,
                "type": "global_or_hyper",
                "category": None,
                "mean": row["mean"],
                "sd": row["sd"],
                "hdi_3%": row["hdi_3%"],
                "hdi_97%": row["hdi_97%"],
                "r_hat": row["r_hat"],
            })

    # alpha_category
    if "alpha_category" in available_vars:
        alpha_summary = az.summary(trace, var_names=["alpha_category"])
        for i, cat in enumerate(CATEGORY_ORDER):
            row = alpha_summary.iloc[i]
            records.append({
                "parameter": f"alpha_category[{cat}]",
                "type": "category_effect",
                "category": cat,
                "mean": row["mean"],
                "sd": row["sd"],
                "hdi_3%": row["hdi_3%"],
                "hdi_97%": row["hdi_97%"],
                "r_hat": row["r_hat"],
            })

    results_df = pd.DataFrame(records)

    log.info(f"  Key posterior estimates ({target_name}):")
    for _, row in results_df.iterrows():
        p_name = row["parameter"]
        log.info(f"    {p_name:<35} mean={row['mean']:.4f} sd={row['sd']:.4f} HDI=[{row['hdi_3%']:.4f}, {row['hdi_97%']:.4f}]")

    return results_df


def main() -> None:
    log.info("=" * 60)
    log.info(f"models.py started at {datetime.now().isoformat()}")
    log.info("=" * 60)

    df = load_and_prepare(PANEL_PATH, FEATURES_PATH)

    # 1. Main model: Raw forecast error (Model A)
    log.info("--- Model 1: Raw Forecast Error (Model A - Macro) ---")
    model_raw = build_model(df, "forecast_error_winsorized")
    trace_raw = sample_model(model_raw)
    check_convergence(trace_raw)
    results_raw = extract_results(trace_raw, "Raw Forecast Error")
    
    # Save Model 1 outputs
    trace_raw.to_netcdf(str(TRACE_PATH))
    log.info(f"Written trace: {TRACE_PATH}")
    results_raw.to_csv(SUMMARY_PATH, index=False)
    log.info(f"Written summary: {SUMMARY_PATH}")

    # 2. Robustness model: Normalized error (Model A)
    log.info("--- Model 2: Scale-Normalized Robustness Error (Model A - Macro) ---")
    model_norm = build_model(df, "normalized_error_winsorized")
    trace_norm = sample_model(model_norm)
    check_convergence(trace_norm)
    results_norm = extract_results(trace_norm, "Normalized Robustness Error")

    # Save Model 2 outputs
    trace_norm.to_netcdf(str(TRACE_NORM_PATH))
    log.info(f"Written trace: {TRACE_NORM_PATH}")
    results_norm.to_csv(SUMMARY_NORM_PATH, index=False)
    log.info(f"Written summary: {SUMMARY_NORM_PATH}")

    # 3. Robustness model: Time-Effect model (Model B - Quarter intercepts)
    log.info("--- Model 3: Time-Effect Robustness Model (Model B - Quarter intercepts) ---")
    model_time = build_time_effect_model(df, "forecast_error_winsorized")
    trace_time = sample_model(model_time)
    check_convergence(trace_time)
    results_time = extract_results(trace_time, "Time-Effect Robustness Model")

    # Save Model 3 outputs
    trace_time.to_netcdf(str(TRACE_TIME_PATH))
    log.info(f"Written trace: {TRACE_TIME_PATH}")
    results_time.to_csv(SUMMARY_TIME_PATH, index=False)
    log.info(f"Written summary: {SUMMARY_TIME_PATH}")

    log.info("=" * 60)
    log.info(f"models.py complete at {datetime.now().isoformat()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()