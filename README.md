<div align="center">

# Analyst Forecast Bias

**Structure, Regime Sensitivity, and Predictive Signals**

*Souradeep Roy*

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyMC](https://img.shields.io/badge/PyMC-5.28-FF6B35?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-189AB4)
![ArViZ](https://img.shields.io/badge/ArViZ-0.23-5C6BC0)
![hmmlearn](https://img.shields.io/badge/hmmlearn-HMM-43A047)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

Every quarter, Wall Street analysts publish their earnings estimates for publicly traded companies. These estimates get aggregated into a consensus figure that the market prices in ahead of the actual report. When a company reports, the difference between what it actually earned and what analysts expected is the forecast error - and the direction of that error is whether the company beat or missed.

The well-documented pattern is that companies beat more often than they miss. Across this dataset of 72 large-cap US companies over nine years, the overall beat rate is around 82%. That alone is not surprising; it is a known feature of how the analyst ecosystem works, driven partly by management guidance, partly by how estimates get revised as the quarter progresses, and partly by the structural incentives analysts face. What is less well understood is how that bias is distributed: which types of companies carry most of it, whether market conditions are associated with changes in its magnitude, and whether the footprint left by the revision process itself contains any signal about what is coming.

Those three questions are what this project is built around.

> [!IMPORTANT]
> **Data Limitation Note**: The source Dolt dataset does not provide actual earnings announcement dates. Therefore, this study uses the fiscal period end as the canonical information cutoff (`prediction_cutoff = period_end_date`) and does not claim to model an actual announcement-time prediction. All predictive features and point-in-time macro variables are constructed strictly using observations available prior to the fiscal period end date.

<p align="center">
  <img src="reports/figures/02_beat_rate_by_category.png" width="760" alt="Analyst beat rate by earnings-driver category (2017-2026)">
</p>

The 72 companies are split into six earnings-driver categories, not by sector, but by the primary mechanism that drives their earnings. Tech cycle companies (NVDA, ASML, AMD) are driven by product and semiconductor cycles. Macro rate sensitive (JPM, GS, BAC) move with the rate environment. Commodity driven (XOM, CVX, COP) follow oil and metals prices. Economically cyclical (CAT, AMZN, HD) track the broader economic cycle. Regulatory idiosyncratic (LLY, PFE, UNH) turn on FDA approvals and policy decisions. Defensive baseline (PG, KO, WMT) is the defensive reference benchmark - stable, predictable businesses with deep analyst coverage, chosen specifically because they should show the least distortion if the patterns found elsewhere are real.

---

## What This Project Studies

**Does the bias vary by what type of company it is?**

This is the structural question. If all categories showed identical bias, the phenomenon would be uniform and category membership would tell you nothing. A hierarchical Bayesian model (PyMC) with nested ticker-level intercepts nested under category-level intercepts is used to estimate these category effects with proper uncertainty quantification, accounting for repeated company observations and common time/macro contexts.

**Does the macro environment change how large the error is?**

This is the regime question. The intuition is that uncertain markets should make forecasting harder. A three-state Hidden Markov Model is fitted to monthly S&P 500 returns to label market regimes for exploratory analysis, while the Bayesian panel model tests the relationship between forecast error and continuous macro indicators.

**Can you predict a beat or miss before the cutoff?**

Analysts revise their estimates continuously in the weeks before a period ends. The direction and shape of those revisions - whether consensus is drifting up or down, whether it is accelerating, whether analysts are disagreeing more or less - might contain information about whether the company will beat. Seven features are engineered from weekly consensus snapshots and fed into an XGBoost classifier using a walk-forward validation framework to test whether that signal is real and, if so, for which types of companies it holds up.

<p align="center">
  <img src="reports/figures/02_consensus_drift.png" width="760" alt="Consensus drift in the 16 weeks before cutoff by category">
</p>

The chart above shows how consensus estimates move in the 16 weeks before each cutoff, by category. Tech cycle companies (green) drift upward as the period closes. Economically cyclical (pink) drift downward. The shape of this drift is the raw input to the revision path features.

---

## What I Found

**The macro environment is associated with forecast error through uncertainty.** In the Bayesian panel model, VIX is associated with a positive effect on forecast error (analysts tend to miss more when uncertainty is high), whereas the quarterly S&P 500 return has no credible effect.

**Bias is structural and category-dependent.** The hierarchical Bayesian panel model confirms that category-level differences remain distinct even after controlling for individual ticker-level random effects and macro conditions. The defensive baseline category exhibits the lowest structural bias, validating its role as a reference benchmark.

**The revision path contains modest predictive signals.** Evaluated under expanding-window walk-forward validation (training on 2017-2019, testing on 2020, and so on up to 2025), the XGBoost classifier trained on pre-cutoff revision path features and point-in-time macro variables achieves a modest out-of-sample ranking signal. 

### Walk-Forward Predictive Performance

The table below shows the performance of the full Category + Revision + Macro model across the walk-forward validation folds:

| Fold | Train Period | Validation | Test Year | N | ROC-AUC | PR-AUC | Baseline PR-AUC |
|---|---|---|---|---|---|---|---|
| 1 | 2017–2018 | 2019 | 2020 | 179 | 0.5421 | 0.8524 | 0.8143 |
| 2 | 2017–2019 | 2020 | 2021 | 184 | 0.5367 | 0.8611 | 0.8260 |
| 3 | 2017–2020 | 2021 | 2022 | 192 | 0.5184 | 0.8402 | 0.8021 |
| 4 | 2017–2021 | 2022 | 2023 | 201 | 0.5510 | 0.8732 | 0.8358 |
| 5 | 2017–2022 | 2023 | 2024 | 210 | 0.5392 | 0.8654 | 0.8286 |
| 6 | 2017–2023 | 2024 | 2025 | 215 | 0.5284 | 0.8590 | 0.8186 |

* **Mean ROC-AUC**: 0.5360 (Std: 0.0109)
* **Mean PR-AUC**: 0.8586 (Prevalence Baseline: 0.8209)

The predictive signal is modest, reflecting the difficulty of predicting earnings surprises out-of-sample under strict leakage-free conditions.

---

## Model Ablation Analysis

Ablation experiments run across identical walk-forward folds reveal where the predictive information originates:

| Model | Mean ROC-AUC | Mean PR-AUC | Std ROC-AUC |
|---|---|---|---|
| Majority baseline | 0.5000 | 0.8209 | 0.0000 |
| Category only | 0.5204 | 0.8324 | 0.0084 |
| Revision only | 0.5112 | 0.8260 | 0.0121 |
| Macro only | 0.5054 | 0.8221 | 0.0062 |
| Category + Revision | 0.5312 | 0.8510 | 0.0098 |
| Revision + Macro | 0.5168 | 0.8304 | 0.0114 |
| Category + Macro | 0.5242 | 0.8386 | 0.0076 |
| **Category + Revision + Macro (Full)** | **0.5360** | **0.8586** | **0.0109** |

The ablation analysis shows that category metadata and revision history carry the majority of the predictive signal, with macro context providing a minor incremental benefit.

---

## How It's Built

The pipeline runs in seven steps, each a standalone Python module:

| Module | What it does |
|---|---|
| [`ingest.py`](src/ingest.py) | Filters raw Dolt table exports to 72 tickers and the study date range |
| [`clean.py`](src/clean.py) | Computes raw and normalized forecast errors, winsorizes by category, sets prediction cutoff |
| [`macro.py`](src/macro.py) | Downloads macro indicators, prepares explanatory data and predictive PIT macro variables |
| [`regime.py`](src/regime.py) | (Exploratory) Fits HMM to monthly S&P 500 returns to label market regimes |
| [`features.py`](src/features.py) | Builds point-in-time revision path features from weekly consensus snapshots |
| [`models.py`](src/models.py) | Fits the hierarchical Bayesian Student-t panel model in PyMC for raw and normalized targets |
| [`signals.py`](src/signals.py) | Executes walk-forward XGBoost training, temporal validation, and model ablations |

> [!NOTE]
> The HMM (`regime.py`) is not a dependency of the primary Bayesian or XGBoost models. It is used only for exploratory regime analysis.

---

## Setup

```bash
git clone https://github.com/souro26/analyst-forecast-bias.git
cd analyst-forecast-bias

conda env create -f environment.yml
conda activate forecast-bias

# Pull data from DoltHub
cd data/external/earnings
dolt clone dolthub/earnings .
dolt table export eps_history   ../../../data/raw/eps_history_full.csv
dolt table export eps_estimate  ../../../data/raw/eps_estimate_full.csv
cd ../../..

# Optional - only needed if HMM fails to converge
echo "FRED_API_KEY=your_key" > .env

# Option A: DVC (tracks which stages need re-running based on file hashes)
dvc init   # first time only - creates .dvc/ directory
dvc repro

# Option B: make (Unix / macOS / WSL on Windows)
make

# Option C: run each step manually in order
python src/ingest.py
python src/clean.py
python src/macro.py
python src/regime.py
python src/features.py
python src/models.py
python src/signals.py
```

`dvc repro` uses [`dvc.yaml`](dvc.yaml) to re-run only stages whose inputs have changed since the last run. `make` does the same via stamp files (`.stamps/`). Both require the Dolt data pull above as a prerequisite. On Windows, GNU make is available via Git Bash, WSL, or `choco install make`.

Logs for each step are written to `logs/`. All model outputs go to `models/`, all figures to `reports/figures/`.

---

## License

MIT
