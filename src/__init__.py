"""
Analyst Forecast Bias research package.

The project is organized as standalone pipeline modules. Run the
pipeline stages from the repository root using module execution:

    python -m src.ingest
    python -m src.clean
    python -m src.macro
    python -m src.features
    python -m src.models
    python -m src.signals

The HMM regime analysis is exploratory:

    python -m src.regime

Individual functions should be imported from their respective modules,
rather than re-exported at the package level. This keeps the package
loosely coupled and prevents internal API changes in one pipeline stage
from breaking imports of another stage.
"""