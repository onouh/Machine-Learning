# City Hotel Booking Machine Learning

A reproducible Python project for the City Hotel H2 booking dataset. The planned system includes ADR regression, cancellation classification, booking-level clustering, PCA/LDA comparisons, hyperparameter tuning, saved inference pipelines, a CLI, and a Streamlit interface. The current repository is a scaffold: task models and training workflows are not implemented yet. Track progress in [TASK.md](TASK.md) and durable decisions in [MEMORY.md](MEMORY.md).

## Requirements

- Python 3.10 or newer.
- The assignment's City Hotel `H2.csv` dataset, placed locally at `data/raw/H2.csv`.
- Git for the required repository history and collaboration workflow.

## Setup

Create and activate a virtual environment, then install the project and development checks:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
uv pip install --python .venv/bin/python -r requirements.txt

# Alternatively, with an activated Python 3.10+ environment:
# python -m pip install -r requirements.txt
```

See [data/README.md](data/README.md) for dataset provenance and placement. Raw data, processed data, generated figures/results, and fitted model artifacts are ignored by Git by default.

## Current commands

Validate the expected H2 columns and cancellation labels:

```bash
python main.py validate
# or choose another local path:
python main.py validate --data /path/to/H2.csv
```

Print a compact data profile:

```bash
python main.py analyze
```

Start the Streamlit analysis shell:

```bash
streamlit run app.py
```

Both interfaces currently provide only loading, schema validation, and basic profiling. Model training, tuning, comparison, prediction, PCA/LDA, and clustering UI/CLI commands are future work listed in [TASK.md](TASK.md).

## Project organization

- `src/hotel_booking/`: reusable data, task, evaluation, inference, and service modules.
- `main.py`: CLI entry point; `app.py`: Streamlit presentation layer.
- `notebooks/`: optional exploratory work; notebooks are not the main implementation.
- `docs/`: data dictionary, experiment protocol, leakage audit, and report material.
- `tests/`: lightweight, deterministic automated checks using synthetic data.
- `data/`, `artifacts/`, `outputs/`: local or generated files, ignored by default.

## Reproducibility and leakage

The final workflow must split data before fitting preprocessing, feature selection, dimensionality reduction, or tuning. The final supervised test set must not be repeatedly used for model selection. See the preliminary [leakage audit](docs/leakage_audit.md) for task-specific feature timing and leakage decisions. No result should be interpreted as a trained model until the corresponding task is implemented and evaluated.

## Development checks

```bash
pytest
ruff check src tests main.py app.py
```

GitHub Actions runs focused tests and lint on pull requests; it does not download the dataset or run expensive model searches.

## GitHub collaboration

All project work should be made in the team GitHub repository using each student's own account and meaningful incremental commits. Use short-lived branches and pull requests into `main`. The repository owner must configure GitHub branch protection separately to require pull requests, one approval, resolved review conversations, and the named passing CI checks. See [TASK.md](TASK.md) for the project checklist.

For each milestone submission, record the repository URL and submitted commit SHA here. The team and repository URL have not yet been supplied.

## Dataset citation

Nuno António, Ana de Almeida, and Luís Nunes. “Hotel booking demand datasets.” *Data in Brief*, Volume 22, 2019, pp. 41–49. <https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/>.
