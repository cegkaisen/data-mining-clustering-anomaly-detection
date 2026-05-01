# Data Mining Clustering Assignment

Clean Python project structure for a data mining assignment focused on text
clustering and anomaly detection.

## Project Structure

```text
data/          Input datasets. Do not modify raw data files.
notebooks/     Interactive EDA, visualization, and experiment notebooks.
src/           Reusable Python modules with preprocessing and modeling helpers.
outputs/       Generated experiment outputs, figures, and model artifacts.
report/        Final report materials.
```

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name data-mining-clustering-assignment
```

## Workflow

Use `notebooks/analysis.ipynb` for exploratory data analysis, visualizations,
and trying different preprocessing, clustering, and anomaly detection methods.
Move reusable logic into `src/` modules so experiments stay clean and repeatable.

The Python modules avoid hardcoded paths. Pass input and output paths from the
notebook or another calling script.
