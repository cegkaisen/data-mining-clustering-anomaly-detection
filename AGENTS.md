# Repository Instructions

## Assignment Constraints

- Use at most 10 clusters for the clustering task.
- Detect exactly 50 anomalous documents for the anomaly detection task.
- Do not modify raw input data files in `data/`.

## Required Outputs

- `data/clusters.csv` or the required submission path must contain document-to-cluster assignments.
- `clusters.csv` must preserve the exact original row order of `data/articles.csv`.
- `data/anomalies.csv` or the required submission path must contain exactly 50 anomalous document IDs.
- Preserve the expected CSV format and column names used by the assignment files.
- Do not add extra unnamed index columns when saving CSV files.

## Project Structure

- Use `notebooks/` for exploration, EDA, visualizations, and experiment comparison.
- Use `src/` for reusable Python code such as preprocessing, clustering, anomaly detection, and utilities.
- Use `outputs/` for generated figures, intermediate experiment outputs, and diagnostics.
- Use `report/` for final write-up materials.
- The structure may be modified if it improves clarity, but keep the notebook/module split.

## Coding Style

- Write simple, modular, readable Python.
- Prefer clear functions with docstrings over long notebook-only code blocks.
- Avoid hardcoded paths inside reusable `src/` functions; pass paths from callers.
- Keep comments useful and brief.

## Workflow Rules

- Keep each change focused on the requested task.
- Do not implement multiple major tasks at once.
- Do not introduce clustering or anomaly detection changes while working only on EDA, and vice versa.
- Clustering experiments should support quantitative evaluation, such as silhouette score, and qualitative inspection, such as top terms and representative documents.
- Preserve existing user work and avoid unrelated refactors.
- Validate outputs before considering a task complete.
