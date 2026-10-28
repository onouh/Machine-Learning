# Project Tasks

Status: repository scaffold, data profiling/validation, CLI, Streamlit analysis shell, and focused CI checks are implemented. Modeling, experiments, and prediction/application functionality remain future work.

## Milestone 1 — data preparation and baseline experiments

- [ ] Confirm team of 4–5 and add all contributors using their own GitHub accounts.
- [ ] Add the GitHub repository URL and submitted commit SHA to the README for each submission.
- [ ] Place the City Hotel `H2.csv` locally under `data/raw/`; verify its 79,330-row, 31-column schema.
- [ ] Complete EDA: types, missingness, duplicate patterns, cardinality, distributions, outliers, and plots.
- [ ] Resolve and document the ADR prediction point and zero-ADR sensitivity analysis.
- [ ] Document booking-creation cancellation prediction features and the task-specific leakage audit.
- [ ] Define reproducible train/validation/test or cross-validation strategy; keep the final supervised test set untouched during model selection.
- [ ] Implement shared, train-fitted preprocessing pipelines and justified feature engineering.
- [ ] Run initial untuned baselines: at least one regression, one classification, and one clustering algorithm.
- [ ] Produce a Milestone 1 progress report with initial results, contribution summary, and remaining-work plan.

## Milestone 2 — complete system

- [ ] Compare at least three regression models and three classification models, each baseline and tuned.
- [ ] Compare at least three clustering algorithms, each baseline and optimized using unsupervised criteria.
- [ ] Complete PCA explained-variance analysis and required supervised comparisons; compare original, PCA, and LDA representations for a classifier.
- [ ] Record search spaces, optimization metrics, best configurations, validation results, test results, and training times.
- [ ] Interpret clusters and produce required visualizations; sample hierarchical clustering/dendrogram work where justified by scale.
- [ ] Select final regression and classification models using documented criteria.
- [ ] Save reusable fitted preprocessing/model pipelines locally and verify inference loads them without retraining.
- [ ] Complete CLI commands for analysis, training, tuning, evaluation, clustering, and predictions.
- [ ] Complete Streamlit views for dataset analysis, model comparisons/tuning, ADR and cancellation prediction, and clustering.
- [ ] Complete report, figures/tables, leakage audit, exact run instructions, and milestone deliverables.
- [ ] Verify the end-to-end app and prepare local artifacts for live demonstration.
- [ ] Exclude trained model and fitted pipeline files from the submission ZIP; include regeneration instructions.

## Repository and process

- [ ] Protect GitHub `main`: require PRs, one approval, resolved conversations, and passing tests/lint checks.
- [ ] Use meaningful incremental commits and ensure each member has identifiable contributions.
- [ ] Keep raw data, generated outputs, and fitted artifacts out of Git by default; explicitly force-add only selected model artifacts if needed.
