# Experiment protocol

This document records the minimum experimental safeguards for the eventual implementation. It does not claim that experiments have been run.

- Use the City Hotel H2 dataset only for required experiments.
- Use fixed random seeds where applicable and record feature set, preprocessing, model parameters, tuning space, metric, and training time.
- Split supervised data before fitting imputers, encoders, scalers, feature selection, PCA/LDA, or other data-dependent transformations. Use pipelines inside cross-validation.
- Reserve the final supervised test set for one final evaluation after model selection.
- Tune each selected supervised model using validation/CV; tune clustering only with unsupervised criteria, never `ADR` or `IsCanceled` labels.
- Compare baseline and tuned configurations for at least three models/algorithms per task.
- Keep preprocessing and fitted model together in reusable inference artifacts. Inference must load artifacts, not retrain at startup.
- Report original-unit ADR errors and use appropriate classification and clustering metrics.
- Record duplicate handling and ADR-zero sensitivity analyses; preserve the chosen rationale.

See [TASK.md](../TASK.md) for pending implementation and [leakage_audit.md](leakage_audit.md) for task-specific feature-timing risks.
