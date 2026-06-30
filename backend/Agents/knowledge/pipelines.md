# Pipelines

The project pipeline has phases.

Phase 1 reviews and prepares gait feature files.
Phase 2 removes highly correlated features.
Phase 3 creates train and test splits.
Phase 4.1 selects feature subsets.
Phase 4.2 trains model combinations and performs cross-validation.
Phase 4.3 ranks scores and saves Top 3 outputs.
Phase 4.4 produces SHAP or permutation-importance style feature explanations.

On Kaggle, copy the dataset from /kaggle/input to /kaggle/working before running scripts, because input datasets are read-only.