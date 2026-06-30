# CNN Pipeline for Gait Classification

This pipeline adds a PyTorch 1D CNN baseline for the existing Autism 2024, Young vs Older Adults 2024, and Flatfoot workflows.

The current project stores processed gait data as tabular Excel feature sheets. For that reason, the implemented CNN is a 1D CNN over ordered biomechanical features. Each participant/trial is reshaped from:

```text
(n_features)
```

to:

```text
(channels=1, n_features)
```

This is more suitable than a 2D CNN unless the project later adds true gait-cycle matrices or raw time-series signals. If raw time-series data becomes available, the same PyTorch structure can be extended so channels are joints/axes and the sequence dimension is gait-cycle time.

## Outputs

For each dataset, the pipeline writes:

- `results/<dataset>_Scores_CNN.xlsx`
- `results/<dataset>_Scores_CNN_Top3.xlsx`
- `output-*/<dataset>_cnn_<sheet>/best_cnn_model.pt`
- `output-*/<dataset>_cnn_<sheet>/best_cnn_training_curve.png`
- `output-*/<dataset>_cnn_<sheet>/best_cnn_permutation_importance.xlsx`
- MLflow metrics, parameters, model files, plots, and Excel artifacts

## Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run one dataset:

```powershell
python pipelines_cnn\run_cnn_pipeline.py --dataset autism_2024
python pipelines_cnn\run_cnn_pipeline.py --dataset young_old_2024
python pipelines_cnn\run_cnn_pipeline.py --dataset flatfoot_control_older_2024
```

Run all datasets:

```powershell
python pipelines_cnn\run_cnn_pipeline.py --dataset all
```

Create the comparison table:

```powershell
python pipelines_cnn\compare_models.py
```

If SVM and Random Forest score files remain in sibling project folders, compare across all three locations:

```powershell
python pipelines_cnn\compare_models.py --results-dirs "results" "..\..\Gait-ML-Shap-main_y_Mohamed -Random_Forest\Gait-ML-Shap-main\results" "..\..\TASK_SVM\Gait-ML-Shap-main\Gait-ML-Shap-main"
```

The comparison table is saved to:

```text
results/model_family_comparison.xlsx
```

## MLflow

Start the local MLflow UI from the project root:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```
