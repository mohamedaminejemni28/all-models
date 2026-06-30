# XGBoost MLflow Results Platform

This Streamlit platform visualizes the XGBoost results already tracked in MLflow for the gait analysis project.

It connects to the local MLflow tracking store, reads experiments, runs, parameters, metrics, Excel artifacts, and SHAP plot images, then provides a clean interface for filtering and comparing XGBoost runs.

## Features

- Connects to the existing local `mlflow.db` by default.
- Reads active MLflow experiments and runs.
- Filters XGBoost runs by dataset, run name, number of features, CV accuracy, test accuracy, sensitivity, and specificity.
- Shows run metrics in tables and charts.
- Shows Top 3 XGBoost models for each dataset from `Scores_XGBoost_Top3.xlsx` artifacts.
- Compares multiple XGBoost runs visually.
- Displays XGBoost SHAP summary images directly in the app.
- Lists MLflow artifacts and previews Excel outputs.

## Folder Structure

```text
xgboost_mlflow_platform/
  app/
    components/
      charts.py
      sidebar.py
    services/
      artifact_service.py
      mlflow_service.py
    utils/
      dataframe.py
    config.py
    main.py
  README.md
  requirements.txt
```

## Installation on Windows PowerShell

From the repository root:

```powershell
cd C:\Users\k6qqj\Desktop\Gait-ML-Shap-main_y_Mohamed\Gait-ML-Shap-main
```

Create and activate a virtual environment if needed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dashboard requirements:

```powershell
pip install -r .\xgboost_mlflow_platform\requirements.txt
```

Run the Streamlit app:

```powershell
python -m streamlit run .\xgboost_mlflow_platform\app\main.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

## MLflow Tracking URI

By default, the app uses:

```text
sqlite:///C:/Users/k6qqj/Desktop/Gait-ML-Shap-main_y_Mohamed/Gait-ML-Shap-main/mlflow.db
```

You can override it in either of two ways:

1. Edit the Tracking URI field in the app sidebar.
2. Set the environment variable before launching:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///C:/path/to/mlflow.db"
python -m streamlit run .\xgboost_mlflow_platform\app\main.py
```

## Notes for Deployment Later

For team access, deploy the same Streamlit app to an internal server and point `MLFLOW_TRACKING_URI` to a shared MLflow backend store. The artifact location must also be reachable by the deployed machine, especially for SHAP images and Excel artifact previews.
