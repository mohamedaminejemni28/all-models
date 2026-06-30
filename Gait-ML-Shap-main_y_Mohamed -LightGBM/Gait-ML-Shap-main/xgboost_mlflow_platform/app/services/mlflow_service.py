from __future__ import annotations

from dataclasses import dataclass

import mlflow
import pandas as pd
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from app.config import XGBOOST_TOKEN
from app.services.artifact_service import list_artifacts_recursive
from app.utils.dataframe import first_existing, to_float, to_int


@dataclass(frozen=True)
class MlflowBundle:
    client: MlflowClient
    experiments: pd.DataFrame
    runs: pd.DataFrame


def connect(tracking_uri: str) -> MlflowBundle:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    experiment_rows = [
        {
            "experiment_id": exp.experiment_id,
            "experiment_name": exp.name,
            "artifact_location": exp.artifact_location,
        }
        for exp in experiments
    ]
    experiments_df = pd.DataFrame(experiment_rows)

    if experiments_df.empty:
        return MlflowBundle(client=client, experiments=experiments_df, runs=pd.DataFrame())

    runs = mlflow.search_runs(
        experiment_ids=experiments_df["experiment_id"].astype(str).tolist(),
        run_view_type=ViewType.ACTIVE_ONLY,
        output_format="pandas",
    )

    if runs.empty:
        return MlflowBundle(client=client, experiments=experiments_df, runs=pd.DataFrame())

    runs = runs.merge(
        experiments_df[["experiment_id", "experiment_name"]],
        left_on="experiment_id",
        right_on="experiment_id",
        how="left",
    )
    runs = runs[runs.apply(_is_xgboost_run, axis=1)].copy()
    runs = _normalize_runs(runs)
    return MlflowBundle(client=client, experiments=experiments_df, runs=runs)


def _is_xgboost_run(row: pd.Series) -> bool:
    searchable = " ".join(str(value).lower() for value in row.fillna("").values)
    return XGBOOST_TOKEN in searchable


def _normalize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    normalized_rows = []

    for _, row in runs.iterrows():
        run_name = first_existing(
            row,
            ["tags.mlflow.runName", "run_name", "tags.run_name"],
            default=row.get("run_id", ""),
        )
        dataset = first_existing(
            row,
            ["params.dataset_name", "params.dataset", "experiment_name"],
            default=row.get("experiment_name", "Unknown"),
        )
        model_type = first_existing(
            row,
            ["params.model_type", "params.XGBoost_model_type", "params.xgboost_model_type"],
            default="XGBoost",
        )

        normalized_rows.append(
            {
                "run_id": row.get("run_id"),
                "run_name": run_name,
                "experiment_id": row.get("experiment_id"),
                "experiment_name": row.get("experiment_name"),
                "dataset": dataset,
                "status": row.get("status"),
                "start_time": row.get("start_time"),
                "artifact_uri": row.get("artifact_uri"),
                "model_type": model_type,
                "best_model_name": first_existing(row, ["params.xgboost_best_model_name"]),
                "num_features": to_int(first_existing(row, ["params.xgboost_best_model_num_features"])),
                "cv_accuracy": to_float(first_existing(row, ["metrics.xgboost_best_CV_accuracy"])),
                "test_accuracy": to_float(first_existing(row, ["metrics.xgboost_best_Test_Accuracy"])),
                "sensitivity": to_float(first_existing(row, ["metrics.xgboost_best_Sensitivity"])),
                "specificity": to_float(first_existing(row, ["metrics.xgboost_best_Specificity"])),
                "f1": to_float(first_existing(row, ["metrics.xgboost_best_F1"])),
                "mcc": to_float(first_existing(row, ["metrics.xgboost_best_MCC"])),
                "n_estimators": first_existing(row, ["params.xgboost_best_n_estimators"]),
                "max_depth": first_existing(row, ["params.xgboost_best_max_depth"]),
                "learning_rate": first_existing(row, ["params.xgboost_best_learning_rate"]),
                "features": first_existing(row, ["params.xgboost_best_features"]),
            }
        )

    normalized = pd.DataFrame(normalized_rows)
    if "start_time" in normalized.columns:
        normalized = normalized.sort_values("start_time", ascending=False)
    return normalized


def load_artifact_index(client: MlflowClient, runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()

    rows = []
    for _, run in runs.iterrows():
        try:
            artifacts = list_artifacts_recursive(client, run["run_id"])
        except Exception:
            artifacts = []

        for artifact in artifacts:
            rows.append(
                {
                    "run_id": run["run_id"],
                    "run_name": run["run_name"],
                    "dataset": run["dataset"],
                    "artifact_uri": run["artifact_uri"],
                    **artifact,
                }
            )

    return pd.DataFrame(rows)

