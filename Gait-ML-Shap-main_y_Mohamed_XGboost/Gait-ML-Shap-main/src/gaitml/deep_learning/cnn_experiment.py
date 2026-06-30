"""End-to-end CNN experiment runner with Excel and MLflow outputs."""

from ast import literal_eval
from itertools import product
from pathlib import Path
import re

import mlflow
import numpy as np
import pandas as pd
import torch

from gaitml import logger
from gaitml.deep_learning.cnn_data import (
    load_processed_sheets,
    prepare_arrays,
    rank_features,
    read_dataset_config,
)
from gaitml.deep_learning.cnn_training import (
    cross_val_accuracy,
    metric_row,
    permutation_importance,
    predict,
    save_training_plot,
    train_one_model,
)
from gaitml.utils.helpers import read_yaml


def candidate_params(training_config):
    for feature_count, hidden_channels, kernel_size, dropout, learning_rate, weight_decay in product(
        training_config["feature_counts"],
        training_config["hidden_channels"],
        training_config["kernel_sizes"],
        training_config["dropout_values"],
        training_config["learning_rates"],
        training_config["weight_decays"],
    ):
        yield {
            "feature_count": feature_count,
            "hidden_channels": hidden_channels,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": training_config["epochs"],
            "batch_size": training_config["batch_size"],
            "patience": training_config["patience"],
        }


def mlflow_name(value):
    return re.sub(r"[^A-Za-z0-9_.\- /]", "_", str(value)).replace(" ", "_")


def run_dataset(dataset_name, cnn_config, project_root):
    dataset_cfg = cnn_config["datasets"][dataset_name]
    training_cfg = cnn_config["training"]
    dataset_config = read_yaml(Path(project_root) / dataset_cfg["config_file"])
    train_path, test_path, label_column, classes = read_dataset_config(dataset_config, project_root)
    output_dir = Path(project_root) / dataset_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.set_tracking_uri(cnn_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cnn_config["mlflow"]["experiment_name"])

    sheets = load_processed_sheets(
        train_path,
        test_path,
        label_column,
        use_existing_train_test=training_cfg["use_existing_train_test"],
    )

    all_scores = {}
    all_top3 = {}

    with mlflow.start_run(run_name=f"cnn_{dataset_name}") as parent_run:
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("architecture", "1D CNN over ordered biomechanical feature vector")
        mlflow.log_param("device", str(device))
        mlflow.log_dict(cnn_config, "cnn_config.yaml")

        for sheet_name, (train_df, test_df) in sheets.items():
            logger.info(f"Running CNN dataset={dataset_name} sheet={sheet_name}")
            X_train, X_test, y_train, y_test, feature_names, encoder, scaler = prepare_arrays(
                train_df,
                test_df,
                label_column,
                training_cfg["random_state"],
                training_cfg["test_size"],
            )
            ranked_indices, ranking_df = rank_features(X_train, y_train, feature_names, training_cfg["random_state"])

            sheet_scores = []
            artifacts_dir = output_dir / f"{dataset_name}_cnn_{sheet_name}"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            ranking_path = artifacts_dir / "cnn_feature_ranking.xlsx"
            ranking_df.to_excel(ranking_path, index=False)

            for model_idx, params in enumerate(candidate_params(training_cfg), start=1):
                selected = ranked_indices[: min(params["feature_count"], X_train.shape[1])]
                selected_features = [feature_names[i] for i in selected]
                X_train_sel = X_train[:, selected]
                X_test_sel = X_test[:, selected]

                cv_accuracy = cross_val_accuracy(
                    X_train_sel,
                    y_train,
                    params,
                    training_cfg["cv_splits"],
                    training_cfg["random_state"],
                    device,
                )
                model, history, best_loss = train_one_model(
                    X_train_sel,
                    y_train,
                    params,
                    training_cfg["validation_size"],
                    training_cfg["random_state"],
                    device,
                )
                y_pred = predict(model, X_test_sel, params["batch_size"], device)
                row = metric_row(y_test.astype(int), y_pred.astype(int))
                row.update({
                    "Name_Model": f"CNN_{model_idx}",
                    "#Features": len(selected),
                    "Feature_idx": model_idx - 1,
                    "Model_Type": "CNN_1D",
                    "CV_accuracy": cv_accuracy,
                    "Hidden_Channels": params["hidden_channels"],
                    "Kernel_Size": params["kernel_size"],
                    "Dropout": params["dropout"],
                    "Learning_Rate": params["learning_rate"],
                    "Weight_Decay": params["weight_decay"],
                    "Epochs_Configured": params["epochs"],
                    "Epochs_Trained": len(history),
                    "Batch_Size": params["batch_size"],
                    "Best_Val_Loss": best_loss,
                    "Features": str(selected_features),
                })
                sheet_scores.append(row)

            scores_df = pd.DataFrame(sheet_scores).sort_values(
                ["Test_Accuracy", "CV_accuracy", "MCC", "F1"],
                ascending=False,
            )
            all_scores[sheet_name] = scores_df
            all_top3[sheet_name] = scores_df.head(3).copy()

            best = scores_df.iloc[0]
            selected_features = literal_eval(best["Features"])
            selected = [feature_names.index(name) for name in selected_features]
            best_params = {
                "feature_count": int(best["#Features"]),
                "hidden_channels": int(best["Hidden_Channels"]),
                "kernel_size": int(best["Kernel_Size"]),
                "dropout": float(best["Dropout"]),
                "learning_rate": float(best["Learning_Rate"]),
                "weight_decay": float(best["Weight_Decay"]),
                "epochs": training_cfg["epochs"],
                "batch_size": training_cfg["batch_size"],
                "patience": training_cfg["patience"],
            }
            best_model, history, _ = train_one_model(
                X_train[:, selected],
                y_train,
                best_params,
                training_cfg["validation_size"],
                training_cfg["random_state"],
                device,
            )
            model_path = artifacts_dir / "best_cnn_model.pt"
            torch.save({
                "model_state_dict": best_model.state_dict(),
                "params": best_params,
                "features": selected_features,
                "classes": list(encoder.classes_),
            }, model_path)
            plot_path = artifacts_dir / "best_cnn_training_curve.png"
            save_training_plot(history, plot_path)
            importance_df = permutation_importance(
                best_model,
                X_test[:, selected],
                y_test.astype(int),
                selected_features,
                best_params["batch_size"],
                device,
                max_features=training_cfg["max_permutation_features"],
                random_state=training_cfg["random_state"],
            )
            importance_path = artifacts_dir / "best_cnn_permutation_importance.xlsx"
            importance_df.to_excel(importance_path, index=False)

            metric_prefix = mlflow_name(sheet_name)
            mlflow.log_metrics({f"{metric_prefix}_{k}": float(best[k]) for k in ["Test_Accuracy", "CV_accuracy", "Sensitivity", "Specificity", "F1", "MCC", "PPV", "NPV"]})
            mlflow.log_params({f"{metric_prefix}_{k}": best_params[k] for k in best_params})
            mlflow.log_artifacts(str(artifacts_dir), artifact_path=f"{dataset_name}/{mlflow_name(sheet_name)}")

        scores_path = Path(project_root) / "results" / f"{dataset_cfg['result_prefix']}_Scores_CNN.xlsx"
        top3_path = Path(project_root) / "results" / f"{dataset_cfg['result_prefix']}_Scores_CNN_Top3.xlsx"
        with pd.ExcelWriter(scores_path) as writer:
            for sheet_name, df in all_scores.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        with pd.ExcelWriter(top3_path) as writer:
            for sheet_name, df in all_top3.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        mlflow.log_artifact(str(scores_path), artifact_path=f"{dataset_name}/excel")
        mlflow.log_artifact(str(top3_path), artifact_path=f"{dataset_name}/excel")
    return scores_path, top3_path
