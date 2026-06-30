from pathlib import Path
import os

import mlflow
from mlflow.exceptions import MlflowException
import pandas as pd
import numpy as np


# ============================================================
# 1) PROJECT ROOTS
# ============================================================

SVM_ROOT = Path(
    os.getenv(
        "SVM_ROOT",
        r"C:\Users\k6qqj\Desktop\TASK_SVM\Gait-ML-Shap-main\Gait-ML-Shap-main",
    )
)

XGB_ROOT = Path(
    os.getenv(
        "XGB_ROOT",
        r"C:\Users\k6qqj\Desktop\Gait-ML-Shap-main_y_Mohamed\Gait-ML-Shap-main",
    )
)

RF_ROOT = Path(
    os.getenv(
        "RF_ROOT",
        r"C:\Users\k6qqj\Desktop\Gait-ML-Shap-main_y_Mohamed -Random_Forest\Gait-ML-Shap-main",
    )
)

MLFLOW_DB = Path(os.getenv("MLFLOW_DB", str(SVM_ROOT / "mlflow.db")))


def p(root, relative_path):
    return root / relative_path


# ============================================================
# 2) MLFLOW SETUP
# ============================================================

REMOTE_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
LOCAL_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

IS_REMOTE_TRACKING = bool(REMOTE_TRACKING_URI)

DEFAULT_LOG_ARTIFACTS = "false" if IS_REMOTE_TRACKING else "true"
LOG_ARTIFACTS = os.getenv("LOG_MLFLOW_ARTIFACTS", DEFAULT_LOG_ARTIFACTS).lower() == "true"
LOG_RAW_DATA = os.getenv("LOG_MLFLOW_RAW_DATA", "false").lower() == "true"

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Young_and_Old")


def set_experiment_safely(base_experiment_name):
    """
    Set the MLflow experiment.

    If the experiment was deleted before, MLflow refuses to reuse the same name.
    In that case, this function automatically tries:
        Young_and_Old_v2
        Young_and_Old_v3
        ...
    """

    try:
        mlflow.set_experiment(base_experiment_name)
        return base_experiment_name

    except MlflowException as error:
        error_message = str(error)

        if "Cannot set a deleted experiment" not in error_message:
            raise

        print(f"[MLflow] Experiment '{base_experiment_name}' was deleted before.")
        print("[MLflow] Trying a new experiment name instead...")

        for version in range(2, 21):
            new_name = f"{base_experiment_name}_v{version}"

            try:
                mlflow.set_experiment(new_name)
                print(f"[MLflow] Using experiment name: {new_name}")
                return new_name

            except MlflowException as retry_error:
                retry_message = str(retry_error)

                if "Cannot set a deleted experiment" in retry_message:
                    continue

                raise

        raise MlflowException(
            f"Could not create an active experiment after trying {base_experiment_name}_v2 to {base_experiment_name}_v20."
        )


def configure_mlflow():
    tracking_uri = REMOTE_TRACKING_URI if IS_REMOTE_TRACKING else LOCAL_TRACKING_URI

    mlflow.set_tracking_uri(tracking_uri)
    active_experiment_name = set_experiment_safely(EXPERIMENT_NAME)

    print("\n================ MLflow Configuration ================")
    print(f"Tracking URI        : {mlflow.get_tracking_uri()}")
    print(f"Experiment name     : {active_experiment_name}")
    print(f"Remote tracking     : {IS_REMOTE_TRACKING}")
    print(f"Log artifacts       : {LOG_ARTIFACTS}")
    print(f"Log raw/train/test  : {LOG_RAW_DATA}")
    print("======================================================\n")

    return active_experiment_name


# ============================================================
# 3) SVM FILES - YOUNG/OLD
# ============================================================

svm_files = {
    "config_current": p(SVM_ROOT, "configs/config.yaml"),
    "config_young_old": p(SVM_ROOT, "configs/config_young_old_2024.yaml"),

    "svm_raw_file": p(
        SVM_ROOT,
        "data/raw/2024 ALL Post-Processing Speeds Young and Older Adults ALL Features - FINAL AUGUST 20.xlsx",
    ),
    "feature_names_file": p(
        SVM_ROOT,
        "data/raw/ISB Abstract - Joint Angles and TS Variable Names.xlsx",
    ),
    "highly_correlated_features": p(
        SVM_ROOT,
        "data/raw/Highly correlated features.xlsx",
    ),

    "svm_phase1_JA_TS": p(
        SVM_ROOT,
        "data/processed/young_old_2024_JA_TS_features.xlsx",
    ),
    "svm_phase2_NH_correlated": p(
        SVM_ROOT,
        "data/processed/young_old_2024_NH_correlated_features.xlsx",
    ),
    "svm_train_file": p(
        SVM_ROOT,
        "data/processed/young_old_2024_train.xlsx",
    ),
    "svm_test_file": p(
        SVM_ROOT,
        "data/processed/young_old_2024_test.xlsx",
    ),

    "svm_sfs_results": p(
        SVM_ROOT,
        "src/gaitml/features/young_old_2024_phase4_1_SFS_results.xlsx",
    ),
    "svm_step2_filtered": p(
        SVM_ROOT,
        "young_old_2024_Step_2_Results_Filtered.xlsx",
    ),
    "svm_scores_rbf": p(
        SVM_ROOT,
        "young_old_2024_Scores_RBF.xlsx",
    ),
    "svm_scores_rbf_top3": p(
        SVM_ROOT,
        "young_old_2024_Scores_RBF_Top3.xlsx",
    ),

    "svm_shap_excel": p(
        SVM_ROOT,
        "output-young_old/young_old_2024_SHAP_data.xlsx",
    ),
    "svm_time_stats": p(
        SVM_ROOT,
        "young_old_2024_SVM_time_variable_stats.xlsx",
    ),
}

svm_folders = {
    "svm_shap_plots": p(
        SVM_ROOT,
        "output-young_old/young_old_2024_SHAP",
    ),
    "svm_time_scatter_plots": p(
        SVM_ROOT,
        "young_old_2024_SVM_time_variable_plots",
    ),
}


# ============================================================
# 4) XGBOOST FILES - YOUNG/OLD
# ============================================================

xgb_files = {
    "config_current": p(XGB_ROOT, "configs/config.yaml"),

    "xgb_raw_file": p(
        XGB_ROOT,
        "data/raw/2024 ALL Post-Processing Speeds Young and Older Adults ALL Features - FINAL AUGUST 20.xlsx",
    ),
    "feature_names_file": p(
        XGB_ROOT,
        "data/raw/ISB Abstract - Joint Angles and TS Variable Names.xlsx",
    ),
    "highly_correlated_features": p(
        XGB_ROOT,
        "data/raw/Highly correlated features.xlsx",
    ),

    "xgb_phase1_JA_TS": p(
        XGB_ROOT,
        "results/young_old_2024_JA_TS_features.xlsx",
    ),
    "xgb_phase2_NH_correlated": p(
        XGB_ROOT,
        "results/young_old_2024_NH_correlated_features.xlsx",
    ),
    "xgb_train_file": p(
        XGB_ROOT,
        "results/young_old_2024_train.xlsx",
    ),
    "xgb_test_file": p(
        XGB_ROOT,
        "results/young_old_2024_test.xlsx",
    ),

    "xgb_sfs_results": p(
        XGB_ROOT,
        "results/young_old_2024_phase4_1_SFS_results_XGBoost.xlsx",
    ),
    "xgb_step2_results": p(
        XGB_ROOT,
        "results/young_old_2024_Step_2_Results_XGBoost.xlsx",
    ),
    "xgb_scores": p(
        XGB_ROOT,
        "results/young_old_2024_Scores_XGBoost.xlsx",
    ),
    "xgb_scores_top3": p(
        XGB_ROOT,
        "results/young_old_2024_Scores_XGBoost_Top3.xlsx",
    ),

    "xgb_shap_excel": p(
        XGB_ROOT,
        "output-young_old/young_old_2024_SHAP_data_XGBoost.xlsx",
    ),
    "xgb_time_stats": p(
        XGB_ROOT,
        "young_old_2024_XGBoost_time_variable_stats.xlsx",
    ),
}

xgb_folders = {
    "xgboost_shap_plots": p(
        XGB_ROOT,
        "output-young_old/young_old_2024_SHAP",
    ),
    "xgboost_time_scatter_plots": p(
        XGB_ROOT,
        "young_old_2024_XGBOOST_time_variable_plots",
    ),
}


# ============================================================
# 5) RANDOM FOREST FILES - YOUNG/OLD
# ============================================================

rf_files = {
    "config_current": p(RF_ROOT, "configs/config.yaml"),

    "rf_raw_file": p(
        RF_ROOT,
        "data/raw/2024 ALL Post-Processing Speeds Young and Older Adults ALL Features - FINAL AUGUST 20.xlsx",
    ),
    "feature_names_file": p(
        RF_ROOT,
        "data/raw/ISB Abstract - Joint Angles and TS Variable Names.xlsx",
    ),
    "highly_correlated_features": p(
        RF_ROOT,
        "data/raw/Highly correlated features.xlsx",
    ),

    "rf_phase1_JA_TS": p(
        RF_ROOT,
        "results/young_old_2024_JA_TS_features.xlsx",
    ),
    "rf_phase2_NH_correlated": p(
        RF_ROOT,
        "results/young_old_2024_NH_correlated_features.xlsx",
    ),
    "rf_train_file": p(
        RF_ROOT,
        "results/young_old_2024_train.xlsx",
    ),
    "rf_test_file": p(
        RF_ROOT,
        "results/young_old_2024_test.xlsx",
    ),

    "rf_sfs_results": p(
        RF_ROOT,
        "results/young_old_2024_phase4_1_SFS_results_RandomForest.xlsx",
    ),
    "rf_step2_results": p(
        RF_ROOT,
        "results/young_old_2024_Step_2_Results_RandomForest.xlsx",
    ),
    "rf_scores": p(
        RF_ROOT,
        "results/young_old_2024_Scores_RandomForest.xlsx",
    ),
    "rf_scores_top3": p(
        RF_ROOT,
        "results/young_old_2024_Scores_RandomForest_Top3.xlsx",
    ),

    "rf_shap_excel": p(
        RF_ROOT,
        "output-young_old/young_old_2024_SHAP_data_RandomForest.xlsx",
    ),
}

rf_folders = {
    "random_forest_shap_plots": p(
        RF_ROOT,
        "output-young_old/young_old_2024_SHAP_RandomForest",
    ),
}


# ============================================================
# 6) HELPER FUNCTIONS
# ============================================================

def safe_int(value, default=None):
    try:
        if pd.isna(value):
            return default

        return int(float(value))

    except Exception:
        return default


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def log_param_safe(name, value):
    if value is None:
        return

    value = str(value)

    if len(value) > 500:
        value = value[:500] + "..."

    mlflow.log_param(name, value)


def log_metric_safe(name, value):
    value = safe_float(value)

    if value is None:
        return

    mlflow.log_metric(name, value)


def read_top3_file(file_path):
    path = Path(file_path)

    if not path.exists():
        print(f"Top3 file not found: {path}")
        return None

    excel = pd.ExcelFile(path)
    sheet_name = excel.sheet_names[0]

    df = pd.read_excel(path, sheet_name=sheet_name)

    print(f"\nLoaded Top3 file: {path}")
    print(f"Sheet used: {sheet_name}")
    print(df.head())

    return df


def get_cv_column(df):
    if "CV_accuracy" in df.columns:
        return "CV_accuracy"

    if "CV_Accuracy" in df.columns:
        return "CV_Accuracy"

    if "SFS_CV_Accuracy" in df.columns:
        return "SFS_CV_Accuracy"

    return None


def log_top3_metrics(df_top3, model_prefix):
    if df_top3 is None or df_top3.empty:
        print(f"No Top3 data available for {model_prefix}")
        return

    cv_col = get_cv_column(df_top3)
    best_row = df_top3.iloc[0]

    log_param_safe(f"{model_prefix}_best_model_name", best_row.get("Name_Model", ""))
    log_param_safe(f"{model_prefix}_best_num_features", safe_int(best_row.get("#Features", 0), 0))

    if "C_Value" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_C_value", safe_float(best_row.get("C_Value", None)))

    if "Gamma_Value" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_gamma_value", safe_float(best_row.get("Gamma_Value", None)))

    if "n_estimators" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_n_estimators", safe_int(best_row.get("n_estimators", None)))

    if "max_depth" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_max_depth", safe_int(best_row.get("max_depth", None)))

    if "learning_rate" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_learning_rate", safe_float(best_row.get("learning_rate", None)))

    if "subsample" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_subsample", safe_float(best_row.get("subsample", None)))

    if "colsample_bytree" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_colsample_bytree", safe_float(best_row.get("colsample_bytree", None)))

    if "min_samples_split" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_min_samples_split", safe_int(best_row.get("min_samples_split", None)))

    if "min_samples_leaf" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_min_samples_leaf", safe_int(best_row.get("min_samples_leaf", None)))

    if "max_features" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_max_features", best_row.get("max_features", ""))

    if "Random_State" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_random_state", best_row.get("Random_State", ""))

    if "Features" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_features", best_row.get("Features", ""))

    if "Ordered Indices" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_ordered_indices", best_row.get("Ordered Indices", ""))

    if "Sheet" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_sheet", best_row.get("Sheet", ""))

    if cv_col is not None:
        log_metric_safe(f"{model_prefix}_best_CV_accuracy", best_row.get(cv_col, None))

    metric_columns = [
        "Test_Accuracy",
        "Sensitivity",
        "Specificity",
        "F1",
        "MCC",
        "NPV",
        "PPV",
        "Likelihood_Ratio",
    ]

    for metric in metric_columns:
        if metric in df_top3.columns:
            log_metric_safe(f"{model_prefix}_best_{metric}", best_row.get(metric, None))

    for rank, (_, row) in enumerate(df_top3.head(3).iterrows(), start=1):
        log_param_safe(f"{model_prefix}_top{rank}_model_name", row.get("Name_Model", ""))
        log_param_safe(f"{model_prefix}_top{rank}_num_features", safe_int(row.get("#Features", 0), 0))

        if cv_col is not None:
            log_metric_safe(f"{model_prefix}_top{rank}_CV_accuracy", row.get(cv_col, None))

        for metric in ["Test_Accuracy", "Sensitivity", "Specificity", "F1", "MCC"]:
            if metric in df_top3.columns:
                log_metric_safe(f"{model_prefix}_top{rank}_{metric}", row.get(metric, None))


def log_test_class_counts(test_file, model_prefix):
    path = Path(test_file)

    if not path.exists():
        print(f"Test file not found for class counts: {path}")
        return

    excel = pd.ExcelFile(path)
    all_dfs = []

    for sheet_name in excel.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet_name)
        df["Sheet"] = sheet_name
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)

    if "Group" not in df_all.columns:
        print(f"Group column not found in test file: {path}")
        return

    counts = df_all["Group"].value_counts().to_dict()

    for group_value, count in counts.items():
        log_param_safe(f"{model_prefix}_test_class_count_{group_value}", int(count))

    log_param_safe(f"{model_prefix}_test_class_counts", counts)

    print(f"{model_prefix} test class counts: {counts}")


def is_private_or_raw_artifact(artifact_name, file_path):
    artifact_name_lower = str(artifact_name).lower()
    file_path_lower = str(file_path).lower()

    sensitive_keywords = [
        "raw",
        "train",
        "test",
        "feature_names",
        "highly_correlated",
        "ja_ts",
        "nh_correlated",
        "data/raw",
        "data\\raw",
        "data/processed",
        "data\\processed",
    ]

    return any(
        keyword in artifact_name_lower or keyword in file_path_lower
        for keyword in sensitive_keywords
    )


def log_individual_files(files_dict, artifact_folder):
    if not LOG_ARTIFACTS:
        print(f"Artifact logging disabled. Skipping folder: {artifact_folder}")
        return

    for artifact_name, file_path in files_dict.items():
        path = Path(file_path)

        if not path.exists():
            print(f"Skipped missing file: {path}")
            continue

        if IS_REMOTE_TRACKING and not LOG_RAW_DATA:
            if is_private_or_raw_artifact(artifact_name, path):
                print(f"Skipped private/raw artifact for remote tracking: {path}")
                continue

        mlflow.log_artifact(str(path), artifact_path=artifact_folder)
        print(f"Logged file: {path}")


def log_folder_files(
    folder_path,
    artifact_folder,
    extensions=None,
    include_keyword=None,
    exclude_keyword=None,
):
    if not LOG_ARTIFACTS:
        print(f"Artifact logging disabled. Skipping folder: {artifact_folder}")
        return

    folder = Path(folder_path)

    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    if extensions is None:
        extensions = [
            ".png",
            ".jpg",
            ".jpeg",
            ".xlsx",
            ".xls",
            ".csv",
            ".docx",
            ".txt",
        ]

    logged_count = 0

    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in extensions:
            continue

        file_name = file_path.name

        if include_keyword is not None and include_keyword not in file_name:
            continue

        if exclude_keyword is not None and exclude_keyword in file_name:
            continue

        if IS_REMOTE_TRACKING and not LOG_RAW_DATA:
            if is_private_or_raw_artifact(file_name, file_path):
                print(f"Skipped private/raw folder artifact for remote tracking: {file_path}")
                continue

        mlflow.log_artifact(str(file_path), artifact_path=artifact_folder)

        print(f"Logged folder file: {file_path}")
        logged_count += 1

    print(f"Total files logged from {folder}: {logged_count}")


def log_common_run_params(model_type, active_experiment_name):
    mlflow.log_param("dataset_name", "Young vs Older 2024")
    mlflow.log_param("experiment_name", active_experiment_name)
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("group_mapping", "Young=0, Older=1")
    mlflow.log_param("feature_selection_method", "SFS")
    mlflow.log_param("phase", "Top3_SHAP")
    mlflow.log_param("tracking_uri", mlflow.get_tracking_uri())
    mlflow.log_param("remote_tracking", IS_REMOTE_TRACKING)
    mlflow.log_param("log_artifacts", LOG_ARTIFACTS)
    mlflow.log_param("log_raw_data", LOG_RAW_DATA)

    if not IS_REMOTE_TRACKING:
        mlflow.log_param("tracking_db", str(MLFLOW_DB))


# ============================================================
# 7) MAIN LOGGING PIPELINE
# ============================================================

def main():
    active_experiment_name = configure_mlflow()

    svm_top3 = read_top3_file(svm_files["svm_scores_rbf_top3"])
    xgb_top3 = read_top3_file(xgb_files["xgb_scores_top3"])
    rf_top3 = read_top3_file(rf_files["rf_scores_top3"])

    with mlflow.start_run(run_name="Young_Old_2024_SVM_RBF_All_Files"):
        log_common_run_params("SVM_RBF", active_experiment_name)

        log_top3_metrics(
            svm_top3,
            model_prefix="svm",
        )

        log_test_class_counts(
            svm_files["svm_test_file"],
            model_prefix="svm",
        )

        log_individual_files(
            svm_files,
            artifact_folder="young_old_svm_files",
        )

        log_folder_files(
            folder_path=svm_folders["svm_shap_plots"],
            artifact_folder="svm_shap_plots",
            extensions=[".png", ".jpg", ".jpeg"],
            exclude_keyword="XGBoost",
        )

        log_folder_files(
            folder_path=svm_folders["svm_time_scatter_plots"],
            artifact_folder="svm_time_scatter_plots",
            extensions=[".png", ".jpg", ".jpeg"],
        )

    print("Done. Young/Old SVM run logged successfully.")

    with mlflow.start_run(run_name="Young_Old_2024_XGBoost_All_Files"):
        log_common_run_params("XGBoost", active_experiment_name)

        log_top3_metrics(
            xgb_top3,
            model_prefix="xgboost",
        )

        log_test_class_counts(
            xgb_files["xgb_test_file"],
            model_prefix="xgboost",
        )

        log_individual_files(
            xgb_files,
            artifact_folder="young_old_xgboost_files",
        )

        log_folder_files(
            folder_path=xgb_folders["xgboost_shap_plots"],
            artifact_folder="xgboost_shap_plots",
            extensions=[".png", ".jpg", ".jpeg"],
            include_keyword="XGBoost",
        )

        log_folder_files(
            folder_path=xgb_folders["xgboost_time_scatter_plots"],
            artifact_folder="xgboost_time_scatter_plots",
            extensions=[".png", ".jpg", ".jpeg"],
        )

    print("Done. Young/Old XGBoost run logged successfully.")

    with mlflow.start_run(run_name="Young_Old_2024_RandomForest_All_Files"):
        log_common_run_params("RandomForest", active_experiment_name)

        log_top3_metrics(
            rf_top3,
            model_prefix="random_forest",
        )

        log_test_class_counts(
            rf_files["rf_test_file"],
            model_prefix="random_forest",
        )

        log_individual_files(
            rf_files,
            artifact_folder="young_old_random_forest_files",
        )

        log_folder_files(
            folder_path=rf_folders["random_forest_shap_plots"],
            artifact_folder="random_forest_shap_plots",
            extensions=[".png", ".jpg", ".jpeg"],
        )

    print("Done. Young/Old Random Forest run logged successfully.")
    print("Done. Young/Old SVM + XGBoost + Random Forest MLflow logging completed.")


if __name__ == "__main__":
    main()