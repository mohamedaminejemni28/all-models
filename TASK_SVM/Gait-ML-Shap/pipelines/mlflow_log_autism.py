from pathlib import Path
import os

import mlflow
import pandas as pd
import numpy as np


# ============================================================
# 1) PROJECT ROOTS
# ============================================================
# You can keep the default Windows paths, or override them without changing
# the code by setting environment variables:
#
# PowerShell examples:
# $env:SVM_ROOT="C:\Users\k6qqj\Desktop\TASK\Gait-ML-Shap-main\Gait-ML-Shap"
# $env:XGB_ROOT="C:\Users\k6qqj\Desktop\Gait-ML-Shap-main_y_Mohamed\Gait-ML-Shap-main"
# $env:RF_ROOT="C:\Users\k6qqj\Desktop\Gait-ML-Shap-main_y_Mohamed -Random_Forest\Gait-ML-Shap-main"

SVM_ROOT = Path(
    os.getenv(
        "SVM_ROOT",
        r"C:\Users\k6qqj\Desktop\TASK\Gait-ML-Shap-main\Gait-ML-Shap",
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


def p(root, relative_path):
    return root / relative_path


# ============================================================
# 2) MLFLOW SETUP
# ============================================================
# LOCAL MODE:
# If MLFLOW_TRACKING_URI is not set, this script logs to the local SQLite DB:
# sqlite:///.../mlflow.db
#
# DAGSHUB / REMOTE MODE:
# Set these variables before running the script:
#
# $env:MLFLOW_TRACKING_URI="https://dagshub.com/YOUR_USERNAME/YOUR_REPO_NAME.mlflow"
# $env:MLFLOW_TRACKING_USERNAME="YOUR_DAGSHUB_USERNAME"
# $env:MLFLOW_TRACKING_PASSWORD="YOUR_DAGSHUB_TOKEN"
# $env:MLFLOW_EXPERIMENT_NAME="Autism"
#
# IMPORTANT:
# Do not write your DagsHub token inside this script.

MLFLOW_DB = Path(os.getenv("MLFLOW_DB", str(SVM_ROOT / "mlflow.db")))

REMOTE_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
LOCAL_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

IS_REMOTE_TRACKING = bool(REMOTE_TRACKING_URI)

# In local mode, artifacts are logged by default.
# In remote/DagsHub mode, artifacts are disabled by default for privacy.
#
# To enable artifact logging manually:
# $env:LOG_MLFLOW_ARTIFACTS="true"
#
# To allow raw/train/test data artifacts manually:
# $env:LOG_MLFLOW_RAW_DATA="true"

DEFAULT_LOG_ARTIFACTS = "false" if IS_REMOTE_TRACKING else "true"
LOG_ARTIFACTS = os.getenv("LOG_MLFLOW_ARTIFACTS", DEFAULT_LOG_ARTIFACTS).lower() == "true"
LOG_RAW_DATA = os.getenv("LOG_MLFLOW_RAW_DATA", "false").lower() == "true"

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Autism")


def configure_mlflow():
    """
    Configure MLflow for local tracking or remote tracking.

    Local default:
        sqlite:///.../mlflow.db

    Remote example for DagsHub:
        https://dagshub.com/<username>/<repo>.mlflow

    Credentials should be provided only through environment variables:
        MLFLOW_TRACKING_USERNAME
        MLFLOW_TRACKING_PASSWORD
    """

    tracking_uri = REMOTE_TRACKING_URI if IS_REMOTE_TRACKING else LOCAL_TRACKING_URI

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\n================ MLflow Configuration ================")
    print(f"Tracking URI        : {mlflow.get_tracking_uri()}")
    print(f"Experiment name     : {EXPERIMENT_NAME}")
    print(f"Remote tracking     : {IS_REMOTE_TRACKING}")
    print(f"Log artifacts       : {LOG_ARTIFACTS}")
    print(f"Log raw/train/test  : {LOG_RAW_DATA}")
    print("======================================================\n")


# ============================================================
# 3) SVM FILES
# ============================================================

svm_files = {
    "config_current": p(SVM_ROOT, "configs/config.yaml"),

    "svm_raw_file": p(
        SVM_ROOT,
        "data/raw/2024 FINAL - Autism Study MAR2024 ALL (AM+NM) Features.xlsx"
    ),
    "feature_names_file": p(
        SVM_ROOT,
        "data/raw/ISB Abstract - Joint Angles and TS Variable Names.xlsx"
    ),
    "highly_correlated_features": p(
        SVM_ROOT,
        "data/raw/Highly correlated features.xlsx"
    ),

    "svm_phase1_JA_TS": p(
        SVM_ROOT,
        "data/processed/autism_2024_JA_TS_features.xlsx"
    ),
    "svm_phase2_NH_correlated": p(
        SVM_ROOT,
        "data/processed/autism_2024_NH_correlated_features.xlsx"
    ),
    "svm_train_file": p(
        SVM_ROOT,
        "data/processed/autism_2024_train.xlsx"
    ),
    "svm_test_file": p(
        SVM_ROOT,
        "data/processed/autism_2024_test.xlsx"
    ),

    "svm_sfs_results": p(
        SVM_ROOT,
        "src/gaitml/features/autism_2024_phase4_1_SFS_results.xlsx"
    ),
    "svm_step2_results": p(
        SVM_ROOT,
        "autism_2024_Step_2_Results_Filtered.xlsx"
    ),
    "svm_scores": p(
        SVM_ROOT,
        "autism_2024_Scores_RBF.xlsx"
    ),
    "svm_scores_top3": p(
        SVM_ROOT,
        "autism_2024_Scores_RBF_Top3.xlsx"
    ),

    "svm_shap_excel": p(
        SVM_ROOT,
        "output-autism/autism_2024_SHAP_data.xlsx"
    ),
}

svm_folders = {
    "svm_shap_plots": p(
        SVM_ROOT,
        "output-autism/autism_2024_SHAP"
    ),
}


# ============================================================
# 4) XGBOOST FILES
# ============================================================

xgb_files = {
    "config_current": p(XGB_ROOT, "configs/config.yaml"),

    "xgb_raw_file": p(
        XGB_ROOT,
        "data/raw/2024 FINAL - Autism Study MAR2024 ALL (AM+NM) Features.xlsx"
    ),
    "feature_names_file": p(
        XGB_ROOT,
        "data/raw/ISB Abstract - Joint Angles and TS Variable Names.xlsx"
    ),
    "highly_correlated_features": p(
        XGB_ROOT,
        "data/raw/Highly correlated features.xlsx"
    ),

    "xgb_phase1_JA_TS": p(
        XGB_ROOT,
        "results/autism_2024_JA_TS_features.xlsx"
    ),
    "xgb_phase2_NH_correlated": p(
        XGB_ROOT,
        "results/autism_2024_NH_correlated_features.xlsx"
    ),
    "xgb_train_file": p(
        XGB_ROOT,
        "results/autism_2024_train.xlsx"
    ),
    "xgb_test_file": p(
        XGB_ROOT,
        "results/autism_2024_test.xlsx"
    ),

    "xgb_sfs_results": p(
        XGB_ROOT,
        "results/autism_2024_phase4_1_SFS_results_XGBoost.xlsx"
    ),
    "xgb_step2_results": p(
        XGB_ROOT,
        "results/autism_2024_Step_2_Results_XGBoost.xlsx"
    ),
    "xgb_scores": p(
        XGB_ROOT,
        "results/autism_2024_Scores_XGBoost.xlsx"
    ),
    "xgb_scores_top3": p(
        XGB_ROOT,
        "results/autism_2024_Scores_XGBoost_Top3.xlsx"
    ),

    "xgb_shap_excel": p(
        XGB_ROOT,
        "output-autism/autism_2024_SHAP_data_XGBoost.xlsx"
    ),
}

xgb_folders = {
    "xgb_shap_plots": p(
        XGB_ROOT,
        "output-autism/autism_2024_SHAP"
    ),
}


# ============================================================
# 5) RANDOM FOREST FILES
# ============================================================

rf_files = {
    "config_current": p(RF_ROOT, "configs/config.yaml"),

    "rf_raw_file": p(
        RF_ROOT,
        "data/raw/2024 FINAL - Autism Study MAR2024 ALL (AM+NM) Features.xlsx"
    ),
    "feature_names_file": p(
        RF_ROOT,
        "data/raw/ISB Abstract - Joint Angles and TS Variable Names.xlsx"
    ),
    "highly_correlated_features": p(
        RF_ROOT,
        "data/raw/Highly correlated features.xlsx"
    ),

    "rf_phase1_JA_TS": p(
        RF_ROOT,
        "results/autism_2024_JA_TS_features.xlsx"
    ),
    "rf_phase2_NH_correlated": p(
        RF_ROOT,
        "results/autism_2024_NH_correlated_features.xlsx"
    ),
    "rf_train_file": p(
        RF_ROOT,
        "results/autism_2024_train.xlsx"
    ),
    "rf_test_file": p(
        RF_ROOT,
        "results/autism_2024_test.xlsx"
    ),

    "rf_sfs_results": p(
        RF_ROOT,
        "results/autism_2024_phase4_1_SFS_results_RandomForest.xlsx"
    ),
    "rf_step2_results": p(
        RF_ROOT,
        "results/autism_2024_Step_2_Results_RandomForest.xlsx"
    ),
    "rf_scores": p(
        RF_ROOT,
        "results/autism_2024_Scores_RandomForest.xlsx"
    ),
    "rf_scores_top3": p(
        RF_ROOT,
        "results/autism_2024_Scores_RandomForest_Top3.xlsx"
    ),

    "rf_shap_excel": p(
        RF_ROOT,
        "output-autism/autism_2024_SHAP_data_RandomForest.xlsx"
    ),
}

rf_folders = {
    "rf_shap_plots": p(
        RF_ROOT,
        "output-autism/autism_2024_SHAP_RandomForest"
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

    return None


def log_top3_metrics(df_top3, model_prefix):
    if df_top3 is None or df_top3.empty:
        print(f"No Top3 data available for {model_prefix}")
        return

    cv_col = get_cv_column(df_top3)
    best_row = df_top3.iloc[0]

    # Best model general info
    log_param_safe(
        f"{model_prefix}_best_model_name",
        best_row.get("Name_Model", "")
    )

    log_param_safe(
        f"{model_prefix}_best_num_features",
        safe_int(best_row.get("#Features", 0), 0)
    )

    # SVM parameters
    if "C_Value" in df_top3.columns:
        log_param_safe(
            f"{model_prefix}_best_C_value",
            safe_float(best_row.get("C_Value", None))
        )

    if "Gamma_Value" in df_top3.columns:
        log_param_safe(
            f"{model_prefix}_best_gamma_value",
            safe_float(best_row.get("Gamma_Value", None))
        )

    # XGBoost / Random Forest common parameters
    if "n_estimators" in df_top3.columns:
        log_param_safe(
            f"{model_prefix}_best_n_estimators",
            safe_int(best_row.get("n_estimators", None))
        )

    if "max_depth" in df_top3.columns:
        log_param_safe(
            f"{model_prefix}_best_max_depth",
            safe_int(best_row.get("max_depth", None))
        )

    # XGBoost specific
    if "learning_rate" in df_top3.columns:
        log_param_safe(
            f"{model_prefix}_best_learning_rate",
            safe_float(best_row.get("learning_rate", None))
        )

    if "subsample" in df_top3.columns:
        log_param_safe(
            f"{model_prefix}_best_subsample",
            safe_float(best_row.get("subsample", None))
        )

    if "colsample_bytree" in df_top3.columns:
        log_param_safe(
            f"{model_prefix}_best_colsample_bytree",
            safe_float(best_row.get("colsample_bytree", None))
        )

    # Random Forest specific
    if "min_samples_split" in df_top3.columns:
        log_param_safe(
            f"{model_prefix}_best_min_samples_split",
            safe_int(best_row.get("min_samples_split", None))
        )

    if "min_samples_leaf" in df_top3.columns:
        log_param_safe(
            f"{model_prefix}_best_min_samples_leaf",
            safe_int(best_row.get("min_samples_leaf", None))
        )

    if "max_features" in df_top3.columns:
        log_param_safe(
            f"{model_prefix}_best_max_features",
            best_row.get("max_features", "")
        )

    # Common extra info
    if "Random_State" in df_top3.columns:
        log_param_safe(
            f"{model_prefix}_best_random_state",
            best_row.get("Random_State", "")
        )

    if "Features" in df_top3.columns:
        log_param_safe(
            f"{model_prefix}_best_features",
            best_row.get("Features", "")
        )

    if "Ordered Indices" in df_top3.columns:
        log_param_safe(
            f"{model_prefix}_best_ordered_indices",
            best_row.get("Ordered Indices", "")
        )

    if "Sheet" in df_top3.columns:
        log_param_safe(
            f"{model_prefix}_best_sheet",
            best_row.get("Sheet", "")
        )

    # Best metrics
    if cv_col is not None:
        log_metric_safe(
            f"{model_prefix}_best_CV_accuracy",
            best_row.get(cv_col, None)
        )

    metric_columns = [
        "Test_Accuracy",
        "Sensitivity",
        "Specificity",
        "F1",
        "MCC",
        "NPV",
        "PPV",
        "Likelihood_Ratio"
    ]

    for metric in metric_columns:
        if metric in df_top3.columns:
            log_metric_safe(
                f"{model_prefix}_best_{metric}",
                best_row.get(metric, None)
            )

    # Top 3 models
    for rank, (_, row) in enumerate(df_top3.head(3).iterrows(), start=1):
        log_param_safe(
            f"{model_prefix}_top{rank}_model_name",
            row.get("Name_Model", "")
        )

        log_param_safe(
            f"{model_prefix}_top{rank}_num_features",
            safe_int(row.get("#Features", 0), 0)
        )

        if cv_col is not None:
            log_metric_safe(
                f"{model_prefix}_top{rank}_CV_accuracy",
                row.get(cv_col, None)
            )

        for metric in [
            "Test_Accuracy",
            "Sensitivity",
            "Specificity",
            "F1",
            "MCC"
        ]:
            if metric in df_top3.columns:
                log_metric_safe(
                    f"{model_prefix}_top{rank}_{metric}",
                    row.get(metric, None)
                )


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
        log_param_safe(
            f"{model_prefix}_test_class_count_{group_value}",
            int(count)
        )

    log_param_safe(
        f"{model_prefix}_test_class_counts",
        counts
    )

    print(f"{model_prefix} test class counts: {counts}")


def is_private_or_raw_artifact(artifact_name, file_path):
    """
    Return True for files that should not be uploaded to a remote server by default.
    This protects raw datasets, train/test splits, and participant-level data.
    """

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

        mlflow.log_artifact(
            str(path),
            artifact_path=artifact_folder
        )
        print(f"Logged file: {path}")


def log_folder_files(
    folder_path,
    artifact_folder,
    extensions=None,
    include_keyword=None,
    exclude_keyword=None
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
            ".txt"
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

        mlflow.log_artifact(
            str(file_path),
            artifact_path=artifact_folder
        )

        print(f"Logged folder file: {file_path}")
        logged_count += 1

    print(f"Total files logged from {folder}: {logged_count}")


def log_common_run_params(model_type):
    mlflow.log_param("dataset_name", "Autism 2024")
    mlflow.log_param("experiment_name", EXPERIMENT_NAME)
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("group_mapping", "Control=0, Autism=1")
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
    configure_mlflow()

    # ------------------------------------------------------------
    # READ TOP3 FILES
    # ------------------------------------------------------------

    svm_top3 = read_top3_file(svm_files["svm_scores_top3"])
    xgb_top3 = read_top3_file(xgb_files["xgb_scores_top3"])
    rf_top3 = read_top3_file(rf_files["rf_scores_top3"])

    # ------------------------------------------------------------
    # LOG SVM RUN
    # ------------------------------------------------------------

    with mlflow.start_run(run_name="Autism_2024_SVM_RBF_All_Files"):

        log_common_run_params("SVM_RBF")

        log_top3_metrics(
            svm_top3,
            model_prefix="svm"
        )

        log_test_class_counts(
            svm_files["svm_test_file"],
            model_prefix="svm"
        )

        log_individual_files(
            svm_files,
            artifact_folder="autism_svm_files"
        )

        log_folder_files(
            folder_path=svm_folders["svm_shap_plots"],
            artifact_folder="svm_shap_plots",
            extensions=[".png", ".jpg", ".jpeg"],
            exclude_keyword="XGBoost"
        )

    print("Done. Autism SVM run logged successfully.")

    # ------------------------------------------------------------
    # LOG XGBOOST RUN
    # ------------------------------------------------------------

    with mlflow.start_run(run_name="Autism_2024_XGBoost_All_Files"):

        log_common_run_params("XGBoost")

        log_top3_metrics(
            xgb_top3,
            model_prefix="xgboost"
        )

        log_test_class_counts(
            xgb_files["xgb_test_file"],
            model_prefix="xgboost"
        )

        log_individual_files(
            xgb_files,
            artifact_folder="autism_xgboost_files"
        )

        log_folder_files(
            folder_path=xgb_folders["xgb_shap_plots"],
            artifact_folder="xgboost_shap_plots",
            extensions=[".png", ".jpg", ".jpeg"],
            include_keyword="XGBoost"
        )

    print("Done. Autism XGBoost run logged successfully.")

    # ------------------------------------------------------------
    # LOG RANDOM FOREST RUN
    # ------------------------------------------------------------

    with mlflow.start_run(run_name="Autism_2024_RandomForest_All_Files"):

        log_common_run_params("RandomForest")

        log_top3_metrics(
            rf_top3,
            model_prefix="random_forest"
        )

        log_test_class_counts(
            rf_files["rf_test_file"],
            model_prefix="random_forest"
        )

        log_individual_files(
            rf_files,
            artifact_folder="autism_random_forest_files"
        )

        log_folder_files(
            folder_path=rf_folders["rf_shap_plots"],
            artifact_folder="random_forest_shap_plots",
            extensions=[".png", ".jpg", ".jpeg"]
        )

    print("Done. Autism Random Forest run logged successfully.")
    print("Done. Autism SVM + XGBoost + Random Forest MLflow logging completed.")


if __name__ == "__main__":
    main()
