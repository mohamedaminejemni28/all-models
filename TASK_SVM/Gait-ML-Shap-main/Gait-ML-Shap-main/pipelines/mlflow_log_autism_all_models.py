from pathlib import Path
import os

import mlflow
from mlflow.exceptions import MlflowException
import pandas as pd
import numpy as np


# ============================================================
# 1) PROJECT ROOTS
# ============================================================
# You can keep the default Windows paths, or override them without changing
# the code by setting environment variables.
#
# PowerShell examples:
# $env:SVM_ROOT="C:\\Users\\k6qqj\\Desktop\\ALL_MODELS\\TASK_SVM\\Gait-ML-Shap-main\\Gait-ML-Shap-main"
# $env:XGB_ROOT="C:\\Users\\k6qqj\\Desktop\\ALL_MODELS\\Gait-ML-Shap-main_y_Mohamed_XGboost\\Gait-ML-Shap-main"
# $env:RF_ROOT="C:\\Users\\k6qqj\\Desktop\\ALL_MODELS\\Gait-ML-Shap-main_y_Mohamed -Random_Forest\\Gait-ML-Shap-main"
# $env:MLP_ROOT="C:\\Users\\k6qqj\\Desktop\\ALL_MODELS\\Gait-ML-Shap-main_y_Mohamed -MLP\\Gait-ML-Shap-main"
# $env:CATBOOST_ROOT="C:\\Users\\k6qqj\\Desktop\\ALL_MODELS\\Gait-ML-Shap-main_y_Mohamed -CatBoost\\Gait-ML-Shap-main"
# $env:LIGHTGBM_ROOT="C:\\Users\\k6qqj\\Desktop\\ALL_MODELS\\Gait-ML-Shap-main_y_Mohamed -LightGBM\\Gait-ML-Shap-main"

THIS_FILE = Path(__file__).resolve()
SVM_PROJECT_ROOT = THIS_FILE.parents[1]
ALL_MODELS_ROOT = THIS_FILE.parents[4]

SVM_ROOT = Path(
    os.getenv(
        "SVM_ROOT",
        str(SVM_PROJECT_ROOT),
    )
)

XGB_ROOT = Path(
    os.getenv(
        "XGB_ROOT",
        str(ALL_MODELS_ROOT / "Gait-ML-Shap-main_y_Mohamed_XGboost" / "Gait-ML-Shap-main"),
    )
)

RF_ROOT = Path(
    os.getenv(
        "RF_ROOT",
        str(ALL_MODELS_ROOT / "Gait-ML-Shap-main_y_Mohamed -Random_Forest" / "Gait-ML-Shap-main"),
    )
)

MLP_ROOT = Path(
    os.getenv(
        "MLP_ROOT",
        str(ALL_MODELS_ROOT / "Gait-ML-Shap-main_y_Mohamed -MLP" / "Gait-ML-Shap-main"),
    )
)

CATBOOST_ROOT = Path(
    os.getenv(
        "CATBOOST_ROOT",
        str(ALL_MODELS_ROOT / "Gait-ML-Shap-main_y_Mohamed -CatBoost" / "Gait-ML-Shap-main"),
    )
)

LIGHTGBM_ROOT = Path(
    os.getenv(
        "LIGHTGBM_ROOT",
        str(ALL_MODELS_ROOT / "Gait-ML-Shap-main_y_Mohamed -LightGBM" / "Gait-ML-Shap-main"),
    )
)

MLFLOW_DB = Path(os.getenv("MLFLOW_DB", str(SVM_ROOT / "mlflow.db")))

DATASET_PREFIX = "autism_2024"
DATASET_NAME = "Autism 2024"
GROUP_MAPPING = "Control=0, Autism=1"


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


def set_experiment_safely(base_experiment_name):
    """
    Set the MLflow experiment.

    If the experiment was deleted before, MLflow refuses to reuse the same name.
    In that case, this function automatically tries:
        Autism_v2
        Autism_v3
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
# 3) HELPER FUNCTIONS
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


def normalize_mlflow_name(name):
    """Make a safe MLflow metric/parameter name."""
    name = str(name).strip()
    for old, new in [
        (" ", "_"),
        ("#", "num"),
        ("/", "_"),
        ("\\", "_"),
        ("(", ""),
        (")", ""),
        ("%", "pct"),
        ("-", "_"),
        (":", "_"),
        (";", "_"),
        (",", "_"),
        (".", "_"),
    ]:
        name = name.replace(old, new)

    while "__" in name:
        name = name.replace("__", "_")

    return name.strip("_")


def log_param_safe(name, value):
    if value is None:
        return

    if isinstance(value, float) and not np.isfinite(value):
        return

    name = normalize_mlflow_name(name)
    value = str(value)

    if value.lower() == "nan":
        return

    if len(value) > 500:
        value = value[:500] + "..."

    mlflow.log_param(name, value)


def log_metric_safe(name, value):
    value = safe_float(value)

    if value is None:
        return

    name = normalize_mlflow_name(name)
    mlflow.log_metric(name, value)


def first_existing_path(candidate_paths):
    """Return the first existing path. If none exists, return the first candidate."""
    candidates = [Path(path) for path in candidate_paths]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


def build_standard_model_files(root, model_key, model_file_label=None, shap_folder_name=None):
    """
    Build paths for a model that stores outputs inside results/.

    model_key: short internal prefix used for artifact names, for example "mlp".
    model_file_label: label used in file names, for example "MLP" or "LightGBM".
    """
    if model_file_label is None:
        model_file_label = model_key

    if shap_folder_name is None:
        shap_folder_name = f"{DATASET_PREFIX}_SHAP_{model_file_label}"

    return {
        "files": {
            "config_current": p(root, "configs/config.yaml"),

            f"{model_key}_raw_file": p(
                root,
                "data/raw/2024 FINAL - Autism Study MAR2024 ALL (AM+NM) Features.xlsx",
            ),
            "feature_names_file": p(
                root,
                "data/raw/ISB Abstract - Joint Angles and TS Variable Names.xlsx",
            ),
            "highly_correlated_features": p(
                root,
                "data/raw/Highly correlated features.xlsx",
            ),

            f"{model_key}_phase1_JA_TS": p(
                root,
                f"results/{DATASET_PREFIX}_JA_TS_features.xlsx",
            ),
            f"{model_key}_phase2_NH_correlated": p(
                root,
                f"results/{DATASET_PREFIX}_NH_correlated_features.xlsx",
            ),
            f"{model_key}_train_file": p(
                root,
                f"results/{DATASET_PREFIX}_train.xlsx",
            ),
            f"{model_key}_test_file": p(
                root,
                f"results/{DATASET_PREFIX}_test.xlsx",
            ),

            f"{model_key}_sfs_results": p(
                root,
                f"results/{DATASET_PREFIX}_phase4_1_SFS_results_{model_file_label}.xlsx",
            ),
            f"{model_key}_step2_results": p(
                root,
                f"results/{DATASET_PREFIX}_Step_2_Results_{model_file_label}.xlsx",
            ),
            f"{model_key}_scores": p(
                root,
                f"results/{DATASET_PREFIX}_Scores_{model_file_label}.xlsx",
            ),
            f"{model_key}_scores_top3": p(
                root,
                f"results/{DATASET_PREFIX}_Scores_{model_file_label}_Top3.xlsx",
            ),
            f"{model_key}_shap_excel": first_existing_path(
                [
                    p(root, f"output-autism/{DATASET_PREFIX}_SHAP_data_{model_file_label}.xlsx"),
                    p(root, f"output-autism/{DATASET_PREFIX}_SHAP_data.xlsx"),
                    p(root, f"results/{DATASET_PREFIX}_SHAP_data_{model_file_label}.xlsx"),
                ]
            ),
        },
        "folders": {
            f"{model_key}_shap_plots": first_existing_path(
                [
                    p(root, f"output-autism/{shap_folder_name}"),
                    p(root, f"output-autism/{DATASET_PREFIX}_SHAP"),
                    p(root, f"results/{shap_folder_name}"),
                ]
            ),
            f"{model_key}_checkpoints": p(root, "results/mlp_checkpoints"),
        },
    }


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
    possible_names = [
        "CV_accuracy",
        "CV_Accuracy",
        "SFS_CV_Accuracy",
        "Mean_CV_Accuracy",
        "Best_CV_Accuracy",
    ]

    for col in possible_names:
        if col in df.columns:
            return col

    return None


def is_metric_column(column_name):
    """Detect common evaluation metric columns."""
    metric_columns = {
        "CV_accuracy",
        "CV_Accuracy",
        "SFS_CV_Accuracy",
        "Mean_CV_Accuracy",
        "Best_CV_Accuracy",
        "Test_Accuracy",
        "Train_Accuracy",
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "F1",
        "F1_Score",
        "MCC",
        "NPV",
        "PPV",
        "Likelihood_Ratio",
        "AUC",
        "ROC_AUC",
        "Precision",
        "Recall",
        "Balanced_Accuracy",
    }

    return str(column_name) in metric_columns


def is_likely_parameter_column(column_name):
    """Detect common model hyperparameter / information columns."""
    parameter_columns = {
        # common information
        "Name_Model",
        "#Features",
        "Features",
        "Ordered Indices",
        "Sheet",
        "Random_State",
        "random_state",
        "Model",
        "Model_Name",

        # SVM
        "C_Value",
        "Gamma_Value",
        "kernel",

        # XGBoost / Random Forest / LightGBM / CatBoost
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "colsample_bylevel",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "criterion",
        "num_leaves",
        "min_child_samples",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
        "boosting_type",
        "objective",
        "iterations",
        "depth",
        "l2_leaf_reg",
        "border_count",
        "loss_function",
        "eval_metric",

        # MLP
        "hidden_layer_sizes",
        "activation",
        "solver",
        "alpha",
        "batch_size",
        "learning_rate_init",
        "learning_rate_schedule",
        "max_iter",
        "early_stopping",
        "validation_fraction",
        "beta_1",
        "beta_2",
        "epsilon",
    }

    return str(column_name) in parameter_columns


def log_best_row_all_supported_columns(best_row, model_prefix):
    """
    Generic logger for the first/best row.
    It logs known metrics as metrics and known hyperparameters/information as params.
    This makes the script robust for MLP, CatBoost, LightGBM, RandomForest, XGBoost, and SVM.
    """
    for column_name, value in best_row.items():
        if pd.isna(value):
            continue

        clean_column = normalize_mlflow_name(column_name)

        if is_metric_column(column_name):
            log_metric_safe(f"{model_prefix}_best_{clean_column}", value)
        elif is_likely_parameter_column(column_name):
            log_param_safe(f"{model_prefix}_best_{clean_column}", value)


def log_top3_metrics(df_top3, model_prefix):
    if df_top3 is None or df_top3.empty:
        print(f"No Top3 data available for {model_prefix}")
        return

    cv_col = get_cv_column(df_top3)
    best_row = df_top3.iloc[0]

    # Best model: generic logging of known metrics and parameters.
    log_best_row_all_supported_columns(best_row, model_prefix)

    # Stable common names, useful for comparing all models in MLflow.
    log_param_safe(f"{model_prefix}_best_model_name", best_row.get("Name_Model", ""))
    log_param_safe(
        f"{model_prefix}_best_num_features",
        safe_int(best_row.get("#Features", 0), 0),
    )

    if cv_col is not None:
        log_metric_safe(f"{model_prefix}_best_CV_accuracy", best_row.get(cv_col, None))

    for metric in [
        "Test_Accuracy",
        "Sensitivity",
        "Specificity",
        "F1",
        "F1_Score",
        "MCC",
        "NPV",
        "PPV",
        "Likelihood_Ratio",
        "AUC",
        "ROC_AUC",
        "Precision",
        "Recall",
        "Balanced_Accuracy",
    ]:
        if metric in df_top3.columns:
            log_metric_safe(f"{model_prefix}_best_{metric}", best_row.get(metric, None))

    # Top 3 models.
    for rank, (_, row) in enumerate(df_top3.head(3).iterrows(), start=1):
        log_param_safe(f"{model_prefix}_top{rank}_model_name", row.get("Name_Model", ""))
        log_param_safe(
            f"{model_prefix}_top{rank}_num_features",
            safe_int(row.get("#Features", 0), 0),
        )

        if "Features" in df_top3.columns:
            log_param_safe(f"{model_prefix}_top{rank}_features", row.get("Features", ""))

        if "Ordered Indices" in df_top3.columns:
            log_param_safe(
                f"{model_prefix}_top{rank}_ordered_indices",
                row.get("Ordered Indices", ""),
            )

        if cv_col is not None:
            log_metric_safe(f"{model_prefix}_top{rank}_CV_accuracy", row.get(cv_col, None))

        for metric in [
            "Test_Accuracy",
            "Sensitivity",
            "Specificity",
            "F1",
            "F1_Score",
            "MCC",
            "NPV",
            "PPV",
            "AUC",
            "ROC_AUC",
            "Precision",
            "Recall",
            "Balanced_Accuracy",
        ]:
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
            ".json",
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


def log_common_run_params(model_type, active_experiment_name, root_path):
    mlflow.log_param("dataset_name", DATASET_NAME)
    mlflow.log_param("dataset_prefix", DATASET_PREFIX)
    mlflow.log_param("experiment_name", active_experiment_name)
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("project_root", str(root_path))
    mlflow.log_param("group_mapping", GROUP_MAPPING)
    mlflow.log_param("feature_selection_method", "SFS")
    mlflow.log_param("phase", "Top3_SHAP")
    mlflow.log_param("tracking_uri", mlflow.get_tracking_uri())
    mlflow.log_param("remote_tracking", IS_REMOTE_TRACKING)
    mlflow.log_param("log_artifacts", LOG_ARTIFACTS)
    mlflow.log_param("log_raw_data", LOG_RAW_DATA)

    if not IS_REMOTE_TRACKING:
        mlflow.log_param("tracking_db", str(MLFLOW_DB))


def log_model_run(model_config, active_experiment_name):
    model_key = model_config["model_key"]
    model_type = model_config["model_type"]
    root = model_config["root"]
    files = model_config["files"]
    folders = model_config.get("folders", {})
    artifact_folder = model_config["artifact_folder"]
    shap_artifact_folder = model_config["shap_artifact_folder"]
    run_name = model_config["run_name"]

    top3_key = f"{model_key}_scores_top3"
    test_key = f"{model_key}_test_file"
    shap_key = f"{model_key}_shap_plots"

    top3 = read_top3_file(files[top3_key])

    with mlflow.start_run(run_name=run_name):
        log_common_run_params(model_type, active_experiment_name, root)

        log_top3_metrics(
            top3,
            model_prefix=model_key,
        )

        log_test_class_counts(
            files[test_key],
            model_prefix=model_key,
        )

        log_individual_files(
            files,
            artifact_folder=artifact_folder,
        )

        if shap_key in folders:
            log_folder_files(
                folder_path=folders[shap_key],
                artifact_folder=shap_artifact_folder,
                extensions=[".png", ".jpg", ".jpeg"],
            )

        # Optional: log MLP checkpoints if they exist.
        checkpoint_key = f"{model_key}_checkpoints"
        if checkpoint_key in folders:
            log_folder_files(
                folder_path=folders[checkpoint_key],
                artifact_folder=f"{model_key}_checkpoints",
                extensions=[".pt", ".pth", ".pkl", ".joblib", ".json", ".txt", ".png"],
            )

    print(f"Done. {DATASET_NAME} {model_type} run logged successfully.")


# ============================================================
# 4) MODEL FILE CONFIGS - AUTISM
# ============================================================

# -----------------------------
# SVM files are kept close to your original script because your SVM project
# stores several outputs outside the results/ folder.
# -----------------------------
svm_files = {
    "config_current": p(SVM_ROOT, "configs/config.yaml"),

    "svm_raw_file": p(
        SVM_ROOT,
        "data/raw/2024 FINAL - Autism Study MAR2024 ALL (AM+NM) Features.xlsx",
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
        "data/processed/autism_2024_JA_TS_features.xlsx",
    ),
    "svm_phase2_NH_correlated": p(
        SVM_ROOT,
        "data/processed/autism_2024_NH_correlated_features.xlsx",
    ),
    "svm_train_file": p(
        SVM_ROOT,
        "data/processed/autism_2024_train.xlsx",
    ),
    "svm_test_file": p(
        SVM_ROOT,
        "data/processed/autism_2024_test.xlsx",
    ),

    "svm_sfs_results": p(
        SVM_ROOT,
        "src/gaitml/features/autism_2024_phase4_1_SFS_results.xlsx",
    ),
    "svm_step2_results": first_existing_path(
        [
            p(SVM_ROOT, "autism_2024_Step_2_Results_Filtered.xlsx"),
            p(SVM_ROOT, "results/autism_2024_Step_2_Results_RBF.xlsx"),
        ]
    ),
    "svm_scores": p(
        SVM_ROOT,
        "autism_2024_Scores_RBF.xlsx",
    ),
    "svm_scores_top3": p(
        SVM_ROOT,
        "autism_2024_Scores_RBF_Top3.xlsx",
    ),
    "svm_shap_excel": p(
        SVM_ROOT,
        "output-autism/autism_2024_SHAP_data.xlsx",
    ),
}

svm_folders = {
    "svm_shap_plots": p(
        SVM_ROOT,
        "output-autism/autism_2024_SHAP",
    ),
}

# -----------------------------
# XGBoost
# -----------------------------
xgb_config = build_standard_model_files(
    root=XGB_ROOT,
    model_key="xgboost",
    model_file_label="XGBoost",
    shap_folder_name="autism_2024_SHAP",
)

# -----------------------------
# Random Forest
# -----------------------------
rf_config = build_standard_model_files(
    root=RF_ROOT,
    model_key="random_forest",
    model_file_label="RandomForest",
    shap_folder_name="autism_2024_SHAP_RandomForest",
)

# -----------------------------
# MLP
# According to your results folder:
# autism_2024_phase4_1_SFS_results_MLP.xlsx
# autism_2024_Step_2_Results_MLP.xlsx
# autism_2024_Scores_MLP.xlsx
# autism_2024_Scores_MLP_Top3.xlsx
# -----------------------------
mlp_config = build_standard_model_files(
    root=MLP_ROOT,
    model_key="mlp",
    model_file_label="MLP",
    shap_folder_name="autism_2024_SHAP_MLP",
)

# -----------------------------
# CatBoost
# Expected names:
# autism_2024_phase4_1_SFS_results_CatBoost.xlsx
# autism_2024_Step_2_Results_CatBoost.xlsx
# autism_2024_Scores_CatBoost.xlsx
# autism_2024_Scores_CatBoost_Top3.xlsx
# -----------------------------
catboost_config = build_standard_model_files(
    root=CATBOOST_ROOT,
    model_key="catboost",
    model_file_label="CatBoost",
    shap_folder_name="autism_2024_SHAP_CatBoost",
)

# -----------------------------
# LightGBM
# According to your results folder:
# autism_2024_phase4_1_SFS_results_LightGBM.xlsx
# autism_2024_Step_2_Results_LightGBM.xlsx
# autism_2024_Scores_LightGBM.xlsx
# autism_2024_Scores_LightGBM_Top3.xlsx
# -----------------------------
lightgbm_config = build_standard_model_files(
    root=LIGHTGBM_ROOT,
    model_key="lightgbm",
    model_file_label="LightGBM",
    shap_folder_name="autism_2024_SHAP_LightGBM",
)


MODEL_CONFIGS = [
    {
        "model_key": "svm",
        "model_type": "SVM_RBF",
        "root": SVM_ROOT,
        "files": svm_files,
        "folders": svm_folders,
        "artifact_folder": "autism_svm_files",
        "shap_artifact_folder": "svm_shap_plots",
        "run_name": "Autism_2024_SVM_RBF_All_Files",
    },
    {
        "model_key": "xgboost",
        "model_type": "XGBoost",
        "root": XGB_ROOT,
        "files": xgb_config["files"],
        "folders": xgb_config["folders"],
        "artifact_folder": "autism_xgboost_files",
        "shap_artifact_folder": "xgboost_shap_plots",
        "run_name": "Autism_2024_XGBoost_All_Files",
    },
    {
        "model_key": "random_forest",
        "model_type": "RandomForest",
        "root": RF_ROOT,
        "files": rf_config["files"],
        "folders": rf_config["folders"],
        "artifact_folder": "autism_random_forest_files",
        "shap_artifact_folder": "random_forest_shap_plots",
        "run_name": "Autism_2024_RandomForest_All_Files",
    },
    {
        "model_key": "mlp",
        "model_type": "MLP",
        "root": MLP_ROOT,
        "files": mlp_config["files"],
        "folders": mlp_config["folders"],
        "artifact_folder": "autism_mlp_files",
        "shap_artifact_folder": "mlp_shap_plots",
        "run_name": "Autism_2024_MLP_All_Files",
    },
    {
        "model_key": "catboost",
        "model_type": "CatBoost",
        "root": CATBOOST_ROOT,
        "files": catboost_config["files"],
        "folders": catboost_config["folders"],
        "artifact_folder": "autism_catboost_files",
        "shap_artifact_folder": "catboost_shap_plots",
        "run_name": "Autism_2024_CatBoost_All_Files",
    },
    {
        "model_key": "lightgbm",
        "model_type": "LightGBM",
        "root": LIGHTGBM_ROOT,
        "files": lightgbm_config["files"],
        "folders": lightgbm_config["folders"],
        "artifact_folder": "autism_lightgbm_files",
        "shap_artifact_folder": "lightgbm_shap_plots",
        "run_name": "Autism_2024_LightGBM_All_Files",
    },
]


# ============================================================
# 5) MAIN LOGGING PIPELINE
# ============================================================

def main():
    active_experiment_name = configure_mlflow()

    for model_config in MODEL_CONFIGS:
        print("\n======================================================")
        print(f"Starting MLflow logging for: {model_config['model_type']}")
        print(f"Root: {model_config['root']}")
        print("======================================================\n")

        log_model_run(
            model_config=model_config,
            active_experiment_name=active_experiment_name,
        )

    print("\nDone. Autism SVM + XGBoost + Random Forest + MLP + CatBoost + LightGBM MLflow logging completed.")


if __name__ == "__main__":
    main()
