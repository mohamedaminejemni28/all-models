from pathlib import Path

import mlflow
import pandas as pd
import numpy as np


# ============================================================
# 1) ROOTS
# ============================================================

MLFLOW_ROOT = Path(
    r"C:\Users\k6qqj\Desktop\Gait-ML-Shap-main_y_Mohamed -Random_Forest\Gait-ML-Shap-main"
)

RF_ROOT = MLFLOW_ROOT


def p(root, relative_path):
    return root / relative_path


# ============================================================
# 2) MLFLOW SETUP
# ============================================================

mlflow_db_path = MLFLOW_ROOT / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path.as_posix()}")
mlflow.set_experiment("Autism")


# ============================================================
# 3) RANDOM FOREST FILES
# ============================================================

rf_files = {
    "config_current": p(RF_ROOT, "configs/config.yaml"),

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
    "random_forest_shap_plots": p(
        RF_ROOT,
        "output-autism/autism_2024_SHAP_RandomForest"
    ),
}


# ============================================================
# 4) HELPERS
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

    print(f"Loaded Top3 file: {path}")
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

    log_param_safe(f"{model_prefix}_best_model_name", best_row.get("Name_Model", ""))
    log_param_safe(f"{model_prefix}_best_num_features", safe_int(best_row.get("#Features", 0), 0))

    if "n_estimators" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_n_estimators", safe_int(best_row.get("n_estimators", None)))

    if "max_depth" in df_top3.columns:
        log_param_safe(f"{model_prefix}_best_max_depth", safe_int(best_row.get("max_depth", None)))

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

    for metric in [
        "Test_Accuracy",
        "Sensitivity",
        "Specificity",
        "F1",
        "MCC",
        "NPV",
        "PPV",
        "Likelihood_Ratio"
    ]:
        if metric in df_top3.columns:
            log_metric_safe(f"{model_prefix}_best_{metric}", best_row.get(metric, None))

    for rank, (_, row) in enumerate(df_top3.head(3).iterrows(), start=1):
        log_param_safe(f"{model_prefix}_top{rank}_model_name", row.get("Name_Model", ""))
        log_param_safe(f"{model_prefix}_top{rank}_num_features", safe_int(row.get("#Features", 0), 0))

        if cv_col is not None:
            log_metric_safe(f"{model_prefix}_top{rank}_CV_accuracy", row.get(cv_col, None))

        for metric in [
            "Test_Accuracy",
            "Sensitivity",
            "Specificity",
            "F1",
            "MCC"
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


def log_individual_files(files_dict, artifact_folder):
    for artifact_name, file_path in files_dict.items():
        path = Path(file_path)

        if path.exists():
            mlflow.log_artifact(str(path), artifact_path=artifact_folder)
            print(f"Logged file: {path}")
        else:
            print(f"Skipped missing file: {path}")


def log_folder_files(folder_path, artifact_folder, extensions=None):
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".xlsx", ".xls", ".csv", ".docx", ".txt"]

    logged_count = 0

    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in extensions:
            continue

        mlflow.log_artifact(str(file_path), artifact_path=artifact_folder)

        print(f"Logged folder file: {file_path}")
        logged_count += 1

    print(f"Total files logged from {folder}: {logged_count}")


# ============================================================
# 5) READ TOP3
# ============================================================

rf_top3 = read_top3_file(rf_files["rf_scores_top3"])


# ============================================================
# 6) LOG RANDOM FOREST RUN
# ============================================================

with mlflow.start_run(run_name="Autism_2024_RandomForest_All_Files"):

    mlflow.log_param("dataset_name", "Autism 2024")
    mlflow.log_param("experiment_name", "Autism")
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("group_mapping", "Control=0, Autism=1")
    mlflow.log_param("feature_selection_method", "SFS")
    mlflow.log_param("phase", "Top3_SHAP")
    mlflow.log_param("shared_tracking_uri", str(mlflow_db_path))

    log_top3_metrics(rf_top3, model_prefix="random_forest")
    log_test_class_counts(rf_files["rf_test_file"], model_prefix="random_forest")

    log_individual_files(
        rf_files,
        artifact_folder="autism_random_forest_files"
    )

    log_folder_files(
        folder_path=rf_folders["random_forest_shap_plots"],
        artifact_folder="random_forest_shap_plots",
        extensions=[".png", ".jpg", ".jpeg"]
    )

print("Done. Autism Random Forest run logged successfully.")