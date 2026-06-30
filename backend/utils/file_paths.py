"""
Central path configuration for model result directories and generated files.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODEL_ROOTS = {
    "svm": PROJECT_ROOT / "TASK_SVM" / "Gait-ML-Shap-main" / "Gait-ML-Shap-main",
    "xgboost": PROJECT_ROOT / "Gait-ML-Shap-main_y_Mohamed_XGboost" / "Gait-ML-Shap-main",
    "random_forest": PROJECT_ROOT / "Gait-ML-Shap-main_y_Mohamed -Random_Forest" / "Gait-ML-Shap-main",
    "mlp": PROJECT_ROOT / "Gait-ML-Shap-main_y_Mohamed -MLP" / "Gait-ML-Shap-main",
    "lightgbm": PROJECT_ROOT / "Gait-ML-Shap-main_y_Mohamed -LightGBM" / "Gait-ML-Shap-main",
}

DATASETS = {
    "autism": {
        "id": "autism",
        "name": "Autism Study 2024",
        "prefix": "autism_2024",
        "classes": {"0": "Control", "1": "Autism"},
        "description": "Classification of Control vs Autism groups from gait biomechanics data.",
    },
    "young_old": {
        "id": "young_old",
        "name": "Young vs Older Adults 2024",
        "prefix": "young_old_2024",
        "classes": {"0": "Young", "1": "Older"},
        "description": "Classification of Young vs Older adult groups from gait biomechanics data.",
    },
    "flatfoot": {
        "id": "flatfoot",
        "name": "Flatfoot Control (Older) 2024",
        "prefix": "flatfoot_control_older_2024",
        "classes": {"0": "Control", "1": "Flatfoot"},
        "description": "Classification of Control vs Flatfoot (Pes Planus) in older adults from gait biomechanics data.",
    },
}

MODEL_META = {
    "svm": {
        "id": "svm",
        "name": "SVM (RBF Kernel)",
        "short_name": "SVM",
        "description": "Support Vector Machine with an RBF kernel and Sequential Feature Selection.",
        "color": "#7c3aed",
        "icon": "SVM",
        "hyperparameters": ["C_Value", "Gamma_Value"],
    },
    "xgboost": {
        "id": "xgboost",
        "name": "XGBoost",
        "short_name": "XGBoost",
        "description": "Extreme Gradient Boosting classifier built from boosted decision trees.",
        "color": "#d97706",
        "icon": "XGB",
        "hyperparameters": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"],
    },
    "random_forest": {
        "id": "random_forest",
        "name": "Random Forest",
        "short_name": "RF",
        "description": "Bagged decision-tree ensemble with random feature subsets.",
        "color": "#059669",
        "icon": "RF",
        "hyperparameters": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"],
    },
    "mlp": {
        "id": "mlp",
        "name": "Multilayer Perceptron",
        "short_name": "MLP",
        "description": "Neural-network classifier using dense hidden layers and nonlinear activations.",
        "color": "#0891b2",
        "icon": "MLP",
        "hyperparameters": ["hidden_layer_sizes", "activation", "solver", "alpha", "learning_rate_init"],
    },
    "lightgbm": {
        "id": "lightgbm",
        "name": "LightGBM",
        "short_name": "LightGBM",
        "description": "Gradient boosting classifier optimized with leaf-wise tree growth.",
        "color": "#e11d48",
        "icon": "LGBM",
        "hyperparameters": ["n_estimators", "num_leaves", "max_depth", "learning_rate", "min_child_samples"],
    },
}

MODEL_FILE_LABELS = {
    "svm": "RBF",
    "xgboost": "XGBoost",
    "random_forest": "RandomForest",
    "mlp": "MLP",
    "lightgbm": "LightGBM",
}

SHAP_FOLDER_SUFFIXES = {
    "svm": "SHAP",
    "xgboost": "SHAP",
    "random_forest": "SHAP_RandomForest",
    "mlp": "SHAP_MLP",
    "lightgbm": "SHAP_LightGBM",
}

OUTPUT_DIRS = {
    "autism": ["output-autism"],
    "young_old": ["output-young_old", "output-young-old"],
    "flatfoot": ["output-flatfoot"],
}


def first_existing_path(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0] if paths else Path()


def _root(model_id: str) -> Path:
    return MODEL_ROOTS[model_id]


def _prefix(dataset_id: str) -> str:
    return DATASETS[dataset_id]["prefix"]


def _label(model_id: str) -> str:
    return MODEL_FILE_LABELS[model_id]


def _output_dirs(dataset_id: str) -> list[str]:
    return OUTPUT_DIRS.get(dataset_id, [])


def get_scores_path(model_id: str, dataset_id: str) -> Path:
    prefix = _prefix(dataset_id)
    root = _root(model_id)
    label = _label(model_id)

    if model_id == "svm":
        return root / f"{prefix}_Scores_RBF.xlsx"
    return root / "results" / f"{prefix}_Scores_{label}.xlsx"


def get_top3_path(model_id: str, dataset_id: str) -> Path:
    prefix = _prefix(dataset_id)
    root = _root(model_id)
    label = _label(model_id)

    if model_id == "svm":
        return root / f"{prefix}_Scores_RBF_Top3.xlsx"
    return root / "results" / f"{prefix}_Scores_{label}_Top3.xlsx"


def get_sfs_path(model_id: str, dataset_id: str) -> Path:
    prefix = _prefix(dataset_id)
    root = _root(model_id)
    label = _label(model_id)

    if model_id == "svm":
        return root / "src" / "gaitml" / "features" / f"{prefix}_phase4_1_SFS_results.xlsx"
    return root / "results" / f"{prefix}_phase4_1_SFS_results_{label}.xlsx"


def get_combination_results_path(model_id: str, dataset_id: str) -> Path:
    prefix = _prefix(dataset_id)
    root = _root(model_id)
    label = _label(model_id)

    if model_id == "svm":
        return root / f"{prefix}_Step_2_Results_Filtered.xlsx"
    return root / "results" / f"{prefix}_Step_2_Results_{label}.xlsx"


def get_shap_data_path(model_id: str, dataset_id: str) -> Path:
    prefix = _prefix(dataset_id)
    root = _root(model_id)
    label = _label(model_id)
    candidates = []

    for output_dir in _output_dirs(dataset_id):
        if model_id == "svm":
            candidates.append(root / output_dir / f"{prefix}_SHAP_data.xlsx")
        else:
            candidates.append(root / output_dir / f"{prefix}_SHAP_data_{label}.xlsx")
            candidates.append(root / output_dir / f"{prefix}_SHAP_data.xlsx")

    if model_id != "svm":
        candidates.append(root / "results" / f"{prefix}_SHAP_data_{label}.xlsx")

    return first_existing_path(candidates)


def get_shap_figures_dir(model_id: str, dataset_id: str) -> Path:
    prefix = _prefix(dataset_id)
    root = _root(model_id)
    suffix = SHAP_FOLDER_SUFFIXES[model_id]
    folder_name = f"{prefix}_{suffix}"

    candidates = []
    for output_dir in _output_dirs(dataset_id):
        candidates.append(root / output_dir / folder_name)
        candidates.append(root / output_dir / f"{prefix}_SHAP")

    candidates.append(root / "results" / folder_name)
    return first_existing_path(candidates)


def get_time_variable_plots_dir(model_id: str, dataset_id: str) -> Path:
    prefix = _prefix(dataset_id)
    root = _root(model_id)

    if model_id == "svm":
        return root / f"{prefix}_SVM_time_variable_plots"
    if model_id == "xgboost":
        return root / f"{prefix}_XGBoost_time_variable_plots"
    return Path()


def get_time_variable_stats_path(model_id: str, dataset_id: str) -> Path:
    prefix = _prefix(dataset_id)
    root = _root(model_id)

    if model_id == "svm":
        return root / f"{prefix}_SVM_time_variable_stats.xlsx"
    if model_id == "xgboost":
        return root / f"{prefix}_XGBoost_time_variable_stats.xlsx"
    return Path()


def get_train_test_paths(model_id: str, dataset_id: str) -> dict:
    prefix = _prefix(dataset_id)
    root = _root(model_id)

    if model_id == "svm":
        return {
            "train": root / "data" / "processed" / f"{prefix}_train.xlsx",
            "test": root / "data" / "processed" / f"{prefix}_test.xlsx",
        }
    return {
        "train": root / "results" / f"{prefix}_train.xlsx",
        "test": root / "results" / f"{prefix}_test.xlsx",
    }