"""
data_loader.py

Core service that reads the existing Excel result files and returns structured
Python dicts for the API layer. Handles multi-sheet Excel files, column-name
variations between SVM / XGBoost / RF, and graceful fallbacks when files are
missing.
"""

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.file_paths import (
    DATASETS,
    MODEL_META,
    MODEL_ROOTS,
    get_scores_path,
    get_top3_path,
    get_sfs_path,
    get_combination_results_path,
    get_train_test_paths,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    """Convert a value to float, returning None for NaN / non-numeric."""
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> Optional[int]:
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _safe_str(val) -> Optional[str]:
    if val is None:
        return None
    s = str(val)
    return None if s.lower() == "nan" else s








def _read_excel_cached(path: Path) -> Optional[dict[str, pd.DataFrame]]:
    """Read all sheets from an Excel file. Returns {sheet_name: DataFrame}."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None
    try:
        excel = pd.ExcelFile(path)
        return {sheet: pd.read_excel(excel, sheet_name=sheet) for sheet in excel.sheet_names}
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return None








def _get_cv_col(df: pd.DataFrame) -> Optional[str]:
    """Detect the CV accuracy column name (varies between models)."""
    for col in ["CV_accuracy", "CV_Accuracy", "SFS_CV_Accuracy", "CV_Accuracy_Mean"]:
        if col in df.columns:
            return col
    return None





# ── Public API ────────────────────────────────────────────────────────────────


def _first_value(row, *columns):
    """Return the first non-empty value from a row across possible column names."""
    for column in columns:
        value = row.get(column)
        if _safe_str(value) is not None:
            return value
    return None


def _first_non_empty(*values):
    """Return the first non-empty value from a sequence of raw values."""
    for value in values:
        if _safe_str(value) is not None:
            return value
    return None


def _parse_model_config(value) -> dict:
    """Parse JSON stored in MLP Model_Config cells."""
    text = _safe_str(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}

def get_model_summaries() -> list[dict]:
    """Return summary info for all configured models."""
    summaries = []
    for model_id, meta in MODEL_META.items():
        root = MODEL_ROOTS.get(model_id)
        datasets_available = []
        for ds_id in DATASETS:
            top3 = get_top3_path(model_id, ds_id)
            if top3.exists():
                datasets_available.append(ds_id)

        summaries.append({
            **meta,
            "datasets_available": datasets_available,
        })
    return summaries


def get_top3_metrics(model_id: str, dataset_id: str) -> dict:
    """
    Read the Top-3 scores file for a model × dataset combination.
    Returns a dict with model info, dataset info, top-3 rows, and total-run count.
    """
    top3_path = get_top3_path(model_id, dataset_id)
    scores_path = get_scores_path(model_id, dataset_id)
    ds = DATASETS[dataset_id]
    meta = MODEL_META[model_id]

    result = {
        "model_id": model_id,
        "model_name": meta["name"],
        "dataset": dataset_id,
        "dataset_name": ds["name"],
        "top3": [],
        "total_runs": None,
    }

    # Count total runs from the full scores file
    all_sheets = _read_excel_cached(scores_path)
    if all_sheets:
        total = sum(len(df) for df in all_sheets.values())
        result["total_runs"] = total

    # Read Top-3 file
    sheets = _read_excel_cached(top3_path)
    if not sheets:
        return result

    # Take the first sheet (typically the primary result)
    first_sheet = list(sheets.values())[0]
    cv_col = _get_cv_col(first_sheet)

    for _, row in first_sheet.iterrows():
        model_config = _parse_model_config(row.get("Model_Config"))
        entry = {
            "name_model": _safe_str(row.get("Name_Model")),
            "num_features": _safe_int(row.get("#Features")),
            "cv_accuracy": _safe_float(row.get(cv_col)) if cv_col else None,
            "test_accuracy": _safe_float(_first_value(row, "Test_Accuracy", "Test_Accuracy_Mean")),
            "sensitivity": _safe_float(_first_value(row, "Sensitivity", "CV_Sensitivity_Mean")),
            "specificity": _safe_float(_first_value(row, "Specificity", "CV_Specificity_Mean")),
            "f1": _safe_float(_first_value(row, "F1", "F1_Score", "CV_F1_Mean")),
            "mcc": _safe_float(row.get("MCC")),
            "npv": _safe_float(row.get("NPV")),
            "ppv": _safe_float(_first_value(row, "PPV", "Precision", "CV_Precision_Mean")),
            "likelihood_ratio": _safe_float(row.get("Likelihood_Ratio")),
            "confusion_matrix": _safe_str(row.get("Confusion_Matrix")),
            "features": _safe_str(_first_value(row, "Features", "Selected_Features", "Feature_Names")),
            # SVM hyperparameters
            "c_value": _safe_float(row.get("C_Value")),
            "gamma_value": _safe_float(row.get("Gamma_Value")),
            # XGBoost hyperparameters
            "n_estimators": _safe_int(row.get("n_estimators")),
            "max_depth": _safe_int(row.get("max_depth")),
            "learning_rate": _safe_float(row.get("learning_rate")),
            "subsample": _safe_float(row.get("subsample")),
            "colsample_bytree": _safe_float(row.get("colsample_bytree")),
            # Random Forest hyperparameters
            "min_samples_split": _safe_int(row.get("min_samples_split")),
            "min_samples_leaf": _safe_int(row.get("min_samples_leaf")),
            "max_features": _safe_str(row.get("max_features")),
            # MLP hyperparameters
            "architecture": _safe_str(_first_non_empty(row.get("Architecture"), row.get("Model"), model_config.get("architecture"))),
            "hidden_layer_sizes": _safe_str(_first_non_empty(row.get("hidden_layer_sizes"), model_config.get("hidden_layers"), model_config.get("hidden_size"), row.get("Architecture"))),
            "cell_type": _safe_str(model_config.get("cell_type")),
            "hidden_size": _safe_int(model_config.get("hidden_size")),
            "num_layers": _safe_int(model_config.get("num_layers")),
            "bidirectional": model_config.get("bidirectional"),
            "activation": _safe_str(row.get("activation")),
            "solver": _safe_str(row.get("solver")),
            "alpha": _safe_float(row.get("alpha")),
            "batch_size": _safe_int(model_config.get("batch_size")),
            "dropout": _safe_float(model_config.get("dropout")),
            "learning_rate_init": _safe_float(_first_non_empty(row.get("learning_rate_init"), row.get("learning_rate"), model_config.get("learning_rate"))),
            "weight_decay": _safe_float(model_config.get("weight_decay")),
            "best_epoch_median": _safe_int(row.get("Best_Epoch_Median")),
            "best_epochs": _safe_str(row.get("Best_Epochs")),
            # LightGBM hyperparameters
            "num_leaves": _safe_int(row.get("num_leaves")),
            "min_child_samples": _safe_int(row.get("min_child_samples")),
            "min_child_weight": _safe_float(row.get("min_child_weight")),
            "reg_alpha": _safe_float(row.get("reg_alpha")),
            "reg_lambda": _safe_float(row.get("reg_lambda")),
            "boosting_type": _safe_str(row.get("boosting_type")),
            "objective": _safe_str(row.get("objective")),
            # Common
            "cv_split": _safe_int(row.get("CV_split")),
            "random_state": _safe_int(row.get("Random_State")),
        }
        result["top3"].append(entry)

    return result




def get_sfs_features(model_id: str, dataset_id: str) -> dict:
    """
    Read the SFS (Sequential Feature Selection) results and return the
    ordered feature list from the first sheet.
    """
    sfs_path = get_sfs_path(model_id, dataset_id)
    result = {
        "model_id": model_id,
        "dataset": dataset_id,
        "features": [],
        "sheet_name": None,
    }

    sheets = _read_excel_cached(sfs_path)
    if not sheets:
        return result

    first_sheet_name = list(sheets.keys())[0]
    df = sheets[first_sheet_name]
    result["sheet_name"] = first_sheet_name

    # The SFS file typically has columns for features ordered by selection
    # The last column usually contains the ordered feature list
    # Try to extract feature names from columns or from the feature list column
    if "Feature" in df.columns:
        for i, row in df.iterrows():
            result["features"].append({
                "rank": i + 1,
                "name": str(row["Feature"]),
                "score": _safe_float(row.get("Score") or row.get("CV_Accuracy") or row.get("Accuracy")),
            })
    else:
        # Fall back to using column names as features (common pattern)
        # Take the feature names from the last column which often has the ordered list
        last_col = df.columns[-1]
        try:
            from ast import literal_eval
            features_str = str(df.iloc[0][last_col])
            features_list = literal_eval(features_str)
            if isinstance(features_list, (list, tuple)):
                for i, feat in enumerate(features_list):
                    result["features"].append({
                        "rank": i + 1,
                        "name": str(feat),
                        "score": None,
                    })
        except Exception:
            # Just list the column names of the DataFrame (excluding metadata columns)
            meta_cols = {"Feature_idx", "#Features", "Confusion_Matrix", "CV_Accuracy",
                         "Test_Accuracy", "Sensitivity", "Score", "Accuracy"}
            feat_cols = [c for c in df.columns if c not in meta_cols]
            for i, col in enumerate(feat_cols[:20]):
                result["features"].append({
                    "rank": i + 1,
                    "name": col,
                    "score": None,
                })

    return result






def get_comparison_data(dataset_id: str) -> dict:
    """
    Build comparison data for all configured models on a given dataset.
    Uses the best (first) row from each model's Top-3 file.
    """
    ds = DATASETS[dataset_id]
    models = []

    for model_id, meta in MODEL_META.items():
        top3 = get_top3_path(model_id, dataset_id)
        sheets = _read_excel_cached(top3)

        entry = {
            "model_id": model_id,
            "model_name": meta["name"],
            "color": meta["color"],
            "cv_accuracy": None,
            "test_accuracy": None,
            "sensitivity": None,
            "specificity": None,
            "f1": None,
            "mcc": None,
            "npv": None,
            "ppv": None,
            "num_features": None,
            "model_run_name": None,
        }

        if sheets:
            df = list(sheets.values())[0]
            if not df.empty:
                row = df.iloc[0]
                cv_col = _get_cv_col(df)
                entry["cv_accuracy"] = _safe_float(row.get(cv_col)) if cv_col else None
                entry["test_accuracy"] = _safe_float(_first_value(row, "Test_Accuracy", "Test_Accuracy_Mean"))
                entry["sensitivity"] = _safe_float(_first_value(row, "Sensitivity", "CV_Sensitivity_Mean"))
                entry["specificity"] = _safe_float(_first_value(row, "Specificity", "CV_Specificity_Mean"))
                entry["f1"] = _safe_float(_first_value(row, "F1", "F1_Score", "CV_F1_Mean"))
                entry["mcc"] = _safe_float(row.get("MCC"))
                entry["npv"] = _safe_float(row.get("NPV"))
                entry["ppv"] = _safe_float(_first_value(row, "PPV", "Precision", "CV_Precision_Mean"))
                entry["num_features"] = _safe_int(row.get("#Features"))
                entry["model_run_name"] = _safe_str(row.get("Name_Model"))

        models.append(entry)

    return {
        "dataset": dataset_id,
        "dataset_name": ds["name"],
        "models": models,
    }





def get_dataset_info_list() -> list[dict]:
    """Return info about all datasets with which models have results available."""
    results = []
    for ds_id, ds in DATASETS.items():
        models_available = []
        for model_id in MODEL_META:
            if get_top3_path(model_id, ds_id).exists():
                models_available.append(model_id)

        results.append({
            **ds,
            "models_available": models_available,
        })
    return results




def get_experiment_files() -> list[dict]:
    """
    Scan all result directories and return a list of available experiment files
    with their metadata.
    """
    files = []
    category_map = {
        "Scores": "scores",
        "Top3": "top3_scores",
        "SFS": "sfs_results",
        "Step_2": "combination_results",
        "SHAP_data": "shap_data",
        "train": "train_data",
        "test": "test_data",
        "time_variable_stats": "time_stats",
        "JA_TS_features": "features",
        "NH_correlated": "correlated_features",
    }

    for model_id in MODEL_META:
        for ds_id in DATASETS:
            # Check various file types
            paths_to_check = [
                ("scores", get_scores_path(model_id, ds_id)),
                ("top3_scores", get_top3_path(model_id, ds_id)),
                ("sfs_results", get_sfs_path(model_id, ds_id)),
                ("combination_results", get_combination_results_path(model_id, ds_id)),
            ]

            for cat, fpath in paths_to_check:
                if fpath.exists():
                    files.append({
                        "filename": fpath.name,
                        "file_type": fpath.suffix.lstrip("."),
                        "model": model_id,
                        "dataset": ds_id,
                        "category": cat,
                        "size_bytes": fpath.stat().st_size,
                    })

    return files
