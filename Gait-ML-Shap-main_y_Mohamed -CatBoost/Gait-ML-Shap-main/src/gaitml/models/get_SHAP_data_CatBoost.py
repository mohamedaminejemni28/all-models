"""
get_SHAP_data_CatBoost.py

CatBoost SHAP Data Analysis Utilities.

Inputs:
- CatBoost Top3 scores file
- Train dataset
- Test dataset

Outputs:
- Excel file with sample-level SHAP values
- Excel sheets with global SHAP feature ranking
- PNG SHAP summary images for each model
"""

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

from gaitml import logger


# ============================================================
# HELPERS
# ============================================================

def parse_list(value):
    """
    Parse lists saved in Excel.

    Supports:
        "['feature1', 'feature2']"
        '["feature1", "feature2"]'
        "[1, 5, 10]"
        already-existing Python lists
    """
    if isinstance(value, list):
        return value

    if isinstance(value, np.ndarray):
        return value.tolist()

    if pd.isna(value):
        return []

    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return []

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except Exception:
            pass

        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except Exception:
            pass

        return [value]

    return [value]


def get_numeric_feature_names(df, labels_column="Group", participant_column="Participant"):
    """
    Get numeric biomechanical feature columns.
    Excludes label and participant columns.
    """
    drop_cols = [labels_column]

    if participant_column in df.columns:
        drop_cols.append(participant_column)

    feature_df = df.drop(columns=drop_cols, errors="ignore")
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

    return numeric_cols


def get_cv_column(df_scores):
    """
    Handle different CV accuracy column names.
    """
    if "CV_Accuracy" in df_scores.columns:
        return "CV_Accuracy"

    if "CV_accuracy" in df_scores.columns:
        return "CV_accuracy"

    raise ValueError(
        "No CV accuracy column found. Expected 'CV_Accuracy' or 'CV_accuracy'. "
        f"Available columns are: {df_scores.columns.tolist()}"
    )


def get_selected_features_from_row(row, all_feature_names):
    """
    Extract selected feature names from one row of the CatBoost scores file.

    Priority:
        1. Feature_Names
        2. Features
        3. Selected_Indices / Feature_idx if indices are available
    """
    name_columns = [
        "Feature_Names",
        "Features",
        "Selected_Features",
        "Selected_Feature_Names",
    ]

    index_columns = [
        "Selected_Indices",
        "Ordered Indices",
        "Feature_idx",
    ]

    for col in name_columns:
        if col in row.index:
            values = parse_list(row[col])

            selected_names = []
            for value in values:
                name = str(value).strip()

                if name in all_feature_names:
                    selected_names.append(name)

            if len(selected_names) > 0:
                return selected_names

    for col in index_columns:
        if col in row.index:
            values = parse_list(row[col])

            selected_names = []
            for value in values:
                try:
                    index = int(value)
                    if 0 <= index < len(all_feature_names):
                        selected_names.append(all_feature_names[index])
                except Exception:
                    continue

            if len(selected_names) > 0:
                return selected_names

    return []


def prepare_labels(train_df, test_df, labels_column):
    """
    Encode labels safely using train + test together.

    Works with:
        Control / Flatfoot
        Control / Autism
        Young / Older
        0 / 1
    """
    if labels_column not in train_df.columns:
        raise ValueError(f"Column '{labels_column}' not found in train dataset.")

    if labels_column not in test_df.columns:
        raise ValueError(f"Column '{labels_column}' not found in test dataset.")

    label_encoder = LabelEncoder()

    train_labels = train_df[labels_column].astype(str).str.strip()
    test_labels = test_df[labels_column].astype(str).str.strip()

    all_labels = pd.concat([train_labels, test_labels], axis=0)

    label_encoder.fit(all_labels)

    y_train = label_encoder.transform(train_labels)
    y_test = label_encoder.transform(test_labels)

    logger.info(f"Label classes: {dict(enumerate(label_encoder.classes_))}")

    return y_train, y_test, label_encoder


def get_base_sheet_name(score_sheet_name, train_sheet_names):
    """
    Map score sheet names like:
        Sheet1
        Sheet1_Top3Overall
        Sheet1_Top3_5F

    back to the dataset sheet:
        Sheet1
    """
    if score_sheet_name in train_sheet_names:
        return score_sheet_name

    if "_Top3Overall" in score_sheet_name:
        candidate = score_sheet_name.split("_Top3Overall")[0]
        if candidate in train_sheet_names:
            return candidate

    if "_Top3_" in score_sheet_name:
        candidate = score_sheet_name.split("_Top3_")[0]
        if candidate in train_sheet_names:
            return candidate

    # fallback for common single-sheet files
    if len(train_sheet_names) == 1:
        return train_sheet_names[0]

    raise ValueError(
        f"Could not map score sheet '{score_sheet_name}' to train/test sheet. "
        f"Available train sheets: {train_sheet_names}"
    )


# ============================================================
# SHAP IMAGE
# ============================================================

def create_shap_bar_image(df_ranking, model_name, output_dir):
    """
    Create a horizontal bar image from global mean absolute SHAP ranking.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_plot = df_ranking.sort_values(
        by="Mean_ABS_SHAP",
        ascending=False,
    ).head(50)

    df_plot = df_plot.iloc[::-1]

    plt.figure(figsize=(10, max(5, len(df_plot) * 0.35)))
    plt.barh(df_plot["Feature"], df_plot["Mean_ABS_SHAP"])
    plt.xlabel("mean(|SHAP value|)")
    plt.ylabel("Feature")
    plt.title(f"CatBoost SHAP Summary - Model {model_name}")
    plt.tight_layout()

    safe_model_name = (
        str(model_name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace("[", "_")
        .replace("]", "_")
    )

    output_path = output_dir / f"SHAP_CatBoost_{safe_model_name}_summary.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved SHAP image: {output_path}")


# ============================================================
# CATBOOST + SHAP
# ============================================================

def normalize_shap_values(shap_values):
    """
    Convert CatBoost SHAP output to a 2D array:
        samples x features
    """
    if isinstance(shap_values, list):
        # Binary classifiers can sometimes return a list.
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    shap_values = np.array(shap_values)

    if shap_values.ndim == 2:
        return shap_values

    if shap_values.ndim == 3:
        # samples x features x classes
        return np.mean(shap_values, axis=2)

    raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")


def train_catboost_and_calculate_shap(
    X_train,
    y_train,
    X_explain,
    selected_feature_names,
    model_params,
):
    """
    Train one CatBoost model and calculate SHAP values.
    """
    model = CatBoostClassifier(
        iterations=int(model_params["iterations"]),
        depth=int(model_params["depth"]),
        learning_rate=float(model_params["learning_rate"]),
        l2_leaf_reg=float(model_params.get("l2_leaf_reg", 3)),
        bagging_temperature=float(model_params.get("bagging_temperature", 0.0)),
        random_strength=float(model_params.get("random_strength", 0.0)),
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=int(model_params.get("random_state", 42)),
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )

    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    shap_array = normalize_shap_values(shap_values)

    return shap_array, selected_feature_names


def create_shap_values_dataframe(shap_values, feature_names, metadata_df):
    """
    Create sample-level SHAP values DataFrame.
    """
    df_shap = pd.DataFrame(shap_values, columns=feature_names)

    metadata_cols = [
        col for col in ["Participant", "Index", "Side", "Group"]
        if col in metadata_df.columns
    ]

    if metadata_cols:
        df_shap = pd.concat(
            [metadata_df[metadata_cols].reset_index(drop=True), df_shap],
            axis=1,
        )

    df_shap["SUM"] = df_shap[feature_names].sum(axis=1)

    min_cols = df_shap[feature_names].idxmin(axis=1)
    max_cols = df_shap[feature_names].idxmax(axis=1)

    if "Group" in df_shap.columns:
        df_shap["Most Contributing Feat"] = np.where(
            df_shap["Group"].astype(str).isin(["0", "Control", "Young"]),
            min_cols,
            max_cols,
        )
    else:
        df_shap["Most Contributing Feat"] = max_cols

    df_shap["Most Contributing Val"] = df_shap.apply(
        lambda row: row[row["Most Contributing Feat"]],
        axis=1,
    )

    return df_shap


def create_global_ranking_dataframe(shap_values, feature_names):
    """
    Create global feature importance ranking using mean absolute SHAP.
    """
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    df_ranking = pd.DataFrame({
        "Feature": feature_names,
        "Mean_ABS_SHAP": mean_abs_shap,
    })

    df_ranking = df_ranking.sort_values(
        by="Mean_ABS_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    df_ranking["SHAP_Rank"] = df_ranking.index + 1

    return df_ranking


# ============================================================
# MAIN FUNCTION CALLED BY PIPELINE
# ============================================================

def get_SHAP_data_CatBoost(
    scores_file,
    train_file,
    test_file,
    output_file,
    output_dir,
    labels_column="Group",
    participant_column="Participant",
):
    """
    Main function to perform CatBoost SHAP analysis.
    """
    logger.info("Loading CatBoost scores file...")

    scores_excel = pd.ExcelFile(scores_file)
    train_excel = pd.ExcelFile(train_file)
    test_excel = pd.ExcelFile(test_file)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    shap_value_sheets = {}
    ranking_sheets = {}

    for score_sheet_name in scores_excel.sheet_names:
        logger.info(f"Processing score sheet: {score_sheet_name}")

        df_scores = pd.read_excel(scores_file, sheet_name=score_sheet_name)

        if df_scores.empty:
            logger.warning(f"No scores found for sheet: {score_sheet_name}")
            continue

        base_sheet_name = get_base_sheet_name(
            score_sheet_name=score_sheet_name,
            train_sheet_names=train_excel.sheet_names,
        )

        if base_sheet_name not in test_excel.sheet_names:
            raise ValueError(
                f"Sheet '{base_sheet_name}' found in train but not in test file. "
                f"Test sheets: {test_excel.sheet_names}"
            )

        logger.info(f"Using dataset sheet: {base_sheet_name}")
        logger.info(f"Columns found in scores file: {df_scores.columns.tolist()}")

        cv_col = get_cv_column(df_scores)

        df_train = pd.read_excel(train_file, sheet_name=base_sheet_name)
        df_test = pd.read_excel(test_file, sheet_name=base_sheet_name)

        y_train, y_test, label_encoder = prepare_labels(
            train_df=df_train,
            test_df=df_test,
            labels_column=labels_column,
        )

        df_all = pd.concat([df_train, df_test], ignore_index=True)

        all_feature_names = get_numeric_feature_names(
            df=df_train,
            labels_column=labels_column,
            participant_column=participant_column,
        )

        if len(all_feature_names) == 0:
            raise ValueError(f"No numeric feature columns found in sheet {base_sheet_name}.")

        required_columns = [
            "Name_Model",
            "#Features",
            "iterations",
            "depth",
            "learning_rate",
            "Test_Accuracy",
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df_scores.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns in scores file: {missing_columns}. "
                f"Available columns are: {df_scores.columns.tolist()}"
            )

        for _, row in df_scores.iterrows():
            model_name = row["Name_Model"]

            selected_feature_names = get_selected_features_from_row(
                row=row,
                all_feature_names=all_feature_names,
            )

            if len(selected_feature_names) == 0:
                logger.warning(
                    f"Skipping model {model_name} because selected features are empty."
                )
                continue

            missing = [
                feature for feature in selected_feature_names
                if feature not in df_train.columns
            ]

            if missing:
                logger.warning(
                    f"Skipping model {model_name}. Missing features: {missing}"
                )
                continue

            X_train_selected = df_train[selected_feature_names].apply(
                pd.to_numeric,
                errors="coerce",
            )

            X_explain_selected = df_all[selected_feature_names].apply(
                pd.to_numeric,
                errors="coerce",
            )

            X_train_selected = X_train_selected.fillna(X_train_selected.median())
            X_explain_selected = X_explain_selected.fillna(X_train_selected.median())

            X_train_selected = X_train_selected.to_numpy(dtype=np.float32)
            X_explain_selected = X_explain_selected.to_numpy(dtype=np.float32)

            model_params = {
                "iterations": row["iterations"],
                "depth": row["depth"],
                "learning_rate": row["learning_rate"],
                "l2_leaf_reg": row["l2_leaf_reg"] if "l2_leaf_reg" in df_scores.columns else 3,
                "bagging_temperature": row["bagging_temperature"] if "bagging_temperature" in df_scores.columns else 0.0,
                "random_strength": row["random_strength"] if "random_strength" in df_scores.columns else 0.0,
                "random_state": row["Random_State"] if "Random_State" in df_scores.columns else 42,
            }

            shap_values, selected_feature_names = train_catboost_and_calculate_shap(
                X_train=X_train_selected,
                y_train=y_train,
                X_explain=X_explain_selected,
                selected_feature_names=selected_feature_names,
                model_params=model_params,
            )

            logger.info(
                f"SHAP completed for {base_sheet_name} | Model {model_name} | "
                f"Selected {len(selected_feature_names)} features"
            )

            df_shap_values = create_shap_values_dataframe(
                shap_values=shap_values,
                feature_names=selected_feature_names,
                metadata_df=df_all,
            )

            df_ranking = create_global_ranking_dataframe(
                shap_values=shap_values,
                feature_names=selected_feature_names,
            )

            df_ranking["Name_Model"] = model_name
            df_ranking["CV_Accuracy"] = row[cv_col]
            df_ranking["Test_Accuracy"] = row["Test_Accuracy"]
            df_ranking["#Features"] = row["#Features"]
            df_ranking["iterations"] = row["iterations"]
            df_ranking["depth"] = row["depth"]
            df_ranking["learning_rate"] = row["learning_rate"]

            if "l2_leaf_reg" in df_scores.columns:
                df_ranking["l2_leaf_reg"] = row["l2_leaf_reg"]

            if "bagging_temperature" in df_scores.columns:
                df_ranking["bagging_temperature"] = row["bagging_temperature"]

            if "random_strength" in df_scores.columns:
                df_ranking["random_strength"] = row["random_strength"]

            if "Sensitivity" in df_scores.columns:
                df_ranking["Sensitivity"] = row["Sensitivity"]

            if "Specificity" in df_scores.columns:
                df_ranking["Specificity"] = row["Specificity"]

            if "F1" in df_scores.columns:
                df_ranking["F1"] = row["F1"]

            if "MCC" in df_scores.columns:
                df_ranking["MCC"] = row["MCC"]

            df_ranking["Selected_Features"] = json.dumps(selected_feature_names)

            create_shap_bar_image(
                df_ranking=df_ranking,
                model_name=model_name,
                output_dir=output_dir,
            )

            shap_sheet_name = f"SHAP_{base_sheet_name}_{model_name}"[:31]
            rank_sheet_name = f"Rank_{base_sheet_name}_{model_name}"[:31]

            shap_value_sheets[shap_sheet_name] = df_shap_values
            ranking_sheets[rank_sheet_name] = df_ranking

    if not shap_value_sheets and not ranking_sheets:
        raise ValueError("No CatBoost SHAP results were generated. Check input files.")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df in shap_value_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        for sheet_name, df in ranking_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Saved CatBoost SHAP Excel results to: {output_file}")
    logger.info(f"Saved CatBoost SHAP images to: {output_dir}")
