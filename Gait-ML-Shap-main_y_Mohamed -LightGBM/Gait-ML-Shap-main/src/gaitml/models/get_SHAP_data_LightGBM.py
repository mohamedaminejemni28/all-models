"""
get_SHAP_data_LightGBM.py

LightGBM SHAP Data Analysis Utilities.

Inputs:
- LightGBM Top3 scores file
- Train dataset
- Test dataset

Outputs:
- Excel file with sample-level SHAP values
- Excel sheets with global SHAP feature ranking
- PNG SHAP summary images for each model
"""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from lightgbm import LGBMClassifier

from gaitml import logger


# ============================================================
# HELPERS
# ============================================================

def make_safe_name(value):
    """
    Create a safe string for file names and Excel sheet names.
    """
    value = str(value)
    value = value.replace("/", "_").replace("\\", "_")
    value = value.replace(":", "_").replace("*", "_")
    value = value.replace("?", "_").replace("[", "_").replace("]", "_")
    return value


def make_safe_sheet_name(name, used_names):
    """
    Excel sheet names must be <= 31 characters.
    """
    name = make_safe_name(name)
    name = name[:31]

    original_name = name
    counter = 1

    while name in used_names:
        suffix = f"_{counter}"
        name = original_name[:31 - len(suffix)] + suffix
        counter += 1

    used_names.add(name)
    return name


def parse_list_value(value):
    """
    Convert selected feature list from Excel text to Python list.

    Supported formats:
    "[1, 5, 10]"
    "['Feature_A', 'Feature_B']"
    "[1 5 10]"
    "1 5 10"
    "1, 5, 10"
    """
    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if pd.isna(value):
        return []

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, np.ndarray)):
            return list(parsed)

        if isinstance(parsed, (int, float, str)):
            return [parsed]

    except Exception:
        pass

    text = text.replace("[", "").replace("]", "").replace(",", " ")
    parts = text.split()

    return parts


def get_dataset_sheet_name(scores_sheet_name, df_scores, train_excel):
    """
    Map a scores sheet name back to the train/test dataset sheet name.

    Example:
        Sheet1_Top3Overall -> Sheet1
        Sheet1_Top3ByFeat  -> Sheet1
    """
    if "Dataset_Sheet" in df_scores.columns:
        values = df_scores["Dataset_Sheet"].dropna().unique()

        if len(values) > 0:
            candidate = str(values[0])

            if candidate in train_excel.sheet_names:
                return candidate

    if scores_sheet_name in train_excel.sheet_names:
        return scores_sheet_name

    suffixes = [
        "_Top3Overall",
        "_Top3ByFeat",
        "_Ranked",
        "_BestByFeat",
    ]

    for suffix in suffixes:
        if scores_sheet_name.endswith(suffix):
            candidate = scores_sheet_name.replace(suffix, "")

            if candidate in train_excel.sheet_names:
                return candidate

    # Fallback for common case
    if "Sheet1" in train_excel.sheet_names:
        return "Sheet1"

    return train_excel.sheet_names[0]


def prepare_labels(df, labels_column):
    """
    Encode binary labels.

    Supports:
        Control / Flatfoot
        Control / Autism
        Young / Older
        0 / 1
    """
    if labels_column not in df.columns:
        raise ValueError(f"Column '{labels_column}' not found in dataset.")

    labels = df[labels_column].astype(str).str.strip()

    label_mapping = {
        "Control": 0,
        "control": 0,
        "Young": 0,
        "young": 0,
        "0": 0,
        0: 0,

        "Flatfoot": 1,
        "flatfoot": 1,
        "Autism": 1,
        "autism": 1,
        "Older": 1,
        "older": 1,
        "1": 1,
        1: 1,
    }

    y = labels.replace(label_mapping)
    y = pd.to_numeric(y, errors="coerce")

    if y.isna().any():
        bad_labels = labels[y.isna()].unique()

        raise ValueError(
            f"Some labels could not be converted to 0/1: {bad_labels}. "
            f"Available labels are: {labels.unique()}"
        )

    return y.astype(int).values


def get_numeric_features_train_explain(
    df_train,
    df_explain,
    labels_column="Group",
    participant_column="Participant",
):
    """
    Extract numeric feature columns from train and explain data.

    Removes:
        Group
        Participant
    """
    drop_cols = [labels_column]

    if participant_column in df_train.columns:
        drop_cols.append(participant_column)

    X_train_df = df_train.drop(columns=drop_cols, errors="ignore")
    X_train_df = X_train_df.select_dtypes(include=[np.number])
    X_train_df = X_train_df.dropna(axis=1, how="all")

    train_medians = X_train_df.median()
    X_train_df = X_train_df.fillna(train_medians)

    X_explain_df = df_explain.drop(columns=drop_cols, errors="ignore")
    X_explain_df = X_explain_df.select_dtypes(include=[np.number])
    X_explain_df = X_explain_df.reindex(columns=X_train_df.columns)
    X_explain_df = X_explain_df.fillna(train_medians)

    return X_train_df, X_explain_df


def get_selected_feature_columns(row, available_feature_columns):
    """
    Get selected feature names from the scores file.

    Supports these columns:
        Selected_Features
        Feature_Names
        Features
        Ordered Indices
        Selected_Indices
    """
    available_feature_columns = list(available_feature_columns)

    # Prefer real feature names
    for col in ["Selected_Features", "Feature_Names"]:
        if col in row.index:
            values = parse_list_value(row[col])
            values = [str(v) for v in values]

            valid_names = [
                value for value in values
                if value in available_feature_columns
            ]

            if len(valid_names) > 0:
                return valid_names

    # Otherwise use indices
    for col in ["Selected_Indices", "Features", "Ordered Indices"]:
        if col in row.index:
            values = parse_list_value(row[col])

            clean_indices = []

            for value in values:
                try:
                    index = int(float(value))

                    if 0 <= index < len(available_feature_columns):
                        clean_indices.append(index)

                except Exception:
                    continue

            if len(clean_indices) > 0:
                return [
                    available_feature_columns[index]
                    for index in clean_indices
                ]

    raise ValueError(
        "No selected features found in scores row. "
        f"Available row columns are: {row.index.tolist()}"
    )


def get_model_name(row, row_index, sheet_name):
    """
    Get model name for output sheets and PNG files.
    """
    if "Name_Model" in row.index and not pd.isna(row["Name_Model"]):
        return row["Name_Model"]

    if "Rank" in row.index and not pd.isna(row["Rank"]):
        return f"{sheet_name}_Rank_{int(row['Rank'])}"

    return f"{sheet_name}_Model_{row_index}"


# ============================================================
# SHAP IMAGE
# ============================================================

def create_shap_bar_image(df_ranking, model_name, output_dir):
    """
    Create horizontal bar image from global mean absolute SHAP ranking.
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
    plt.title(f"LightGBM SHAP Summary - Model {model_name}")
    plt.tight_layout()

    safe_model_name = make_safe_name(model_name)
    output_path = output_dir / f"SHAP_LightGBM_{safe_model_name}_summary.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved SHAP image: {output_path}")


# ============================================================
# LIGHTGBM + SHAP
# ============================================================

def train_lightgbm_and_calculate_shap(
    X_train_df,
    y_train,
    X_explain_df,
    selected_feature_columns,
    model_params,
):
    """
    Train one LightGBM model and calculate SHAP values.
    """
    missing_features = [
        feature for feature in selected_feature_columns
        if feature not in X_train_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Selected features not found in training data: {missing_features}"
        )

    X_train_selected = X_train_df[selected_feature_columns]
    X_explain_selected = X_explain_df[selected_feature_columns]

    model = LGBMClassifier(
        n_estimators=int(model_params["n_estimators"]),
        max_depth=int(model_params["max_depth"]),
        learning_rate=float(model_params["learning_rate"]),
        num_leaves=int(model_params.get("num_leaves", 31)),
        subsample=float(model_params.get("subsample", 1.0)),
        colsample_bytree=float(model_params.get("colsample_bytree", 1.0)),
        min_child_samples=int(model_params.get("min_child_samples", 5)),
        reg_alpha=float(model_params.get("reg_alpha", 0.0)),
        reg_lambda=float(model_params.get("reg_lambda", 0.0)),
        objective="binary",
        random_state=int(model_params.get("random_state", 42)),
        n_jobs=1,
        verbose=-1,
    )

    model.fit(X_train_selected, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain_selected)

    # LightGBM binary classification may return:
    # 1. list [class0_array, class1_array]
    # 2. array samples x features
    # 3. array samples x features x classes
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_array = np.array(shap_values[1])
        else:
            shap_array = np.array(shap_values[0])
    else:
        shap_array = np.array(shap_values)

    if shap_array.ndim == 2:
        pass
    elif shap_array.ndim == 3:
        if shap_array.shape[2] == 2:
            shap_array = shap_array[:, :, 1]
        else:
            shap_array = np.mean(shap_array, axis=2)
    else:
        raise ValueError(f"Unexpected SHAP values shape: {shap_array.shape}")

    return shap_array, selected_feature_columns


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

def get_SHAP_data_LightGBM(
    scores_file,
    train_file,
    test_file,
    output_file,
    output_dir,
    labels_column="Group",
    participant_column="Participant",
):
    """
    Main function to perform LightGBM SHAP analysis.
    """
    logger.info("Loading LightGBM scores file...")

    scores_excel = pd.ExcelFile(scores_file)
    train_excel = pd.ExcelFile(train_file)

    if Path(test_file).exists():
        test_excel = pd.ExcelFile(test_file)
    else:
        test_excel = None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    shap_value_sheets = {}
    ranking_sheets = {}
    used_sheet_names = set()

    for scores_sheet_name in scores_excel.sheet_names:
        logger.info(f"Processing scores sheet: {scores_sheet_name}")

        df_scores = pd.read_excel(scores_file, sheet_name=scores_sheet_name)

        if df_scores.empty:
            logger.warning(f"No scores found for sheet: {scores_sheet_name}")
            continue

        dataset_sheet_name = get_dataset_sheet_name(
            scores_sheet_name=scores_sheet_name,
            df_scores=df_scores,
            train_excel=train_excel,
        )

        logger.info(f"Mapped scores sheet to dataset sheet: {dataset_sheet_name}")
        logger.info(f"Columns found in scores file: {df_scores.columns.tolist()}")

        df_train_raw = pd.read_excel(train_file, sheet_name=dataset_sheet_name)

        if test_excel is not None and dataset_sheet_name in test_excel.sheet_names:
            df_test_raw = pd.read_excel(test_file, sheet_name=dataset_sheet_name)
        else:
            logger.warning(
                f"Test sheet {dataset_sheet_name} not found. "
                f"SHAP will explain train data only."
            )
            df_test_raw = pd.DataFrame(columns=df_train_raw.columns)

        y_train = prepare_labels(df_train_raw, labels_column)

        df_all_raw = pd.concat(
            [df_train_raw, df_test_raw],
            ignore_index=True,
        )

        X_train_all, X_explain_all = get_numeric_features_train_explain(
            df_train=df_train_raw,
            df_explain=df_all_raw,
            labels_column=labels_column,
            participant_column=participant_column,
        )

        required_columns = [
            "#Features",
            "n_estimators",
            "max_depth",
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

        for row_index, row in df_scores.iterrows():
            model_name = get_model_name(
                row=row,
                row_index=row_index,
                sheet_name=scores_sheet_name,
            )

            try:
                selected_feature_columns = get_selected_feature_columns(
                    row=row,
                    available_feature_columns=X_train_all.columns,
                )
            except Exception as e:
                logger.warning(
                    f"Skipping model {model_name} because selected features "
                    f"could not be read: {e}"
                )
                continue

            if len(selected_feature_columns) == 0:
                logger.warning(
                    f"Skipping model {model_name} because selected features are empty."
                )
                continue

            model_params = {
                "n_estimators": row["n_estimators"],
                "max_depth": row["max_depth"],
                "learning_rate": row["learning_rate"],
                "num_leaves": row["num_leaves"] if "num_leaves" in df_scores.columns else 31,
                "subsample": row["subsample"] if "subsample" in df_scores.columns else 1.0,
                "colsample_bytree": row["colsample_bytree"] if "colsample_bytree" in df_scores.columns else 1.0,
                "min_child_samples": row["min_child_samples"] if "min_child_samples" in df_scores.columns else 5,
                "reg_alpha": row["reg_alpha"] if "reg_alpha" in df_scores.columns else 0.0,
                "reg_lambda": row["reg_lambda"] if "reg_lambda" in df_scores.columns else 0.0,
                "random_state": row["Random_State"] if "Random_State" in df_scores.columns else 42,
            }

            shap_values, selected_feature_names = train_lightgbm_and_calculate_shap(
                X_train_df=X_train_all,
                y_train=y_train,
                X_explain_df=X_explain_all,
                selected_feature_columns=selected_feature_columns,
                model_params=model_params,
            )

            logger.info(
                f"SHAP completed for {dataset_sheet_name} | "
                f"Model {model_name} | "
                f"Selected {len(selected_feature_names)} features"
            )

            df_shap_values = create_shap_values_dataframe(
                shap_values=shap_values,
                feature_names=selected_feature_names,
                metadata_df=df_all_raw,
            )

            df_ranking = create_global_ranking_dataframe(
                shap_values=shap_values,
                feature_names=selected_feature_names,
            )

            df_ranking["Name_Model"] = model_name

            if "CV_Accuracy" in df_scores.columns:
                df_ranking["CV_Accuracy"] = row["CV_Accuracy"]

            if "Test_Accuracy" in df_scores.columns:
                df_ranking["Test_Accuracy"] = row["Test_Accuracy"]

            df_ranking["#Features"] = row["#Features"]
            df_ranking["n_estimators"] = row["n_estimators"]
            df_ranking["max_depth"] = row["max_depth"]
            df_ranking["learning_rate"] = row["learning_rate"]

            if "num_leaves" in df_scores.columns:
                df_ranking["num_leaves"] = row["num_leaves"]

            if "Sensitivity" in df_scores.columns:
                df_ranking["Sensitivity"] = row["Sensitivity"]

            if "Specificity" in df_scores.columns:
                df_ranking["Specificity"] = row["Specificity"]

            df_ranking["Selected_Features"] = str(selected_feature_names)

            create_shap_bar_image(
                df_ranking=df_ranking,
                model_name=model_name,
                output_dir=output_dir,
            )

            shap_sheet_name = make_safe_sheet_name(
                f"SHAP_{dataset_sheet_name}_{model_name}",
                used_sheet_names,
            )

            rank_sheet_name = make_safe_sheet_name(
                f"Rank_{dataset_sheet_name}_{model_name}",
                used_sheet_names,
            )

            shap_value_sheets[shap_sheet_name] = df_shap_values
            ranking_sheets[rank_sheet_name] = df_ranking

    if not shap_value_sheets and not ranking_sheets:
        raise ValueError("No LightGBM SHAP results were generated. Check input files.")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df in shap_value_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        for sheet_name, df in ranking_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Saved LightGBM SHAP Excel results to: {output_file}")
    logger.info(f"Saved LightGBM SHAP images to: {output_dir}")
