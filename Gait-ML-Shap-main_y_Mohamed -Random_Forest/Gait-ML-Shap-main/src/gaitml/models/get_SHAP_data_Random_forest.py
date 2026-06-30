import os
import ast
import warnings

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from gaitml import logger

warnings.filterwarnings("ignore")


# ============================================================
# HELPERS
# ============================================================

def clean_feature_matrix(X):
    """
    Convert all feature columns to numeric and remove fully empty columns.
    """
    X = X.copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.dropna(axis=1, how="all")

    return X


def encode_train_test_labels(y_train, y_test):
    """
    Encode train/test labels with the same mapping.
    """
    y_train = y_train.astype(str).str.strip()
    y_test = y_test.astype(str).str.strip()

    encoder = LabelEncoder()
    encoder.fit(pd.concat([y_train, y_test], axis=0))

    return encoder.transform(y_train), encoder.transform(y_test)


def parse_indices(value):
    """
    Parse feature indices.

    Expected examples:
        [1, 2, 3]
        "[1, 2, 3]"
        "[1 2 3]"
    """

    if isinstance(value, list):
        return [int(v) for v in value]

    value = str(value).strip()

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [int(v) for v in parsed]

        if isinstance(parsed, tuple):
            return [int(v) for v in parsed]

    except Exception:
        pass

    value = value.replace("[", "").replace("]", "").replace(",", " ")

    indices = []

    for part in value.split():
        try:
            indices.append(int(part))
        except Exception:
            pass

    return indices


def get_selected_indices_from_row(row):
    """
    Read feature numbers from the scores file.

    Priority:
        1. Ordered Indices
        2. Features

    In your Random Forest Phase 4.2, Features now contains feature numbers,
    not feature names.
    """

    if "Ordered Indices" in row and pd.notna(row["Ordered Indices"]):
        return parse_indices(row["Ordered Indices"])

    if "Features" in row and pd.notna(row["Features"]):
        return parse_indices(row["Features"])

    return []


def parse_max_depth(value):
    """
    Convert max_depth from Excel to valid RandomForest value.
    """
    if pd.isna(value):
        return None

    value_str = str(value).strip().lower()

    if value_str in ["none", "nan", "null", ""]:
        return None

    return int(float(value))


def get_column_value(row, column_name, default=None):
    """
    Safely get value from row.
    """
    if column_name in row and pd.notna(row[column_name]):
        return row[column_name]

    return default


def build_random_forest_model(row):
    """
    Build Random Forest model from one row of the Top3 scores file.
    """

    n_estimators = int(float(get_column_value(row, "n_estimators", 100)))
    max_depth = parse_max_depth(get_column_value(row, "max_depth", None))

    min_samples_split = int(
        float(get_column_value(row, "min_samples_split", 2))
    )

    min_samples_leaf = int(
        float(get_column_value(row, "min_samples_leaf", 1))
    )

    max_features = get_column_value(row, "max_features", "sqrt")

    if pd.isna(max_features):
        max_features = "sqrt"

    max_features = str(max_features).strip()

    random_state = get_column_value(row, "Random_State", 0)

    try:
        random_state = int(float(random_state))
    except Exception:
        random_state = 0

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1
    )

    return model


def get_shap_values_for_positive_class(shap_values):
    """
    Return SHAP values for class 1.

    Depending on SHAP version, output can be:
        - list[class0_values, class1_values]
        - ndarray with shape (samples, features, classes)
        - ndarray with shape (samples, features)
    """

    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            return shap_values[1]
        return shap_values[0]

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        return shap_values[:, :, 1]

    return shap_values


def safe_sheet_name(name, used_names):
    """
    Excel sheet names must be <= 31 chars and unique.
    """

    base_name = str(name).replace("/", "_").replace("\\", "_")
    base_name = base_name.replace("[", "").replace("]", "")
    base_name = base_name.replace("*", "_").replace("?", "_")
    base_name = base_name.replace(":", "_")

    base_name = base_name[:31]

    final_name = base_name
    counter = 1

    while final_name in used_names:
        suffix = f"_{counter}"
        final_name = base_name[:31 - len(suffix)] + suffix
        counter += 1

    used_names.add(final_name)

    return final_name


# ============================================================
# MAIN FUNCTION
# ============================================================

def get_SHAP_data_Random_forest(
    scores_file,
    train_file,
    test_file,
    output_file,
    output_dir,
    labels_column
):
    """
    Generate SHAP data and SHAP summary plots for Random Forest Top models.

    Important:
        The scores file contains feature numbers/indices.
        This function converts them to feature names internally.
    """

    logger.info("Starting Random Forest SHAP generation")
    logger.info(f"Scores file: {scores_file}")
    logger.info(f"Train file: {train_file}")
    logger.info(f"Test file: {test_file}")
    logger.info(f"Output SHAP Excel: {output_file}")
    logger.info(f"Output SHAP directory: {output_dir}")

    if not os.path.exists(scores_file):
        raise FileNotFoundError(f"Scores file not found: {scores_file}")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    os.makedirs(output_dir, exist_ok=True)

    output_parent = os.path.dirname(output_file)

    if output_parent != "":
        os.makedirs(output_parent, exist_ok=True)

    scores_df = pd.read_excel(scores_file)

    if scores_df.empty:
        logger.warning("Scores file is empty. No SHAP generated.")
        return

    train_excel = pd.ExcelFile(train_file)
    test_excel = pd.ExcelFile(test_file)

    shap_sheets = {}
    used_sheet_names = set()

    for idx, row in scores_df.iterrows():

        model_name = str(row.get("Name_Model", f"RF_Model_{idx + 1}"))
        logger.info(f"Processing Random Forest SHAP for model: {model_name}")

        sheet_name = row.get("Sheet", None)

        if sheet_name is None or pd.isna(sheet_name):
            sheet_name = train_excel.sheet_names[0]

        sheet_name = str(sheet_name)

        if sheet_name not in train_excel.sheet_names:
            logger.warning(
                f"Sheet {sheet_name} not found in train file. Skipping {model_name}."
            )
            continue

        if sheet_name not in test_excel.sheet_names:
            logger.warning(
                f"Sheet {sheet_name} not found in test file. Skipping {model_name}."
            )
            continue

        train_df = pd.read_excel(train_file, sheet_name=sheet_name)
        test_df = pd.read_excel(test_file, sheet_name=sheet_name)

        if labels_column not in train_df.columns or labels_column not in test_df.columns:
            logger.warning(
                f"Label column {labels_column} not found. Skipping {model_name}."
            )
            continue

        y_train_raw = train_df[labels_column]
        y_test_raw = test_df[labels_column]

        # Same feature segment used in SFS and Phase 4.2
        X_train_raw = train_df.loc[:, "Pelv_Angle_Y_MAX_SW":"Hip_Angle_Z_OHS"]
        X_test_raw = test_df.loc[:, "Pelv_Angle_Y_MAX_SW":"Hip_Angle_Z_OHS"]

        X_train = clean_feature_matrix(X_train_raw)
        X_test = clean_feature_matrix(X_test_raw)

        common_features = [
            col for col in X_train.columns
            if col in X_test.columns
        ]

        X_train = X_train[common_features]
        X_test = X_test[common_features]

        feature_columns = list(X_train.columns)

        y_train, y_test = encode_train_test_labels(y_train_raw, y_test_raw)

        selected_indices = get_selected_indices_from_row(row)

        selected_features = [
            feature_columns[i]
            for i in selected_indices
            if 0 <= i < len(feature_columns)
        ]

        if len(selected_features) == 0:
            logger.warning(
                f"No valid selected feature indices found for {model_name}. Skipping."
            )
            continue

        logger.info(f"Selected indices for {model_name}: {selected_indices}")
        logger.info(f"Selected features for {model_name}: {selected_features}")

        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        # ====================================================
        # IMPUTE DATA
        # ====================================================

        imputer = SimpleImputer(strategy="median")

        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train_selected),
            columns=selected_features,
            index=X_train_selected.index
        )

        X_test_imputed = pd.DataFrame(
            imputer.transform(X_test_selected),
            columns=selected_features,
            index=X_test_selected.index
        )

        # ====================================================
        # TRAIN RANDOM FOREST
        # ====================================================

        model = build_random_forest_model(row)
        model.fit(X_train_imputed, y_train)

        # ====================================================
        # SHAP
        # ====================================================

        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(X_test_imputed)
        shap_values = get_shap_values_for_positive_class(shap_values_raw)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        shap_df = pd.DataFrame(
            shap_values,
            columns=[f"SHAP_{feature}" for feature in selected_features]
        )

        raw_features_df = X_test_imputed.reset_index(drop=True)
        raw_features_df.columns = [
            f"RAW_{feature}" for feature in selected_features
        ]

        model_info_df = pd.DataFrame({
            "Model_Name": [model_name] * len(X_test_imputed),
            "Sheet": [sheet_name] * len(X_test_imputed),
            "Selected_Indices": [str(selected_indices)] * len(X_test_imputed),
            "True_Label": y_test,
            "Original_Label": y_test_raw.reset_index(drop=True)
        })

        final_shap_df = pd.concat(
            [
                model_info_df.reset_index(drop=True),
                raw_features_df.reset_index(drop=True),
                shap_df.reset_index(drop=True)
            ],
            axis=1
        )

        sheet_output_name = safe_sheet_name(
            f"{model_name}_SHAP",
            used_sheet_names
        )

        shap_sheets[sheet_output_name] = final_shap_df

        importance_df = pd.DataFrame({
            "Feature_Index": selected_indices[:len(selected_features)],
            "Feature": selected_features,
            "Mean_Abs_SHAP": mean_abs_shap
        }).sort_values(by="Mean_Abs_SHAP", ascending=False)

        importance_sheet_name = safe_sheet_name(
            f"{model_name}_importance",
            used_sheet_names
        )

        shap_sheets[importance_sheet_name] = importance_df

        # ====================================================
        # SAVE SHAP PLOT
        # ====================================================

        safe_model_name = str(model_name).replace("/", "_").replace("\\", "_")
        safe_model_name = safe_model_name.replace(":", "_").replace(" ", "_")

        plot_file = os.path.join(
            output_dir,
            f"SHAP_RandomForest_{safe_model_name}_summary.png"
        )

        plt.figure()
        shap.summary_plot(
            shap_values,
            X_test_imputed,
            feature_names=selected_features,
            show=False,
            plot_type="bar"
        )
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved SHAP plot: {plot_file}")

    # ========================================================
    # SAVE EXCEL
    # ========================================================

    if len(shap_sheets) == 0:
        logger.warning("No SHAP sheets generated.")
        return

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df in shap_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Random Forest SHAP data saved to: {output_file}")
    logger.info("Completed Random Forest SHAP generation")