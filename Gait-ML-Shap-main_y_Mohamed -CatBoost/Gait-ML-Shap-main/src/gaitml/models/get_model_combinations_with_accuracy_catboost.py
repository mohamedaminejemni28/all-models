"""
get_model_combinations_with_accuracy_catboost.py

This script performs Phase 4.2 / Step 2:
CatBoost model combinations using feature sets produced by CatBoost SFS.

It is the CatBoost version of:
    get_model_combinations_with_accuracy_xgboost.py

Output columns include:
    CV_Accuracy
    Test_Accuracy
    Sensitivity
    Specificity
    #Features
    iterations
    depth
    learning_rate
    l2_leaf_reg
    bagging_temperature
    random_strength
    CV_split
    Random_State
    Feature_idx
    Feature_Names
    Confusion_Matrix
    y_pred
"""

import os
import ast
import json
import random
import logging
import itertools

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier


def get_sensitivity(y_true, y_pred):
    """
    Sensitivity = TP / (TP + FN)
    Also called recall for the positive class.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return 0.0


def get_specificity(y_true, y_pred):
    """
    Specificity = TN / (TN + FP)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return 0.0


def parse_list(value):
    """
    Parse list values saved as strings in Excel.

    Supports:
        "['feature1', 'feature2']"
        "[1, 2, 3]"
        JSON lists
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


def clean_and_encode_labels(df_train, df_test, group_column):
    """
    Clean and encode labels safely.

    Works with:
        Control / Autism
        Control / Flatfoot
        Young / Older
        0 / 1
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train[group_column] = df_train[group_column].astype(str).str.strip()
    df_test[group_column] = df_test[group_column].astype(str).str.strip()

    label_encoder = LabelEncoder()

    all_labels = pd.concat(
        [df_train[group_column], df_test[group_column]],
        axis=0,
    )

    label_encoder.fit(all_labels)

    df_train[group_column] = label_encoder.transform(df_train[group_column])
    df_test[group_column] = label_encoder.transform(df_test[group_column])

    return df_train, df_test, label_encoder


def get_numeric_feature_columns(df, group_column="Group", participant_column="Participant"):
    """
    Get numeric feature columns only.
    Remove label and participant columns.
    """
    drop_cols = [group_column]

    if participant_column in df.columns:
        drop_cols.append(participant_column)

    feature_df = df.drop(columns=drop_cols, errors="ignore")
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

    return numeric_cols


def extract_feature_set(row, train_feature_columns):
    """
    Extract selected features from one SFS row.

    Priority:
        1. Feature_Names / Selected_Features if available
        2. Ordered Indices / Features / Selected_Indices if available

    Returns:
        selected_feature_names
    """
    name_columns = [
        "Feature_Names",
        "Selected_Features",
        "Selected_Feature_Names",
    ]

    index_columns = [
        "Ordered Indices",
        "Features",
        "Selected_Indices",
        "Feature_idx",
    ]

    for col in name_columns:
        if col in row.index:
            values = parse_list(row[col])

            selected_names = []
            for value in values:
                name = str(value).strip()

                if name in train_feature_columns:
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

                    if 0 <= index < len(train_feature_columns):
                        selected_names.append(train_feature_columns[index])
                except Exception:
                    continue

            if len(selected_names) > 0:
                return selected_names

    return []


def build_catboost_model(
    iterations,
    depth,
    learning_rate,
    l2_leaf_reg,
    bagging_temperature,
    random_strength,
    random_state,
):
    """
    Create CatBoost binary classifier.
    """
    return CatBoostClassifier(
        iterations=int(iterations),
        depth=int(depth),
        learning_rate=float(learning_rate),
        l2_leaf_reg=float(l2_leaf_reg),
        bagging_temperature=float(bagging_temperature),
        random_strength=float(random_strength),
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=int(random_state),
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )


def save_step2_results(step2_results, output_excel_file):
    """
    Save all sheet results to Excel.
    """
    output_dir = os.path.dirname(output_excel_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_excel_file, engine="openpyxl") as writer:
        for sheet_name, df_result in step2_results.items():
            safe_sheet_name = str(sheet_name)[:31]

            if df_result.empty:
                empty_df = pd.DataFrame(
                    columns=[
                        "CV_Accuracy",
                        "Test_Accuracy",
                        "Sensitivity",
                        "Specificity",
                        "#Features",
                        "iterations",
                        "depth",
                        "learning_rate",
                        "l2_leaf_reg",
                        "bagging_temperature",
                        "random_strength",
                        "CV_split",
                        "Random_State",
                        "Feature_idx",
                        "Feature_Names",
                        "Confusion_Matrix",
                        "y_pred",
                    ]
                )
                empty_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
            else:
                df_result.to_excel(writer, sheet_name=safe_sheet_name, index=False)


def get_combination_list_with_accuracy_catboost(
    train_excel_file,
    test_excel_file,
    sfs_results_excel_file,
    output_excel_file="Step2_Results_CatBoost.xlsx",
    group_column="Group",
    participant_column="Participant",
    feature_range=(5, 20),
    iterations_set=None,
    depth_set=None,
    learning_rate_set=None,
    l2_leaf_reg_set=None,
    bagging_temperature_set=None,
    random_strength_set=None,
    n_splits=5,
    random_state=42,
    max_candidates=50,
    logger=None,
):
    """
    Evaluate CatBoost models on selected feature subsets and hyperparameter combinations.

    Parameters:
        train_excel_file:
            Training Excel file.

        test_excel_file:
            Testing Excel file.

        sfs_results_excel_file:
            CatBoost SFS results from Phase 4.1.

        output_excel_file:
            Output Excel file for Phase 4.2 CatBoost results.

        group_column:
            Target column.

        participant_column:
            Participant column to exclude from features.

        feature_range:
            Example: (5, 20) means feature counts from 5 to 19.

        max_candidates:
            Maximum number of hyperparameter combinations to test.
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    if iterations_set is None:
        iterations_set = [50, 100, 150, 200]

    if depth_set is None:
        depth_set = [2, 3, 4, 5]

    if learning_rate_set is None:
        learning_rate_set = [0.01, 0.03, 0.05, 0.1]

    if l2_leaf_reg_set is None:
        l2_leaf_reg_set = [1, 3, 5, 7]

    if bagging_temperature_set is None:
        bagging_temperature_set = [0.0, 0.5, 1.0]

    if random_strength_set is None:
        random_strength_set = [0.0, 0.5, 1.0]

    feature_len_list = list(range(int(feature_range[0]), int(feature_range[1])))

    parameter_grid = list(
        itertools.product(
            iterations_set,
            depth_set,
            learning_rate_set,
            l2_leaf_reg_set,
            bagging_temperature_set,
            random_strength_set,
        )
    )

    total_before_sampling = len(parameter_grid)

    if max_candidates is not None and int(max_candidates) > 0:
        if len(parameter_grid) > int(max_candidates):
            random.seed(random_state)
            parameter_grid = random.sample(parameter_grid, int(max_candidates))

    logger.info(f"Loading train file: {train_excel_file}")
    logger.info(f"Loading test file: {test_excel_file}")
    logger.info(f"Loading CatBoost SFS results file: {sfs_results_excel_file}")
    logger.info(f"Parameter combinations before sampling: {total_before_sampling}")
    logger.info(f"Parameter combinations used: {len(parameter_grid)}")
    logger.info(f"Feature lengths: {feature_len_list}")

    excel_train = pd.ExcelFile(train_excel_file)
    excel_test = pd.ExcelFile(test_excel_file)
    excel_sfs = pd.ExcelFile(sfs_results_excel_file)

    step2_results = {}

    for sheet_name in excel_train.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")

        if sheet_name not in excel_test.sheet_names:
            logger.warning(f"Sheet {sheet_name} not found in test file. Skipping.")
            continue

        if sheet_name not in excel_sfs.sheet_names:
            logger.warning(f"Sheet {sheet_name} not found in SFS file. Skipping.")
            continue

        df_train_original = pd.read_excel(train_excel_file, sheet_name=sheet_name)
        df_test_original = pd.read_excel(test_excel_file, sheet_name=sheet_name)
        df_features = pd.read_excel(sfs_results_excel_file, sheet_name=sheet_name)

        if group_column not in df_train_original.columns:
            raise ValueError(
                f"Column '{group_column}' not found in train sheet: {sheet_name}"
            )

        if group_column not in df_test_original.columns:
            raise ValueError(
                f"Column '{group_column}' not found in test sheet: {sheet_name}"
            )

        df_train, df_test, label_encoder = clean_and_encode_labels(
            df_train=df_train_original,
            df_test=df_test_original,
            group_column=group_column,
        )

        y_train = df_train[group_column].values
        y_test = df_test[group_column].values

        train_feature_columns = get_numeric_feature_columns(
            df=df_train,
            group_column=group_column,
            participant_column=participant_column,
        )

        if len(train_feature_columns) == 0:
            logger.warning(f"No numeric features found in sheet {sheet_name}. Skipping.")
            continue

        logger.info(f"Number of numeric features: {len(train_feature_columns)}")
        logger.info(f"Classes: {dict(enumerate(label_encoder.classes_))}")

        min_class_count = np.min(np.bincount(y_train))
        local_n_splits = min(int(n_splits), int(min_class_count))

        if local_n_splits < 2:
            logger.warning(
                f"Not enough samples per class for CV in sheet {sheet_name}. Skipping."
            )
            continue

        skf_cv = StratifiedKFold(
            n_splits=local_n_splits,
            shuffle=True,
            random_state=random_state,
        )

        rows = []
        seen_feature_sets = set()

        for s, feature_row in df_features.iterrows():
            base_selected_features = extract_feature_set(
                row=feature_row,
                train_feature_columns=train_feature_columns,
            )

            if len(base_selected_features) == 0:
                logger.warning(
                    f"Could not extract selected features at row {s} in sheet {sheet_name}"
                )
                continue

            for feature_len in feature_len_list:
                selected_feature_names = base_selected_features[:feature_len]

                if len(selected_feature_names) != feature_len:
                    continue

                feature_set_key = tuple(selected_feature_names)

                if feature_set_key in seen_feature_sets:
                    continue

                seen_feature_sets.add(feature_set_key)

                X_train = df_train[selected_feature_names].apply(
                    pd.to_numeric,
                    errors="coerce",
                )

                X_test = df_test[selected_feature_names].apply(
                    pd.to_numeric,
                    errors="coerce",
                )

                X_train = X_train.fillna(X_train.median())
                X_test = X_test.fillna(X_train.median())

                X_train = X_train.to_numpy(dtype=np.float32)
                X_test = X_test.to_numpy(dtype=np.float32)

                for combo_id, (
                    iterations,
                    depth,
                    learning_rate,
                    l2_leaf_reg,
                    bagging_temperature,
                    random_strength,
                ) in enumerate(parameter_grid, start=1):

                    y_true_cv_all = []
                    y_pred_cv_all = []

                    for train_idx, valid_idx in skf_cv.split(X_train, y_train):
                        x_tr = X_train[train_idx]
                        y_tr = y_train[train_idx]

                        x_valid = X_train[valid_idx]
                        y_valid = y_train[valid_idx]

                        model = build_catboost_model(
                            iterations=iterations,
                            depth=depth,
                            learning_rate=learning_rate,
                            l2_leaf_reg=l2_leaf_reg,
                            bagging_temperature=bagging_temperature,
                            random_strength=random_strength,
                            random_state=random_state,
                        )

                        model.fit(x_tr, y_tr)

                        y_pred_valid = model.predict(x_valid)
                        y_pred_valid = np.array(y_pred_valid).astype(int).ravel()

                        y_true_cv_all.extend(y_valid)
                        y_pred_cv_all.extend(y_pred_valid)

                    cv_acc = accuracy_score(y_true_cv_all, y_pred_cv_all)

                    final_model = build_catboost_model(
                        iterations=iterations,
                        depth=depth,
                        learning_rate=learning_rate,
                        l2_leaf_reg=l2_leaf_reg,
                        bagging_temperature=bagging_temperature,
                        random_strength=random_strength,
                        random_state=random_state,
                    )

                    final_model.fit(X_train, y_train)

                    if len(X_test) == 0:
                        y_pred_test = np.array([], dtype=int)
                        test_acc = np.nan
                        sensitivity = np.nan
                        specificity = np.nan
                        cm = np.array([])
                    else:
                        y_pred_test = final_model.predict(X_test)
                        y_pred_test = np.array(y_pred_test).astype(int).ravel()

                        test_acc = accuracy_score(y_test, y_pred_test)
                        sensitivity = get_sensitivity(y_test, y_pred_test)
                        specificity = get_specificity(y_test, y_pred_test)
                        cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])

                    row = {
                        "CV_Accuracy": float(cv_acc),
                        "Test_Accuracy": float(test_acc) if not pd.isna(test_acc) else np.nan,
                        "Sensitivity": float(sensitivity) if not pd.isna(sensitivity) else np.nan,
                        "Specificity": float(specificity) if not pd.isna(specificity) else np.nan,
                        "#Features": int(len(selected_feature_names)),
                        "iterations": int(iterations),
                        "depth": int(depth),
                        "learning_rate": float(learning_rate),
                        "l2_leaf_reg": float(l2_leaf_reg),
                        "bagging_temperature": float(bagging_temperature),
                        "random_strength": float(random_strength),
                        "CV_split": int(local_n_splits),
                        "Random_State": int(random_state),
                        "Feature_idx": int(s),
                        "Feature_Names": json.dumps(selected_feature_names),
                        "Confusion_Matrix": str(cm.tolist() if hasattr(cm, "tolist") else cm),
                        "y_pred": json.dumps(y_pred_test.tolist()),
                    }

                    rows.append(row)

                logger.info(
                    f"Sheet: {sheet_name} | "
                    f"Feature_idx: {s} | "
                    f"#Features: {len(selected_feature_names)} | "
                    f"Results so far: {len(rows)}"
                )

        df_results = pd.DataFrame(rows)

        if not df_results.empty:
            df_results = df_results.sort_values(
                by=[
                    "CV_Accuracy",
                    "Test_Accuracy",
                    "Sensitivity",
                    "Specificity",
                    "#Features",
                ],
                ascending=[False, False, False, False, True],
            ).reset_index(drop=True)

        step2_results[sheet_name] = df_results

        save_step2_results(
            step2_results=step2_results,
            output_excel_file=output_excel_file,
        )

        logger.info(f"Partial CatBoost Step 2 results saved for sheet {sheet_name}")

    logger.info(f"Saving final CatBoost Step 2 results to {output_excel_file}")

    save_step2_results(
        step2_results=step2_results,
        output_excel_file=output_excel_file,
    )

    logger.info("CatBoost Step 2 completed successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 2: Combinations - Evaluate CatBoost on feature subsets and hyperparameters."
    )

    parser.add_argument(
        "--train",
        required=True,
        help="Path to train Excel file",
    )

    parser.add_argument(
        "--test",
        required=True,
        help="Path to test Excel file",
    )

    parser.add_argument(
        "--sfs",
        required=True,
        help="Path to CatBoost SFS results Excel file from Step 1",
    )

    parser.add_argument(
        "--output",
        default="Step2_Results_CatBoost.xlsx",
        help="Output Excel file",
    )

    args = parser.parse_args()

    get_combination_list_with_accuracy_catboost(
        train_excel_file=args.train,
        test_excel_file=args.test,
        sfs_results_excel_file=args.sfs,
        output_excel_file=args.output,
    )
