"""
get_model_combinations_with_accuracy_lightgbm.py

Phase 4.2: Model combinations using LightGBM.

This is the LightGBM version of:
    src/gaitml/models/get_model_combinations_with_accuracy_xgboost.py
"""

import os
import random
import itertools
import logging
from ast import literal_eval

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMClassifier


def get_sensitivity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        TP = cm[1, 1]
        FN = cm[1, 0]
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    return 0.0


def get_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        TN = cm[0, 0]
        FP = cm[0, 1]
        return TN / (TN + FP) if (TN + FP) > 0 else 0.0

    return 0.0


def clean_and_encode_labels(df_train, df_test, group_column):
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train[group_column] = df_train[group_column].astype(str).str.strip()

    label_encoder = LabelEncoder()

    if len(df_test) > 0:
        df_test[group_column] = df_test[group_column].astype(str).str.strip()

        all_labels = pd.concat(
            [df_train[group_column], df_test[group_column]],
            axis=0
        )

        label_encoder.fit(all_labels)

        df_train[group_column] = label_encoder.transform(df_train[group_column])
        df_test[group_column] = label_encoder.transform(df_test[group_column])
    else:
        label_encoder.fit(df_train[group_column])
        df_train[group_column] = label_encoder.transform(df_train[group_column])

    return df_train, df_test, label_encoder


def get_numeric_features_train_test(
    df_train,
    df_test,
    group_column="Group",
    participant_column="Participant",
):
    drop_cols = [group_column]

    if participant_column in df_train.columns:
        drop_cols.append(participant_column)

    X_train_df = df_train.drop(columns=drop_cols, errors="ignore")
    X_train_df = X_train_df.select_dtypes(include=[np.number])
    X_train_df = X_train_df.dropna(axis=1, how="all")

    train_medians = X_train_df.median()
    X_train_df = X_train_df.fillna(train_medians)

    if len(df_test) > 0:
        X_test_df = df_test.drop(columns=drop_cols, errors="ignore")
        X_test_df = X_test_df.select_dtypes(include=[np.number])
        X_test_df = X_test_df.reindex(columns=X_train_df.columns)
        X_test_df = X_test_df.fillna(train_medians)
    else:
        X_test_df = pd.DataFrame(columns=X_train_df.columns)

    return X_train_df, X_test_df


def parse_list_value(value):
    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if pd.isna(value):
        return []

    try:
        parsed = literal_eval(str(value))
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return list(parsed)
    except Exception:
        return []

    return []


def get_selected_feature_columns(row, all_feature_columns, feature_len=None):
    all_feature_columns = list(all_feature_columns)

    # Prefer Feature_Names from our LightGBM SFS output
    if "Feature_Names" in row.index:
        feature_names = parse_list_value(row["Feature_Names"])
        feature_names = [str(name) for name in feature_names]

        valid_names = [
            name for name in feature_names
            if name in all_feature_columns
        ]

        if len(valid_names) > 0:
            if feature_len is not None:
                valid_names = valid_names[:feature_len]

            return valid_names

    # Otherwise use numeric indices
    possible_index_columns = [
        "Ordered Indices",
        "Features",
        "Feature_idx",
        "Feature_Indices",
    ]

    selected_indices = []

    for col in possible_index_columns:
        if col in row.index:
            selected_indices = parse_list_value(row[col])
            if len(selected_indices) > 0:
                break

    if len(selected_indices) == 0:
        selected_indices = parse_list_value(row.iloc[-1])

    clean_indices = []

    for index in selected_indices:
        try:
            clean_index = int(index)
            if 0 <= clean_index < len(all_feature_columns):
                clean_indices.append(clean_index)
        except Exception:
            continue

    if feature_len is not None:
        clean_indices = clean_indices[:feature_len]

    selected_columns = [
        all_feature_columns[index]
        for index in clean_indices
    ]

    return selected_columns


def save_partial_results(step2_results, output_excel_file):
    output_dir = os.path.dirname(output_excel_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_excel_file, engine="openpyxl") as writer:
        for sheet_name, df_result in step2_results.items():
            safe_sheet_name = str(sheet_name)[:31]
            df_result.to_excel(writer, sheet_name=safe_sheet_name, index=False)


def get_combination_list_with_accuracy_lightgbm(
    train_excel_file,
    test_excel_file,
    sfs_results_excel_file,
    output_excel_file="Step2_Results_LightGBM.xlsx",
    group_column="Group",
    participant_column="Participant",
    feature_range=(5, 20),
    n_estimators_set=None,
    max_depth_set=None,
    learning_rate_set=None,
    num_leaves_set=None,
    subsample_set=None,
    colsample_bytree_set=None,
    min_child_samples_set=None,
    reg_alpha_set=None,
    reg_lambda_set=None,
    n_splits=10,
    random_state=0,
    max_candidates=1000,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    if n_estimators_set is None:
        n_estimators_set = [50, 100, 150]

    if max_depth_set is None:
        max_depth_set = [-1, 2, 3, 4]

    if learning_rate_set is None:
        learning_rate_set = [0.01, 0.05, 0.1]

    if num_leaves_set is None:
        num_leaves_set = [7, 15, 31]

    if subsample_set is None:
        subsample_set = [0.8, 1.0]

    if colsample_bytree_set is None:
        colsample_bytree_set = [0.8, 1.0]

    if min_child_samples_set is None:
        min_child_samples_set = [2, 5, 10]

    if reg_alpha_set is None:
        reg_alpha_set = [0.0]

    if reg_lambda_set is None:
        reg_lambda_set = [0.0]

    parameter_grid = list(
        itertools.product(
            n_estimators_set,
            max_depth_set,
            learning_rate_set,
            num_leaves_set,
            subsample_set,
            colsample_bytree_set,
            min_child_samples_set,
            reg_alpha_set,
            reg_lambda_set,
        )
    )

    total_before_sampling = len(parameter_grid)

    if max_candidates is not None and max_candidates > 0 and len(parameter_grid) > max_candidates:
        random.seed(random_state)
        parameter_grid = random.sample(parameter_grid, int(max_candidates))

    logger.info(f"Loading train file: {train_excel_file}")
    logger.info(f"Loading test file: {test_excel_file}")
    logger.info(f"Loading LightGBM SFS file: {sfs_results_excel_file}")
    logger.info(f"Parameter combinations before sampling: {total_before_sampling}")
    logger.info(f"Parameter combinations used: {len(parameter_grid)}")

    excel_train = pd.ExcelFile(train_excel_file)
    excel_test = pd.ExcelFile(test_excel_file)
    sfs_results = pd.ExcelFile(sfs_results_excel_file)

    step2_results = {}

    for sheet_name in excel_train.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")

        if sheet_name not in sfs_results.sheet_names:
            logger.warning(f"Sheet {sheet_name} not found in SFS file. Skipping.")
            continue

        df_train = pd.read_excel(train_excel_file, sheet_name=sheet_name)

        if sheet_name in excel_test.sheet_names:
            df_test = pd.read_excel(test_excel_file, sheet_name=sheet_name)
        else:
            logger.warning(f"Sheet {sheet_name} not found in test file. Using empty test set.")
            df_test = pd.DataFrame()

        df_features = pd.read_excel(sfs_results_excel_file, sheet_name=sheet_name)

        if group_column not in df_train.columns:
            raise ValueError(f"Column '{group_column}' not found in train sheet: {sheet_name}")

        if len(df_test) > 0 and group_column not in df_test.columns:
            raise ValueError(f"Column '{group_column}' not found in test sheet: {sheet_name}")

        df_train, df_test, label_encoder = clean_and_encode_labels(
            df_train=df_train,
            df_test=df_test,
            group_column=group_column,
        )

        y_train = df_train[group_column].values

        if len(df_test) > 0:
            y_test = df_test[group_column].values
        else:
            y_test = np.array([])

        X_train_all, X_test_all = get_numeric_features_train_test(
            df_train=df_train,
            df_test=df_test,
            group_column=group_column,
            participant_column=participant_column,
        )

        min_class_count = np.min(np.bincount(y_train))

        if min_class_count < 2:
            logger.warning(f"Not enough samples per class in sheet {sheet_name}. Skipping.")
            continue

        local_n_splits = min(n_splits, int(min_class_count))

        skf_cv = StratifiedKFold(
            n_splits=local_n_splits,
            shuffle=True,
            random_state=random_state,
        )

        # Build unique feature sets from SFS output
        feature_sets = []
        seen_feature_sets = set()

        for s in range(len(df_features)):
            row = df_features.iloc[s]

            if "#Features" in df_features.columns:
                try:
                    feature_lengths_to_test = [int(row["#Features"])]
                except Exception:
                    feature_lengths_to_test = []
            else:
                feature_lengths_to_test = list(np.arange(feature_range[0], feature_range[1]))

            for feature_len in feature_lengths_to_test:
                if feature_len < feature_range[0] or feature_len >= feature_range[1]:
                    continue

                selected_feature_columns = get_selected_feature_columns(
                    row=row,
                    all_feature_columns=X_train_all.columns,
                    feature_len=feature_len,
                )

                if len(selected_feature_columns) == 0:
                    continue

                feature_key = tuple(selected_feature_columns)

                if feature_key in seen_feature_sets:
                    continue

                seen_feature_sets.add(feature_key)

                feature_sets.append({
                    "Source_SFS_Row": s,
                    "#Features": len(selected_feature_columns),
                    "Selected_Features": selected_feature_columns,
                })

        logger.info(f"Unique feature sets found: {len(feature_sets)}")

        results_rows = []

        for fs_id, fs_info in enumerate(feature_sets, start=1):
            selected_feature_columns = fs_info["Selected_Features"]

            X_train = X_train_all[selected_feature_columns]

            if len(df_test) > 0:
                X_test = X_test_all[selected_feature_columns]
            else:
                X_test = pd.DataFrame(columns=selected_feature_columns)

            logger.info(
                f"[Feature set {fs_id}/{len(feature_sets)}] "
                f"#Features={len(selected_feature_columns)}"
            )

            for combo_id, (
                n_estimators,
                max_depth,
                learning_rate,
                num_leaves,
                subsample,
                colsample_bytree,
                min_child_samples,
                reg_alpha,
                reg_lambda,
            ) in enumerate(parameter_grid, start=1):

                lgbm_model = LGBMClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    learning_rate=float(learning_rate),
                    num_leaves=int(num_leaves),
                    subsample=float(subsample),
                    colsample_bytree=float(colsample_bytree),
                    min_child_samples=int(min_child_samples),
                    reg_alpha=float(reg_alpha),
                    reg_lambda=float(reg_lambda),
                    objective="binary",
                    random_state=random_state,
                    n_jobs=1,
                    verbose=-1,
                )

                y_true_all = []
                y_pred_all = []

                for train_idx, valid_idx in skf_cv.split(X_train, y_train):
                    x_tr = X_train.iloc[train_idx]
                    y_tr = y_train[train_idx]

                    x_val = X_train.iloc[valid_idx]
                    y_val = y_train[valid_idx]

                    lgbm_model.fit(x_tr, y_tr)
                    y_pred_cv = lgbm_model.predict(x_val)

                    y_true_all.extend(y_val)
                    y_pred_all.extend(y_pred_cv)

                cv_acc = accuracy_score(y_true_all, y_pred_all)

                lgbm_model.fit(X_train, y_train)

                if len(df_test) > 0:
                    y_pred_test = lgbm_model.predict(X_test)

                    test_acc = accuracy_score(y_test, y_pred_test)
                    sensitivity = get_sensitivity(y_test, y_pred_test)
                    specificity = get_specificity(y_test, y_pred_test)
                    cm = confusion_matrix(y_test, y_pred_test)

                    keep_row = (
                        sensitivity >= 0.5
                        and specificity > 0.5
                        and cv_acc >= 0.5
                    )
                else:
                    y_pred_test = []
                    test_acc = np.nan
                    sensitivity = np.nan
                    specificity = np.nan
                    cm = ""
                    keep_row = cv_acc >= 0.5

                if keep_row:
                    results_rows.append({
                        "CV_Accuracy": float(cv_acc),
                        "Test_Accuracy": test_acc,
                        "Sensitivity": sensitivity,
                        "Specificity": specificity,
                        "#Features": len(selected_feature_columns),
                        "n_estimators": int(n_estimators),
                        "max_depth": int(max_depth),
                        "learning_rate": float(learning_rate),
                        "num_leaves": int(num_leaves),
                        "subsample": float(subsample),
                        "colsample_bytree": float(colsample_bytree),
                        "min_child_samples": int(min_child_samples),
                        "reg_alpha": float(reg_alpha),
                        "reg_lambda": float(reg_lambda),
                        "CV_split": local_n_splits,
                        "Random_State": random_state,
                        "Source_SFS_Row": fs_info["Source_SFS_Row"],
                        "Selected_Features": str(selected_feature_columns),
                        "Confusion_Matrix": str(cm),
                        "y_pred": str(list(y_pred_test)),
                    })

            logger.info(
                f"Completed feature set {fs_id}/{len(feature_sets)} | "
                f"Current kept rows: {len(results_rows)}"
            )

            step2_results[sheet_name] = pd.DataFrame(results_rows)

            save_partial_results(
                step2_results=step2_results,
                output_excel_file=output_excel_file,
            )

            logger.info(f"Partial LightGBM Step 2 results saved to {output_excel_file}")

        step2_results[sheet_name] = pd.DataFrame(results_rows)

    logger.info(f"Saving final LightGBM Step 2 results to {output_excel_file}")

    save_partial_results(
        step2_results=step2_results,
        output_excel_file=output_excel_file,
    )

    logger.info("LightGBM Step 2 completed successfully.")
