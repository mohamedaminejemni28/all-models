"""
sequential_feature_selection_lightgbm.py

LightGBM version of Sequential Feature Selection.
Used by:
    pipelines_LightGBM/phase4_1_prepare_LightGBM_features.py
"""

import os
import warnings
import itertools
import random

import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from lightgbm import LGBMClassifier

from gaitml import logger


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*"
)


def get_numeric_feature_data(df, group_column="Group", participant_column="Participant"):
    """
    Select numeric feature columns only.
    Remove Group and Participant to avoid leakage.
    """
    columns_to_drop = [group_column]

    if participant_column in df.columns:
        columns_to_drop.append(participant_column)

    feature_df = df.drop(columns=columns_to_drop, errors="ignore")

    numeric_feature_df = feature_df.select_dtypes(include=[np.number])
    numeric_feature_df = numeric_feature_df.dropna(axis=1, how="all")
    numeric_feature_df = numeric_feature_df.fillna(numeric_feature_df.median())

    return numeric_feature_df


def save_partial_results(results_per_sheet, output_excel_file):
    """
    Save partial results after each LightGBM parameter combination.
    This avoids losing progress if the run is interrupted.
    """
    output_dir = os.path.dirname(output_excel_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_excel_file, engine="openpyxl") as writer:
        for sheet_name, df_result in results_per_sheet.items():
            if df_result.empty:
                continue

            safe_sheet_name = str(sheet_name)[:31]
            df_result.to_excel(writer, sheet_name=safe_sheet_name, index=False)


def sequential_feature_selection_lightgbm(
    input_excel_file,
    output_excel_file,
    group_column="Group",
    classes_dict=None,
    participant_column="Participant",
    feature_counts=None,
    n_estimators_set=None,
    max_depth_set=None,
    learning_rate_set=None,
    num_leaves_set=None,
    subsample_set=None,
    colsample_bytree_set=None,
    min_child_samples_set=None,
    reg_alpha_set=None,
    reg_lambda_set=None,
    kfold_n_splits=5,
    random_state=42,
    max_candidates=50,
    save_partial=True,
):
    """
    Run LightGBM-based Sequential Feature Selection.

    Each row in output:
    one LightGBM config + one feature subset.
    """

    if feature_counts is None:
        feature_counts = list(range(5, 20))

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

    feature_counts = sorted([int(count) for count in feature_counts])
    requested_max_features = max(feature_counts)

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

    logger.info(f"Loading training Excel file: {input_excel_file}")
    logger.info(f"Output file: {output_excel_file}")
    logger.info(f"Feature counts requested: {feature_counts}")
    logger.info(f"Parameter combinations before sampling: {total_before_sampling}")
    logger.info(f"Parameter combinations used: {len(parameter_grid)}")
    logger.info(f"Maximum SFS features: {requested_max_features}")

    excel_file = pd.ExcelFile(input_excel_file)
    results_per_sheet = {}

    for sheet_name in excel_file.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")

        df = pd.read_excel(input_excel_file, sheet_name=sheet_name)

        if group_column not in df.columns:
            raise ValueError(
                f"Column '{group_column}' not found in sheet '{sheet_name}'"
            )

        df[group_column] = df[group_column].astype(str).str.strip()

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[group_column])

        if len(np.unique(y)) < 2:
            logger.warning(f"Sheet {sheet_name} has fewer than two classes. Skipping.")
            continue

        X_df = get_numeric_feature_data(
            df=df,
            group_column=group_column,
            participant_column=participant_column,
        )

        if X_df.shape[1] == 0:
            raise ValueError(f"No numeric features found in sheet '{sheet_name}'.")

        feature_names_all = X_df.columns.astype(str).tolist()

        local_max_features = min(requested_max_features, X_df.shape[1])
        local_feature_counts = [
            count for count in feature_counts if count <= local_max_features
        ]

        if not local_feature_counts:
            raise ValueError(
                f"No valid feature counts for sheet {sheet_name}. "
                f"Available features: {X_df.shape[1]}"
            )

        logger.info(f"Number of numeric features: {X_df.shape[1]}")
        logger.info(f"Local feature counts: {local_feature_counts}")
        logger.info(f"SFS will select up to {local_max_features} features.")

        skf_cv = StratifiedKFold(
            n_splits=kfold_n_splits,
            shuffle=True,
            random_state=random_state,
        )

        sheet_results = []
        results_per_sheet[sheet_name] = pd.DataFrame()

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

            logger.info(
                f"[{combo_id}/{len(parameter_grid)}] Running LightGBM SFS | "
                f"n_estimators={n_estimators}, "
                f"max_depth={max_depth}, "
                f"learning_rate={learning_rate}, "
                f"num_leaves={num_leaves}, "
                f"subsample={subsample}, "
                f"colsample_bytree={colsample_bytree}, "
                f"min_child_samples={min_child_samples}, "
                f"reg_alpha={reg_alpha}, "
                f"reg_lambda={reg_lambda}"
            )

            base_model = LGBMClassifier(
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

            sfs = SFS(
                estimator=base_model,
                k_features=local_max_features,
                forward=True,
                floating=False,
                verbose=1,
                scoring="accuracy",
                cv=skf_cv,
                n_jobs=1,
            )

            sfs.fit(X_df, y)

            sfs_results = sfs.get_metric_dict()

            for feature_count in local_feature_counts:
                if feature_count not in sfs_results:
                    logger.warning(f"SFS result for {feature_count} features not found.")
                    continue

                selected_indices = list(sfs_results[feature_count]["feature_idx"])
                selected_indices = [int(index) for index in selected_indices]

                selected_feature_names = [
                    feature_names_all[index] for index in selected_indices
                ]

                X_selected_df = X_df.iloc[:, selected_indices]

                y_true_all = []
                y_pred_all = []

                for train_idx, valid_idx in skf_cv.split(X_selected_df, y):
                    X_train = X_selected_df.iloc[train_idx]
                    X_valid = X_selected_df.iloc[valid_idx]

                    y_train = y[train_idx]
                    y_valid = y[valid_idx]

                    fold_model = clone(base_model)
                    fold_model.fit(X_train, y_train)

                    y_pred = fold_model.predict(X_valid)

                    y_true_all.extend(y_valid)
                    y_pred_all.extend(y_pred)

                cv_accuracy = accuracy_score(y_true_all, y_pred_all)

                row = {
                    "Dataset_Sheet": sheet_name,
                    "Model": "LightGBM",
                    "CV_Accuracy": float(cv_accuracy),
                    "n_estimators": int(n_estimators),
                    "max_depth": int(max_depth),
                    "learning_rate": float(learning_rate),
                    "num_leaves": int(num_leaves),
                    "subsample": float(subsample),
                    "colsample_bytree": float(colsample_bytree),
                    "min_child_samples": int(min_child_samples),
                    "reg_alpha": float(reg_alpha),
                    "reg_lambda": float(reg_lambda),
                    "#Features": int(feature_count),
                    "Features": str(selected_indices),
                    "Feature_Names": str(selected_feature_names),
                    "Ordered Indices": str(selected_indices),
                }

                sheet_results.append(row)

                logger.info(
                    f"Sheet={sheet_name} | "
                    f"#Features={feature_count} | "
                    f"CV_Accuracy={cv_accuracy:.4f}"
                )

            results_per_sheet[sheet_name] = pd.DataFrame(sheet_results)

            if save_partial:
                save_partial_results(
                    results_per_sheet=results_per_sheet,
                    output_excel_file=output_excel_file,
                )

                logger.info(
                    f"Partial LightGBM SFS results saved after combo "
                    f"{combo_id}/{len(parameter_grid)}"
                )

    logger.info(f"Saving final LightGBM SFS results to: {output_excel_file}")

    save_partial_results(
        results_per_sheet=results_per_sheet,
        output_excel_file=output_excel_file,
    )

    logger.info("LightGBM SFS completed successfully.")
