"""
sequential_feature_selection_random_forest.py

Random Forest version of Sequential Feature Selection.

Output format is similar to XGBoost:
    CV_Accuracy
    n_estimators
    max_depth
    learning_rate
    #Features
    Ordered Indices
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from gaitml import logger


def sequential_feature_selection_random_forest(
    input_excel_file,
    output_excel_file,
    group_column="Group",
    classes_dict=None,
    n_features=19,
    n_estimators_set=None,
    max_depth_set=None,
    learning_rate_set=None,
    kfold_n_splits=10,
    random_state=0
):
    if n_estimators_set is None:
        n_estimators_set = [50, 100]

    if max_depth_set is None:
        max_depth_set = [2, 3]

    if learning_rate_set is None:
        learning_rate_set = [None]

    logger.info(f"Loading training Excel file: {input_excel_file}")
    logger.info("Random Forest does not use learning_rate. Column kept only for same output format as XGBoost.")

    excel_file = pd.ExcelFile(input_excel_file)
    results_per_sheet = {}

    for sheet_name in excel_file.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")

        df = pd.read_excel(input_excel_file, sheet_name=sheet_name)

        if group_column not in df.columns:
            raise ValueError(
                f"Column '{group_column}' not found in sheet '{sheet_name}'"
            )

        # Clean labels
        df[group_column] = df[group_column].astype(str).str.strip()

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[group_column])

        # Same feature segment as XGBoost/SVM pipeline
        X_df = df.loc[:, "Pelv_Angle_Y_MAX_SW":"Hip_Angle_Z_OHS"]

        X_df = X_df.apply(pd.to_numeric, errors="coerce")
        X_df = X_df.dropna(axis=1, how="all")

        X = X_df.values

        logger.info(f"Number of samples: {X.shape[0]}")
        logger.info(f"Number of features: {X.shape[1]}")
        logger.info(f"SFS max features: {n_features}")

        skf_cv = StratifiedKFold(
            n_splits=kfold_n_splits,
            shuffle=True,
            random_state=random_state
        )

        df_results = pd.DataFrame(
            columns=[
                "CV_Accuracy",
                "n_estimators",
                "max_depth",
                "learning_rate",
                "#Features",
                "Ordered Indices"
            ]
        )

        for n_estimators in n_estimators_set:
            for max_depth in max_depth_set:
                for learning_rate in learning_rate_set:

                    logger.info(
                        f"Running Random Forest SFS | "
                        f"n_estimators={n_estimators}, "
                        f"max_depth={max_depth}, "
                        f"learning_rate={learning_rate}"
                    )

                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        class_weight="balanced",
                        n_jobs=1
                    )

                    pipeline = Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("classifier", model)
                        ]
                    )

                    sfs = SFS(
                        estimator=pipeline,
                        k_features=n_features,
                        forward=True,
                        floating=False,
                        verbose=0,
                        scoring="accuracy",
                        cv=skf_cv,
                        n_jobs=-1
                    )

                    sfs.fit(X, y)

                    sfs_results = sfs.get_metric_dict()

                    # ====================================================
                    # Save many rows: 5, 6, 7, ..., n_features
                    # ====================================================

                    for feature_count in range(5, n_features + 1):

                        if feature_count not in sfs_results:
                            continue

                        feature_indices = list(sfs_results[feature_count]["feature_idx"])
                        cv_accuracy = sfs_results[feature_count]["avg_score"]

                        new_row = pd.DataFrame(
                            [[
                                cv_accuracy,
                                n_estimators,
                                max_depth,
                                learning_rate,
                                feature_count,
                                feature_indices
                            ]],
                            columns=[
                                "CV_Accuracy",
                                "n_estimators",
                                "max_depth",
                                "learning_rate",
                                "#Features",
                                "Ordered Indices"
                            ]
                        )

                        df_results = pd.concat(
                            [df_results, new_row],
                            ignore_index=True
                        )

                        logger.info(
                            f"Sheet={sheet_name} | "
                            f"CV_Accuracy={cv_accuracy:.4f} | "
                            f"#Features={feature_count} | "
                            f"n_estimators={n_estimators} | "
                            f"max_depth={max_depth}"
                        )

        results_per_sheet[sheet_name] = df_results

    logger.info(f"Saving Random Forest SFS results to: {output_excel_file}")

    with pd.ExcelWriter(output_excel_file) as writer:
        for sheet_name, df_result in results_per_sheet.items():
            df_result.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info("Random Forest SFS completed successfully.")