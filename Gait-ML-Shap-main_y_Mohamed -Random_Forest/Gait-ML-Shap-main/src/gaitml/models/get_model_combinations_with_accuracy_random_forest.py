"""
get_model_combinations_with_accuracy_random_forest.py

Random Forest version of Phase 4.2 Model Combinations.

It reads the Phase 4.1 SFS output with:
    CV_Accuracy
    n_estimators
    max_depth
    learning_rate
    #Features
    Ordered Indices

Then it tests multiple Random Forest parameter combinations.
"""

import os
import ast
from itertools import product

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


# ============================================================
# HELPERS
# ============================================================

def encode_train_test_labels(y_train, y_test):
    """
    Encode train and test labels together to keep the same label mapping.
    Example:
        Control -> 0
        Autism  -> 1
    """

    y_train = y_train.astype(str).str.strip()
    y_test = y_test.astype(str).str.strip()

    encoder = LabelEncoder()
    encoder.fit(pd.concat([y_train, y_test], axis=0))

    return encoder.transform(y_train), encoder.transform(y_test)


def parse_ordered_indices(value):
    """
    Parse Ordered Indices column.

    Examples:
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

    return [int(v) for v in value.split() if v.strip().isdigit()]


def build_random_forest_pipeline(
    n_estimators,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    max_features,
    random_state
):
    """
    Build a Random Forest pipeline.

    The imputer replaces missing values using the median.
    """

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

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", model)
        ]
    )

    return pipeline


def get_valid_cv(y, requested_splits, random_state):
    """
    Create safe StratifiedKFold.

    If one class has fewer samples than requested_splits,
    the function automatically reduces the number of folds.
    """

    _, counts = np.unique(y, return_counts=True)
    min_class_count = counts.min()

    n_splits = min(requested_splits, min_class_count)

    if n_splits < 2:
        return None

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )


def calculate_metrics(y_true, y_pred):
    """
    Calculate binary classification metrics.
    """

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    test_accuracy = accuracy_score(y_true, y_pred)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    likelihood_ratio = (
        sensitivity / (1 - specificity)
        if (1 - specificity) > 0
        else 0
    )

    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "Test_Accuracy": test_accuracy,
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "NPV": npv,
        "PPV": ppv,
        "Likelihood_Ratio": likelihood_ratio,
        "F1": f1,
        "MCC": mcc,
        "Confusion_Matrix": str(cm)
    }


# ============================================================
# MAIN FUNCTION
# ============================================================

def get_combination_list_with_accuracy_random_forest(
    train_excel_file,
    test_excel_file,
    sfs_results_excel_file,
    output_excel_file,
    group_column,

    feature_range=(19, 19),

    n_estimators_set=None,
    max_depth_set=None,
    min_samples_split_set=None,
    min_samples_leaf_set=None,
    max_features_set=None,

    n_splits=10,
    random_state=0,
    logger=None
):
    """
    Phase 4.2 Random Forest model combinations.
    """

    if n_estimators_set is None:
        n_estimators_set = [50, 100, 150]

    if max_depth_set is None:
        max_depth_set = [2, 3, 4]

    if min_samples_split_set is None:
        min_samples_split_set = [2, 5]

    if min_samples_leaf_set is None:
        min_samples_leaf_set = [1, 2]

    if max_features_set is None:
        max_features_set = ["sqrt", "log2"]

    if logger:
        logger.info("Starting Random Forest model combinations")
        logger.info(f"Train file: {train_excel_file}")
        logger.info(f"Test file: {test_excel_file}")
        logger.info(f"SFS file: {sfs_results_excel_file}")
        logger.info(f"Output file: {output_excel_file}")
        logger.info(f"Feature range: {feature_range}")
        logger.info(f"n_estimators_set = {n_estimators_set}")
        logger.info(f"max_depth_set = {max_depth_set}")
        logger.info(f"min_samples_split_set = {min_samples_split_set}")
        logger.info(f"min_samples_leaf_set = {min_samples_leaf_set}")
        logger.info(f"max_features_set = {max_features_set}")
        logger.info(f"CV splits = {n_splits}")

    if not os.path.exists(train_excel_file):
        raise FileNotFoundError(f"Train file not found: {train_excel_file}")

    if not os.path.exists(test_excel_file):
        raise FileNotFoundError(f"Test file not found: {test_excel_file}")

    if not os.path.exists(sfs_results_excel_file):
        raise FileNotFoundError(f"SFS results file not found: {sfs_results_excel_file}")

    train_excel = pd.ExcelFile(train_excel_file)
    test_excel = pd.ExcelFile(test_excel_file)
    sfs_excel = pd.ExcelFile(sfs_results_excel_file)

    all_results = []
    model_counter = 1

    for sheet_name in sfs_excel.sheet_names:

        if logger:
            logger.info(f"Processing sheet: {sheet_name}")

        if sheet_name not in train_excel.sheet_names:
            if logger:
                logger.warning(f"Sheet {sheet_name} not found in train file. Skipping.")
            continue

        if sheet_name not in test_excel.sheet_names:
            if logger:
                logger.warning(f"Sheet {sheet_name} not found in test file. Skipping.")
            continue

        sfs_df = pd.read_excel(sfs_results_excel_file, sheet_name=sheet_name)
        train_df = pd.read_excel(train_excel_file, sheet_name=sheet_name)
        test_df = pd.read_excel(test_excel_file, sheet_name=sheet_name)

        if group_column not in train_df.columns or group_column not in test_df.columns:
            if logger:
                logger.warning(
                    f"Group column '{group_column}' not found in sheet '{sheet_name}'. Skipping."
                )
            continue

        # Same feature range as Phase 4.1 / XGBoost / SVM
        X_train_df = train_df.loc[:, "Pelv_Angle_Y_MAX_SW":"Hip_Angle_Z_OHS"]
        X_test_df = test_df.loc[:, "Pelv_Angle_Y_MAX_SW":"Hip_Angle_Z_OHS"]

        X_train_df = X_train_df.apply(pd.to_numeric, errors="coerce")
        X_test_df = X_test_df.apply(pd.to_numeric, errors="coerce")

        common_features = [
            col for col in X_train_df.columns
            if col in X_test_df.columns
        ]

        X_train_df = X_train_df[common_features]
        X_test_df = X_test_df[common_features]

        feature_columns = list(X_train_df.columns)

        y_train, y_test = encode_train_test_labels(
            train_df[group_column],
            test_df[group_column]
        )

        for _, sfs_row in sfs_df.iterrows():

            sfs_num_features = int(sfs_row.get("#Features", 0))

            # This is where we keep only 19-feature rows
            if not (feature_range[0] <= sfs_num_features <= feature_range[1]):
                continue

            if "Ordered Indices" not in sfs_row:
                if logger:
                    logger.warning(
                        "Column 'Ordered Indices' not found in SFS file. Skipping row."
                    )
                continue

            ordered_indices = parse_ordered_indices(sfs_row["Ordered Indices"])

            selected_features = [
                feature_columns[i]
                for i in ordered_indices
                if i < len(feature_columns)
            ]

            if len(selected_features) == 0:
                if logger:
                    logger.warning("No selected features found. Skipping row.")
                continue

            X_train_selected = X_train_df[selected_features]
            X_test_selected = X_test_df[selected_features]

            for (
                n_estimators,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                max_features
            ) in product(
                n_estimators_set,
                max_depth_set,
                min_samples_split_set,
                min_samples_leaf_set,
                max_features_set
            ):

                model_name = f"{model_counter}"

                if logger:
                    logger.info(
                        f"Running {model_name} | "
                        f"Sheet={sheet_name}, "
                        f"#Features={len(selected_features)}, "
                        f"n_estimators={n_estimators}, "
                        f"max_depth={max_depth}, "
                        f"min_samples_split={min_samples_split}, "
                        f"min_samples_leaf={min_samples_leaf}, "
                        f"max_features={max_features}"
                    )

                try:
                    pipeline = build_random_forest_pipeline(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        random_state=random_state
                    )

                    cv = get_valid_cv(
                        y=y_train,
                        requested_splits=n_splits,
                        random_state=random_state
                    )

                    if cv is not None:
                        cv_scores = cross_val_score(
                            pipeline,
                            X_train_selected,
                            y_train,
                            scoring="accuracy",
                            cv=cv,
                            n_jobs=-1
                        )

                        cv_accuracy = float(np.mean(cv_scores))
                        cv_split_used = cv.get_n_splits()
                    else:
                        cv_accuracy = np.nan
                        cv_split_used = 0

                    pipeline.fit(X_train_selected, y_train)
                    y_pred = pipeline.predict(X_test_selected)

                    metrics = calculate_metrics(y_test, y_pred)

                    result = {
                        "Name_Model": model_name,
                        "Sheet": sheet_name,

                        "SFS_CV_Accuracy": sfs_row.get("CV_Accuracy", np.nan),

                        "#Features": len(selected_features),
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf,
                        "max_features": max_features,

                        "CV_split": cv_split_used,
                        "Random_State": random_state,

                        "Ordered Indices": str(ordered_indices),
                        "Model_Type": "RandomForest",

                        "CV_accuracy": cv_accuracy,
                        "Test_Accuracy": metrics["Test_Accuracy"],
                        "Specificity": metrics["Specificity"],
                        "Sensitivity": metrics["Sensitivity"],
                        "NPV": metrics["NPV"],
                        "PPV": metrics["PPV"],
                        "Likelihood_Ratio": metrics["Likelihood_Ratio"],
                        "F1": metrics["F1"],
                        "MCC": metrics["MCC"],
                        "Confusion_Matrix": metrics["Confusion_Matrix"],

                        "Features": str(ordered_indices)
                    }

                    all_results.append(result)

                    if logger:
                        logger.info(
                            f"{model_name} completed | "
                            f"CV={cv_accuracy:.4f} | "
                            f"Test={metrics['Test_Accuracy']:.4f}"
                        )

                except Exception as e:
                    if logger:
                        logger.error(f"Error in {model_name}: {e}")

                model_counter += 1

    if len(all_results) == 0:
        if logger:
            logger.warning("No Random Forest combination results generated.")
            logger.warning(
                "Check that your SFS file contains rows with #Features = 19."
            )
        return

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by=["CV_accuracy", "Test_Accuracy", "Sensitivity"],
        ascending=False
    )

    output_dir = os.path.dirname(output_excel_file)

    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    results_df.to_excel(output_excel_file, index=False)

    if logger:
        logger.info(f"Random Forest combination results saved to: {output_excel_file}")
        logger.info("Completed Random Forest model combinations")