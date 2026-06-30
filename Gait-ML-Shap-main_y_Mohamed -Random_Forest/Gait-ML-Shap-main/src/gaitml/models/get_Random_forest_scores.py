import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# HELPERS
# ============================================================

def normalize_column_names(df):
    """
    Normalize possible column name variations.
    """
    df = df.copy()

    rename_map = {}

    # CV accuracy
    if "CV_Accuracy" in df.columns and "CV_accuracy" not in df.columns:
        rename_map["CV_Accuracy"] = "CV_accuracy"

    if "CV Accuracy" in df.columns and "CV_accuracy" not in df.columns:
        rename_map["CV Accuracy"] = "CV_accuracy"

    # Test accuracy
    if "Test_accuracy" in df.columns and "Test_Accuracy" not in df.columns:
        rename_map["Test_accuracy"] = "Test_Accuracy"

    if "Test Accuracy" in df.columns and "Test_Accuracy" not in df.columns:
        rename_map["Test Accuracy"] = "Test_Accuracy"

    # Ordered feature indices
    if "Ordered Indices" not in df.columns and "Feature_idx" in df.columns:
        rename_map["Feature_idx"] = "Ordered Indices"

    if "Ordered Indices" not in df.columns and "Selected Indices" in df.columns:
        rename_map["Selected Indices"] = "Ordered Indices"

    # SFS CV accuracy
    if "SFS_CV_Accuracy" not in df.columns and "SFS CV Accuracy" in df.columns:
        rename_map["SFS CV Accuracy"] = "SFS_CV_Accuracy"

    df = df.rename(columns=rename_map)

    return df


def ensure_required_columns(df, required_columns):
    """
    Add missing columns with empty values so the final Excel is consistent.
    """
    df = df.copy()

    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    return df


def convert_numeric_columns(df):
    """
    Convert metric columns to numeric values.
    """
    df = df.copy()

    numeric_columns = [
        "SFS_CV_Accuracy",
        "#Features",
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "CV_split",
        "Random_State",
        "CV_accuracy",
        "Test_Accuracy",
        "Specificity",
        "Sensitivity",
        "NPV",
        "PPV",
        "Likelihood_Ratio",
        "F1",
        "MCC"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def sort_scores(df):
    """
    Sort Random Forest scores using the same ranking logic:
        1. CV accuracy
        2. Test accuracy
        3. Sensitivity
    """
    df = df.copy()
    df = convert_numeric_columns(df)

    sort_columns = []
    ascending = []

    if "CV_accuracy" in df.columns:
        sort_columns.append("CV_accuracy")
        ascending.append(False)

    if "Test_Accuracy" in df.columns:
        sort_columns.append("Test_Accuracy")
        ascending.append(False)

    if "Sensitivity" in df.columns:
        sort_columns.append("Sensitivity")
        ascending.append(False)

    if len(sort_columns) > 0:
        df = df.sort_values(
            by=sort_columns,
            ascending=ascending,
            na_position="last"
        )

    return df


# ============================================================
# MAIN FUNCTION
# ============================================================

def get_Random_forest_scores(
    sfs_results_file,
    combination_results_file,
    output_file,
    logger=None
):
    """
    Phase 4.3 Random Forest Scores.

    This function reads the Phase 4.2 Random Forest combination results
    and creates the final Random Forest score file.

    The Phase 4.2 file already contains:
        - selected features
        - Random Forest hyperparameters
        - CV accuracy
        - test accuracy
        - sensitivity
        - specificity
        - F1
        - MCC
        - confusion matrix
    """

    if logger:
        logger.info("Starting Random Forest scores generation")
        logger.info(f"SFS results file: {sfs_results_file}")
        logger.info(f"Combination results file: {combination_results_file}")
        logger.info(f"Output file: {output_file}")

    if not os.path.exists(combination_results_file):
        raise FileNotFoundError(
            f"Combination results file not found: {combination_results_file}"
        )

    if not os.path.exists(sfs_results_file):
        if logger:
            logger.warning(
                f"SFS results file not found: {sfs_results_file}. "
                "Scores will be generated using combination results only."
            )

    # ========================================================
    # READ PHASE 4.2 RESULTS
    # ========================================================

    combination_df = pd.read_excel(combination_results_file)
    combination_df = normalize_column_names(combination_df)

    if logger:
        logger.info(f"Loaded combination results: {combination_df.shape}")

    # ========================================================
    # OPTIONAL: READ SFS RESULTS ONLY FOR LOGGING
    # ========================================================

    if os.path.exists(sfs_results_file):
        try:
            sfs_excel = pd.ExcelFile(sfs_results_file)

            if logger:
                logger.info(f"Loaded SFS file with sheets: {sfs_excel.sheet_names}")

        except Exception as e:
            if logger:
                logger.warning(f"Could not read SFS file: {e}")

    # ========================================================
    # FINAL COLUMN ORDER
    # ========================================================

    final_columns = [
        "Name_Model",
        "Sheet",

        "SFS_CV_Accuracy",

        "#Features",
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",

        "CV_split",
        "Random_State",

        "Ordered Indices",
        "Model_Type",

        "CV_accuracy",
        "Test_Accuracy",
        "Specificity",
        "Sensitivity",
        "NPV",
        "PPV",
        "Likelihood_Ratio",
        "F1",
        "MCC",
        "Confusion_Matrix",

        "Features"
    ]

    combination_df = ensure_required_columns(
        combination_df,
        final_columns
    )

    scores_df = combination_df[final_columns]

    # ========================================================
    # SORT FINAL SCORES
    # ========================================================

    scores_df = sort_scores(scores_df)

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    output_dir = os.path.dirname(output_file)

    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    scores_df.to_excel(output_file, index=False)

    if logger:
        logger.info(f"Random Forest scores saved to: {output_file}")
        logger.info("Completed Random Forest scores generation")