"""
get_CatBoost_Scores.py

This script performs Phase 4.3: Scores CatBoost.

It reads:
    1) CatBoost SFS results from Phase 4.1
    2) CatBoost combination results from Phase 4.2

Then it calculates:
    Specificity
    Sensitivity
    NPV
    PPV
    Likelihood_Ratio
    F1
    MCC

And saves:
    1) Full CatBoost scores file
    2) Top 3 CatBoost scores file
"""

import os
import ast
import json
import logging

import numpy as np
import pandas as pd


# ============================================================
# Metric helpers
# ============================================================

def parse_confusion_matrix(value):
    """
    Parse confusion matrix saved as string.

    Supports:
        "[[10, 2], [1, 8]]"
        "array([[10, 2], [1, 8]])"
    """
    if isinstance(value, np.ndarray):
        return value

    if isinstance(value, list):
        return np.array(value)

    if pd.isna(value):
        return np.array([])

    value = str(value).strip()

    if value == "":
        return np.array([])

    value = value.replace("array(", "").replace(")", "")

    try:
        matrix = ast.literal_eval(value)
        return np.array(matrix)
    except Exception:
        pass

    try:
        matrix = json.loads(value)
        return np.array(matrix)
    except Exception:
        pass

    return np.array([])


def get_specificity_from_cm(tn, fp, fn, tp):
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def get_npv_from_cm(tn, fn):
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0


def get_ppv_from_cm(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def get_f1_from_cm(tp, fp, fn, tn):
    return (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) > 0 else 0.0


def get_MCC_from_cm(tn, fp, fn, tp):
    denominator = np.sqrt(
        (tp + fp) *
        (tp + fn) *
        (tn + fp) *
        (tn + fn)
    )

    if denominator == 0:
        return 0.0

    return ((tp * tn) - (fp * fn)) / denominator


def get_likelihood_ratios_from_cm(tp, fp, fn, tn):
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    false_positive_rate = 1 - specificity

    if false_positive_rate == 0:
        return np.inf

    return sensitivity / false_positive_rate


def parse_list(value):
    """
    Parse feature lists from Excel.
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


def get_features_for_row(df_step1, df_step2, counter):
    """
    Get feature list for one model.

    Priority:
        1. Feature_Names column from Phase 4.2
        2. Feature_Names column from Phase 4.1 using Feature_idx
        3. Last column from Phase 4.1 using Feature_idx
    """
    if "Feature_Names" in df_step2.columns:
        features = parse_list(df_step2.loc[counter, "Feature_Names"])
        if len(features) > 0:
            return features

    feature_idx = int(df_step2.loc[counter, "Feature_idx"])
    feature_count = int(df_step2.loc[counter, "#Features"])

    if "Feature_Names" in df_step1.columns:
        features = parse_list(df_step1.loc[feature_idx, "Feature_Names"])
        return features[:feature_count]

    features = parse_list(df_step1.iloc[feature_idx, -1])
    return features[:feature_count]


def safe_get(row, column, default=np.nan):
    if column in row.index:
        return row[column]

    return default


# ============================================================
# Main scoring function
# ============================================================

def get_CatBoost_scores(
    sfs_results_file,
    combination_results_file,
    output_file="Scores_CatBoost.xlsx",
    output_top3_file=None,
    top3_feature_counts=None,
    logger=None,
):
    """
    Calculate comprehensive performance metrics for CatBoost models.
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    if top3_feature_counts is None:
        top3_feature_counts = [5, 6, 10]

    logger.info(f"Loading CatBoost SFS results from: {sfs_results_file}")
    logger.info(f"Loading CatBoost combination results from: {combination_results_file}")

    excel_step1 = pd.ExcelFile(sfs_results_file)
    excel_step2 = pd.ExcelFile(combination_results_file)

    score_catboost = {}
    top3_results = {}

    for sheet_name in excel_step2.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")

        if sheet_name not in excel_step1.sheet_names:
            logger.warning(
                f"Sheet {sheet_name} not found in SFS file. "
                "Continuing using Phase 4.2 Feature_Names only."
            )
            df_step1 = pd.DataFrame()
        else:
            df_step1 = pd.read_excel(excel_step1, sheet_name=sheet_name)

        df_step2 = pd.read_excel(excel_step2, sheet_name=sheet_name)

        # Remove Excel index columns if present
        unnamed_cols = [
            col for col in df_step2.columns
            if str(col).startswith("Unnamed")
        ]

        if unnamed_cols:
            df_step2 = df_step2.drop(columns=unnamed_cols)

        if df_step2.empty:
            logger.warning(f"No CatBoost combinations found for sheet: {sheet_name}")
            score_catboost[sheet_name] = pd.DataFrame()
            continue

        required_columns = [
            "Feature_idx",
            "#Features",
            "Confusion_Matrix",
            "CV_Accuracy",
            "Test_Accuracy",
            "Sensitivity",
            "iterations",
            "depth",
            "learning_rate",
            "l2_leaf_reg",
            "bagging_temperature",
            "random_strength",
            "CV_split",
            "Random_State",
        ]

        for col in required_columns:
            if col not in df_step2.columns:
                raise ValueError(
                    f"Column '{col}' not found in sheet '{sheet_name}' "
                    f"of {combination_results_file}"
                )

        rows = []

        for counter in range(len(df_step2)):
            row_step2 = df_step2.loc[counter]

            feature_idx = int(row_step2["Feature_idx"])
            feature_count = int(row_step2["#Features"])

            features = get_features_for_row(
                df_step1=df_step1,
                df_step2=df_step2,
                counter=counter,
            )

            matrix = parse_confusion_matrix(row_step2["Confusion_Matrix"])

            if matrix.shape != (2, 2):
                logger.warning(
                    f"Confusion matrix is not 2x2 for sheet {sheet_name}, "
                    f"row {counter}. Metrics will be NaN."
                )

                specificity_val = np.nan
                npv_val = np.nan
                ppv_val = np.nan
                mcc_val = np.nan
                f1_val = np.nan
                likelihood_ratio_val = np.nan
            else:
                tn, fp, fn, tp = matrix.ravel()

                specificity_val = get_specificity_from_cm(tn, fp, fn, tp)
                npv_val = get_npv_from_cm(tn, fn)
                ppv_val = get_ppv_from_cm(tp, fp)
                mcc_val = get_MCC_from_cm(tn, fp, fn, tp)
                f1_val = get_f1_from_cm(tp, fp, fn, tn)
                likelihood_ratio_val = get_likelihood_ratios_from_cm(tp, fp, fn, tn)

            name_model = (
                f"CatBoost_IT{row_step2['iterations']}_"
                f"D{row_step2['depth']}_"
                f"LR{row_step2['learning_rate']}_"
                f"L2{row_step2['l2_leaf_reg']}_"
                f"F{feature_count}"
            )

            rows.append({
                "Name_Model": name_model,
                "#Features": feature_count,
                "iterations": row_step2["iterations"],
                "depth": row_step2["depth"],
                "learning_rate": row_step2["learning_rate"],
                "l2_leaf_reg": row_step2["l2_leaf_reg"],
                "bagging_temperature": row_step2["bagging_temperature"],
                "random_strength": row_step2["random_strength"],
                "CV_split": row_step2["CV_split"],
                "Random_State": row_step2["Random_State"],
                "Feature_idx": feature_idx,
                "Model_Type": "CatBoost",
                "CV_accuracy": row_step2["CV_Accuracy"],
                "CV_Accuracy": row_step2["CV_Accuracy"],
                "Test_Accuracy": row_step2["Test_Accuracy"],
                "Specificity": specificity_val,
                "Sensitivity": row_step2["Sensitivity"],
                "NPV": npv_val,
                "PPV": ppv_val,
                "Likelihood_Ratio": likelihood_ratio_val,
                "F1": f1_val,
                "MCC": mcc_val,
                "Confusion_Matrix": row_step2["Confusion_Matrix"],
                "Features": str(features),
                "Feature_Names": json.dumps([str(feature) for feature in features]),
            })

        df_results = pd.DataFrame(rows)

        if not df_results.empty:
            numeric_columns = [
                "CV_accuracy",
                "CV_Accuracy",
                "Test_Accuracy",
                "Specificity",
                "Sensitivity",
                "NPV",
                "PPV",
                "F1",
                "MCC",
                "#Features",
            ]

            for col in numeric_columns:
                if col in df_results.columns:
                    df_results[col] = pd.to_numeric(df_results[col], errors="coerce")

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

        score_catboost[sheet_name] = df_results

        top3_results[f"{sheet_name}_Top3Overall"] = df_results.head(3).copy()

        for feature_count in top3_feature_counts:
            top3_feature_df = df_results[
                df_results["#Features"].astype(int) == int(feature_count)
            ].head(3).copy()

            if not top3_feature_df.empty:
                sheet_top3_name = f"{sheet_name}_Top3_{feature_count}F"
                top3_results[sheet_top3_name] = top3_feature_df

    logger.info(f"Saving CatBoost scores to: {output_file}")

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df_results in score_catboost.items():
            safe_sheet_name = str(sheet_name)[:31]
            df_results.to_excel(writer, sheet_name=safe_sheet_name, index=False)

    if output_top3_file is not None:
        logger.info(f"Saving CatBoost Top 3 scores to: {output_top3_file}")

        top3_output_dir = os.path.dirname(output_top3_file)

        if top3_output_dir:
            os.makedirs(top3_output_dir, exist_ok=True)

        with pd.ExcelWriter(output_top3_file, engine="openpyxl") as writer:
            for sheet_name, df_results in top3_results.items():
                safe_sheet_name = str(sheet_name)[:31]
                df_results.to_excel(writer, sheet_name=safe_sheet_name, index=False)

    logger.info("CatBoost scores calculation completed successfully.")

    return score_catboost


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 4.3: Scores CatBoost - Calculate metrics for CatBoost models."
    )

    parser.add_argument(
        "--sfs",
        required=True,
        help="Path to CatBoost SFS results Excel file from Phase 4.1",
    )

    parser.add_argument(
        "--combinations",
        required=True,
        help="Path to CatBoost combination results Excel file from Phase 4.2",
    )

    parser.add_argument(
        "--output",
        default="Scores_CatBoost.xlsx",
        help="Output Excel file path",
    )

    parser.add_argument(
        "--top3",
        default=None,
        help="Optional Top 3 output Excel file path",
    )

    args = parser.parse_args()

    get_CatBoost_scores(
        sfs_results_file=args.sfs,
        combination_results_file=args.combinations,
        output_file=args.output,
        output_top3_file=args.top3,
    )
