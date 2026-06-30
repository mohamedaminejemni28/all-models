"""
get_LightGBM_scores.py

Phase 4.3: LightGBM scores selection.

This file reads Phase 4.2 LightGBM model-combination results and creates:

1. Full ranked scores file
2. Top 3 LightGBM models file

Used by:
    pipelines_LightGBM/phase4_3_scores_LightGBM.py
"""

import os
import re
import pandas as pd
import numpy as np


def make_safe_sheet_name(name, used_names):
    """
    Excel sheet names must be <= 31 characters and cannot contain:
    : \ / ? * [ ]
    """
    name = str(name)
    name = re.sub(r"[:\\/?*\[\]]", "_", name)
    name = name[:31]

    original_name = name
    counter = 1

    while name in used_names:
        suffix = f"_{counter}"
        name = original_name[:31 - len(suffix)] + suffix
        counter += 1

    used_names.add(name)
    return name


def clean_lightgbm_scores_dataframe(df):
    """
    Clean LightGBM Phase 4.2 dataframe.
    """
    df = df.copy()

    # Remove Excel automatic index columns
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    numeric_columns = [
        "CV_Accuracy",
        "Test_Accuracy",
        "Sensitivity",
        "Specificity",
        "#Features",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "num_leaves",
        "subsample",
        "colsample_bytree",
        "min_child_samples",
        "reg_alpha",
        "reg_lambda",
        "CV_split",
        "Random_State",
        "Source_SFS_Row",
        "Feature_idx",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def rank_lightgbm_scores(df):
    """
    Rank LightGBM models.

    Ranking priority:
        1. CV_Accuracy
        2. Test_Accuracy
        3. Sensitivity
        4. Specificity
        5. Fewer features
    """
    df = df.copy()

    if df.empty:
        return df

    sort_columns = []
    ascending_values = []

    for col in ["CV_Accuracy", "Test_Accuracy", "Sensitivity", "Specificity"]:
        if col in df.columns:
            sort_columns.append(col)
            ascending_values.append(False)

    if "#Features" in df.columns:
        sort_columns.append("#Features")
        ascending_values.append(True)

    if len(sort_columns) == 0:
        return df

    ranked_df = df.sort_values(
        by=sort_columns,
        ascending=ascending_values,
        na_position="last",
    ).reset_index(drop=True)

    if "Rank" in ranked_df.columns:
        ranked_df = ranked_df.drop(columns=["Rank"])

    ranked_df.insert(0, "Rank", range(1, len(ranked_df) + 1))

    return ranked_df


def get_best_per_feature_count(ranked_df):
    """
    Select best LightGBM model for each number of features.
    """
    if ranked_df.empty or "#Features" not in ranked_df.columns:
        return pd.DataFrame()

    sort_columns = ["#Features"]
    ascending_values = [True]

    for col in ["CV_Accuracy", "Test_Accuracy", "Sensitivity", "Specificity"]:
        if col in ranked_df.columns:
            sort_columns.append(col)
            ascending_values.append(False)

    best_df = (
        ranked_df
        .sort_values(
            by=sort_columns,
            ascending=ascending_values,
            na_position="last",
        )
        .groupby("#Features", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    return best_df


def get_top3_overall(ranked_df):
    """
    Select Top 3 LightGBM models overall.
    """
    if ranked_df.empty:
        return pd.DataFrame()

    return ranked_df.head(3).copy()


def get_top3_by_feature_counts(ranked_df, top3_feature_counts):
    """
    Select Top 3 models for specific feature counts.
    Example:
        5 features, 6 features, 10 features
    """
    if ranked_df.empty or "#Features" not in ranked_df.columns:
        return pd.DataFrame()

    top_rows = []

    for feature_count in top3_feature_counts:
        subset = ranked_df[ranked_df["#Features"] == feature_count].copy()

        if subset.empty:
            continue

        subset = subset.head(3)
        subset.insert(0, "Target_Feature_Count", feature_count)

        top_rows.append(subset)

    if len(top_rows) == 0:
        return pd.DataFrame()

    return pd.concat(top_rows, ignore_index=True)


def save_scores_excel(
    ranked_results_per_sheet,
    best_per_feature_per_sheet,
    output_scores_file,
):
    """
    Save full scores file.
    """
    output_dir = os.path.dirname(output_scores_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    used_names = set()

    with pd.ExcelWriter(output_scores_file, engine="openpyxl") as writer:
        written = False

        for sheet_name, ranked_df in ranked_results_per_sheet.items():
            safe_name = make_safe_sheet_name(
                f"{sheet_name}_Ranked",
                used_names,
            )

            ranked_df.to_excel(
                writer,
                sheet_name=safe_name,
                index=False,
            )

            written = True

        for sheet_name, best_df in best_per_feature_per_sheet.items():
            safe_name = make_safe_sheet_name(
                f"{sheet_name}_BestByFeat",
                used_names,
            )

            best_df.to_excel(
                writer,
                sheet_name=safe_name,
                index=False,
            )

            written = True

        if not written:
            pd.DataFrame(
                {"Message": ["No LightGBM scores found."]}
            ).to_excel(
                writer,
                sheet_name="No_Results",
                index=False,
            )


def save_top3_excel(
    top3_overall_per_sheet,
    top3_by_feature_per_sheet,
    output_top3_file,
):
    """
    Save Top 3 LightGBM file.
    """
    output_dir = os.path.dirname(output_top3_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    used_names = set()

    with pd.ExcelWriter(output_top3_file, engine="openpyxl") as writer:
        written = False

        for sheet_name, top3_df in top3_overall_per_sheet.items():
            safe_name = make_safe_sheet_name(
                f"{sheet_name}_Top3Overall",
                used_names,
            )

            top3_df.to_excel(
                writer,
                sheet_name=safe_name,
                index=False,
            )

            written = True

        for sheet_name, top3_feature_df in top3_by_feature_per_sheet.items():
            safe_name = make_safe_sheet_name(
                f"{sheet_name}_Top3ByFeat",
                used_names,
            )

            top3_feature_df.to_excel(
                writer,
                sheet_name=safe_name,
                index=False,
            )

            written = True

        if not written:
            pd.DataFrame(
                {"Message": ["No LightGBM Top 3 models found."]}
            ).to_excel(
                writer,
                sheet_name="No_Results",
                index=False,
            )


def get_lightgbm_scores(
    combination_results_file,
    output_scores_file,
    output_top3_file,
    top3_feature_counts=None,
    logger=None,
):
    """
    Main function for Phase 4.3 LightGBM.

    Inputs:
        combination_results_file:
            Phase 4.2 LightGBM Step 2 results.

        output_scores_file:
            Full ranked LightGBM scores file.

        output_top3_file:
            Top 3 LightGBM scores file.

        top3_feature_counts:
            Feature counts for Top 3 by feature count.
            Example: [5, 6, 10]
    """

    if top3_feature_counts is None:
        top3_feature_counts = [5, 6, 10]

    if logger is not None:
        logger.info(f"Reading LightGBM Step 2 file: {combination_results_file}")

    if not os.path.exists(combination_results_file):
        raise FileNotFoundError(
            f"LightGBM Step 2 file not found: {combination_results_file}"
        )

    excel_file = pd.ExcelFile(combination_results_file)

    ranked_results_per_sheet = {}
    best_per_feature_per_sheet = {}
    top3_overall_per_sheet = {}
    top3_by_feature_per_sheet = {}

    for sheet_name in excel_file.sheet_names:
        if logger is not None:
            logger.info(f"Processing sheet: {sheet_name}")

        df = pd.read_excel(
            combination_results_file,
            sheet_name=sheet_name,
        )

        df = clean_lightgbm_scores_dataframe(df)

        if df.empty:
            if logger is not None:
                logger.warning(f"Sheet {sheet_name} is empty. Skipping.")
            continue

        ranked_df = rank_lightgbm_scores(df)

        if ranked_df.empty:
            if logger is not None:
                logger.warning(f"No ranked results for sheet {sheet_name}. Skipping.")
            continue

        best_per_feature_df = get_best_per_feature_count(ranked_df)
        top3_overall_df = get_top3_overall(ranked_df)
        top3_by_feature_df = get_top3_by_feature_counts(
            ranked_df=ranked_df,
            top3_feature_counts=top3_feature_counts,
        )

        ranked_results_per_sheet[sheet_name] = ranked_df
        best_per_feature_per_sheet[sheet_name] = best_per_feature_df
        top3_overall_per_sheet[sheet_name] = top3_overall_df
        top3_by_feature_per_sheet[sheet_name] = top3_by_feature_df

        if logger is not None:
            logger.info(
                f"Sheet={sheet_name} | "
                f"Total models={len(ranked_df)} | "
                f"Best CV_Accuracy={ranked_df['CV_Accuracy'].iloc[0]:.4f}"
            )

            logger.info("Top 3 LightGBM models:")

            for _, row in top3_overall_df.iterrows():
                test_acc = row["Test_Accuracy"] if "Test_Accuracy" in row else np.nan
                sensitivity = row["Sensitivity"] if "Sensitivity" in row else np.nan
                specificity = row["Specificity"] if "Specificity" in row else np.nan

                logger.info(
                    f"Rank {int(row['Rank'])} | "
                    f"#Features={int(row['#Features'])} | "
                    f"CV_Accuracy={row['CV_Accuracy']:.4f} | "
                    f"Test_Accuracy={test_acc} | "
                    f"Sensitivity={sensitivity} | "
                    f"Specificity={specificity}"
                )

    save_scores_excel(
        ranked_results_per_sheet=ranked_results_per_sheet,
        best_per_feature_per_sheet=best_per_feature_per_sheet,
        output_scores_file=output_scores_file,
    )

    save_top3_excel(
        top3_overall_per_sheet=top3_overall_per_sheet,
        top3_by_feature_per_sheet=top3_by_feature_per_sheet,
        output_top3_file=output_top3_file,
    )

    if logger is not None:
        logger.info(f"Saved LightGBM scores file to: {output_scores_file}")
        logger.info(f"Saved LightGBM Top 3 file to: {output_top3_file}")

    return {
        "ranked_results": ranked_results_per_sheet,
        "best_per_feature": best_per_feature_per_sheet,
        "top3_overall": top3_overall_per_sheet,
        "top3_by_feature": top3_by_feature_per_sheet,
    }

