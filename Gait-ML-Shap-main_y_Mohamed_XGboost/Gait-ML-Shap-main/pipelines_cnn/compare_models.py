"""Create comparison Excel tables for SVM, XGBoost, Random Forest, and CNN."""

import argparse
from pathlib import Path

import pandas as pd


MODEL_PATTERNS = {
    "SVM": ["{prefix}_Scores_RBF_Top3.xlsx", "{prefix}_Scores_RBF.xlsx"],
    "XGBoost": ["{prefix}_Scores_XGBoost_Top3.xlsx", "{prefix}_Scores_XGBoost.xlsx"],
    "RandomForest": ["{prefix}_Scores_RandomForest_Top3.xlsx", "{prefix}_Scores_RandomForest.xlsx"],
    "CNN": ["{prefix}_Scores_CNN_Top3.xlsx", "{prefix}_Scores_CNN.xlsx"],
}


def first_existing(results_dirs, prefix, patterns):
    for results_dir in results_dirs:
        for pattern in patterns:
            path = results_dir / pattern.format(prefix=prefix)
            if path.exists():
                return path
    return None


def read_best_rows(path, model_name):
    rows = []
    excel = pd.ExcelFile(path)
    for sheet_name in excel.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet_name)
        if df.empty:
            continue
        sort_cols = [col for col in ["Test_Accuracy", "CV_accuracy", "CV_Accuracy", "MCC", "F1"] if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=False)
        row = df.iloc[0].to_dict()
        row["Sheet"] = sheet_name
        row["Model_Family"] = model_name
        row["Source_File"] = str(path)
        rows.append(row)
    return rows


def build_comparison(results_dirs, prefixes, output_file):
    results_dirs = [Path(path) for path in results_dirs]
    all_rows = []
    missing = []
    for prefix in prefixes:
        for model_name, patterns in MODEL_PATTERNS.items():
            path = first_existing(results_dirs, prefix, patterns)
            if path is None:
                missing.append({"Dataset": prefix, "Model_Family": model_name, "Missing": ", ".join(patterns)})
                continue
            for row in read_best_rows(path, model_name):
                row["Dataset"] = prefix
                all_rows.append(row)

    comparison = pd.DataFrame(all_rows)
    metric_cols = [
        "Dataset", "Sheet", "Model_Family", "Name_Model", "Model_Type", "#Features",
        "CV_accuracy", "CV_Accuracy", "Test_Accuracy", "Sensitivity", "Specificity",
        "F1", "MCC", "PPV", "NPV", "Likelihood_Ratio", "Source_File",
    ]
    ordered = [col for col in metric_cols if col in comparison.columns]
    comparison = comparison[ordered + [col for col in comparison.columns if col not in ordered]]

    with pd.ExcelWriter(output_file) as writer:
        comparison.to_excel(writer, sheet_name="Best_Model_Comparison", index=False)
        pd.DataFrame(missing).to_excel(writer, sheet_name="Missing_Files", index=False)
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Compare classical ML and CNN score files.")
    parser.add_argument("--results-dir", default=None, help="Backward-compatible single results directory.")
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=None,
        help="One or more directories containing score Excel files. Earlier directories take priority.",
    )
    parser.add_argument(
        "--prefixes",
        nargs="+",
        default=["autism_2024", "young_old_2024", "flatfoot_control_older_2024"],
    )
    parser.add_argument("--output", default="results/model_family_comparison.xlsx")
    args = parser.parse_args()
    results_dirs = args.results_dirs or [args.results_dir or "results"]
    output = build_comparison(results_dirs, args.prefixes, args.output)
    print(f"Saved comparison table to {output}")


if __name__ == "__main__":
    main()
