import pandas as pd
from pathlib import Path

# ============================================================
# USER INPUTS
# ============================================================

input_file = "young_old_2024_Scores_RBF.xlsx"
output_file = "young_old_2024_Scores_RBF_Top3.xlsx"

top_n = 3

# ============================================================
# READ EXCEL
# ============================================================

input_path = Path(input_file)

if not input_path.exists():
    raise FileNotFoundError(f"File not found: {input_file}")

excel = pd.ExcelFile(input_file)
output_sheets = {}

# ============================================================
# PROCESS EACH SHEET
# ============================================================

for sheet_name in excel.sheet_names:
    print("=" * 80)
    print(f"Processing sheet: {sheet_name}")

    df = pd.read_excel(input_file, sheet_name=sheet_name)

    print("Columns found:")
    print(df.columns.tolist())

    # Correct column names from your file
    required_columns = ["CV_accuracy", "Test_Accuracy", "Sensitivity"]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing columns in sheet '{sheet_name}': {missing_columns}\n"
            f"Available columns are: {df.columns.tolist()}"
        )

    # Sort by best performance
    df_sorted = df.sort_values(
        by=["CV_accuracy", "Test_Accuracy", "Sensitivity"],
        ascending=[False, False, False]
    )

    # Keep top X models
    df_top = df_sorted.head(top_n)

    output_sheets[sheet_name] = df_top

    print(f"Top {top_n} rows selected from sheet: {sheet_name}")

# ============================================================
# SAVE OUTPUT
# ============================================================

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    for sheet_name, df_top in output_sheets.items():
        df_top.to_excel(writer, sheet_name=sheet_name, index=False)

print("=" * 80)
print(f"Done. Created file: {output_file}")
print("=" * 80)