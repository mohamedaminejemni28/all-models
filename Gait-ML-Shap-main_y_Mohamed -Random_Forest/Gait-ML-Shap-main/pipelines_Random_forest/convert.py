import os
import sys
import pandas as pd

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from gaitml import logger
from gaitml.utils.helpers import read_yaml


# ============================================================
# LABEL MAP
# ============================================================

label_map = {
    "Young": 0,
    "Older": 1,
    "young": 0,
    "older": 1,
    "Young Adults": 0,
    "Older Adults": 1,
    "Control": 0,
    "Autism": 1,
    "control": 0,
    "autism": 1,
    "0": 0,
    "1": 1,
    0: 0,
    1: 1
}


# ============================================================
# HELPER FUNCTION
# ============================================================

def convert_group_column_to_numeric(file_path):
    """
    Convert the Group column in all sheets of an Excel file to numeric 0/1.
    The file is overwritten after conversion.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Processing file: {file_path}")

    excel = pd.ExcelFile(file_path)
    output_sheets = {}

    for sheet_name in excel.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")

        df = pd.read_excel(file_path, sheet_name=sheet_name)

        if "Group" not in df.columns:
            raise ValueError(
                f"Group column not found in file '{file_path}', sheet '{sheet_name}'"
            )

        # Clean labels
        df["Group"] = df["Group"].astype(str).str.strip()

        # Map labels to 0/1
        df["Group"] = df["Group"].replace(label_map)

        # Force numeric conversion
        df["Group"] = pd.to_numeric(df["Group"], errors="coerce")

        # Check failed values
        if df["Group"].isna().any():
            logger.error("Problematic Group values:")
            logger.error(df["Group"].value_counts(dropna=False))
            raise ValueError(
                f"Some Group labels could not be converted to 0/1 in sheet '{sheet_name}'."
            )

        # Convert to integer
        df["Group"] = df["Group"].astype(int)

        logger.info(f"Sheet: {sheet_name}")
        logger.info(f"Group counts:\n{df['Group'].value_counts(dropna=False)}")
        logger.info(f"Unique values: {df['Group'].unique()}")
        logger.info(f"Dtype: {df['Group'].dtype}")

        output_sheets[sheet_name] = df

    # Save back to same Excel file
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        for sheet_name, df in output_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Saved converted file: {file_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info(">>>>>>>>>> Starting Group Column Conversion to 0/1 <<<<<<<<<<")

    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    config = read_yaml(config_path)

    train_file = config["split_datasets"]["output_train_file"]
    test_file = config["split_datasets"]["output_test_file"]

    # Convert relative paths to absolute paths
    if not os.path.isabs(train_file):
        train_file = os.path.join(PROJECT_ROOT, train_file)

    if not os.path.isabs(test_file):
        test_file = os.path.join(PROJECT_ROOT, test_file)

    files = [train_file, test_file]

    logger.info(f"Train file: {train_file}")
    logger.info(f"Test file: {test_file}")

    for file_path in files:
        convert_group_column_to_numeric(file_path)

    logger.info("Done. Group column converted to numeric 0/1.")
    logger.info(">>>>>>>>>> Completed Group Column Conversion <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred while converting Group column: {e}")
        raise e