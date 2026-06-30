import os
import sys

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from gaitml import logger
from gaitml.utils.helpers import read_yaml
from gaitml.models.get_SHAP_data_Random_forest import get_SHAP_data_Random_forest


def main():
    logger.info(">>>>>>>>>> Starting Phase 4.4: Random Forest SHAP <<<<<<<<<<")

    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    config = read_yaml(config_path)

    scores_file = config["data"]["scores_file_top3_random_forest"]
    train_file = config["SFS"]["dataset_train_file"]
    test_file = config["SFS"]["dataset_test_file"]
    output_file = config["data"]["output_file_random_forest"]
    output_dir = config["data"]["output_dir_random_forest"]
    labels_column = config["SFS"]["labels_column"]

    # Convert relative paths to absolute paths
    if not os.path.isabs(scores_file):
        scores_file = os.path.join(PROJECT_ROOT, scores_file)

    if not os.path.isabs(train_file):
        train_file = os.path.join(PROJECT_ROOT, train_file)

    if not os.path.isabs(test_file):
        test_file = os.path.join(PROJECT_ROOT, test_file)

    if not os.path.isabs(output_file):
        output_file = os.path.join(PROJECT_ROOT, output_file)

    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)

    logger.info(f"Scores file: {scores_file}")
    logger.info(f"Train file: {train_file}")
    logger.info(f"Test file: {test_file}")
    logger.info(f"Output SHAP Excel: {output_file}")
    logger.info(f"Output SHAP directory: {output_dir}")
    logger.info(f"Labels column: {labels_column}")
    logger.info(f"Using SRC_PATH = {SRC_PATH}")

    get_SHAP_data_Random_forest(
        scores_file=scores_file,
        train_file=train_file,
        test_file=test_file,
        output_file=output_file,
        output_dir=output_dir,
        labels_column=labels_column
    )

    logger.info(">>>>>>>>>> Completed Phase 4.4: Random Forest SHAP <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in Phase 4.4 Random Forest SHAP: {e}")
        raise e