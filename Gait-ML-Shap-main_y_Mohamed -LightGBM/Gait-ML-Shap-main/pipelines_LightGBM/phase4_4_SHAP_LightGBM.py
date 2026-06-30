import os
import sys

# Add project src directory to Python path first
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from gaitml import logger
from gaitml.utils.helpers import read_yaml

from gaitml.models.get_SHAP_data_LightGBM import (
    get_SHAP_data_LightGBM,
)


def main():
    logger.info(">>>>>>>>>> Starting Phase 4.4: LightGBM SHAP <<<<<<<<<<")

    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    config = read_yaml(config_path)

    scores_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["scores_file_top3_lightgbm"],
    )

    train_file = os.path.join(
        PROJECT_ROOT,
        config["SFS"]["dataset_train_file"],
    )

    test_file = os.path.join(
        PROJECT_ROOT,
        config["SFS"]["dataset_test_file"],
    )

    output_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["output_file_lightgbm"],
    )

    output_dir = os.path.join(
        PROJECT_ROOT,
        config["data"]["output_dir_lightgbm"],
    )

    labels_column = config["SFS"]["labels_column"]

    lightgbm_config = config.get("LightGBM", {})
    participant_column = lightgbm_config.get("participant_column", "Participant")

    logger.info(f"Scores file: {scores_file}")
    logger.info(f"Train file: {train_file}")
    logger.info(f"Test file: {test_file}")
    logger.info(f"Output SHAP Excel file: {output_file}")
    logger.info(f"Output SHAP images directory: {output_dir}")
    logger.info(f"Labels column: {labels_column}")
    logger.info(f"Participant column: {participant_column}")

    get_SHAP_data_LightGBM(
        scores_file=scores_file,
        train_file=train_file,
        test_file=test_file,
        output_file=output_file,
        output_dir=output_dir,
        labels_column=labels_column,
        participant_column=participant_column,
    )

    logger.info(">>>>>>>>>> Completed Phase 4.4: LightGBM SHAP <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in Phase 4.4 LightGBM: {e}")
        raise e
