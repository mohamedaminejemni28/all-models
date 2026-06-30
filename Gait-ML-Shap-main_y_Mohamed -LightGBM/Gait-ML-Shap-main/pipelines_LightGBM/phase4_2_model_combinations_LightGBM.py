import os
import sys

# Add NEW copied project src directory to Python path FIRST
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from gaitml import logger
from gaitml.utils.helpers import read_yaml

from gaitml.models.get_model_combinations_with_accuracy_LightGBM import (
    get_combination_list_with_accuracy_lightgbm,
)


def main():
    logger.info(
        ">>>>>>>>>> Starting Phase 4.2: LightGBM Model Combinations <<<<<<<<<<"
    )

    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    config = read_yaml(config_path)

    dataset_train_file = os.path.join(
        PROJECT_ROOT,
        config["SFS"]["dataset_train_file"],
    )

    dataset_test_file = os.path.join(
        PROJECT_ROOT,
        config["SFS"]["dataset_test_file"],
    )

    labels_column = config["SFS"]["labels_column"]

    sfs_results_excel_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["sfs_results_excel_file_lightgbm"],
    )

    output_excel_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["combination_results_file_lightgbm"],
    )

    lightgbm_config = config.get("LightGBM", {})

    participant_column = lightgbm_config.get("participant_column", "Participant")

    logger.info(f"Train file: {dataset_train_file}")
    logger.info(f"Test file: {dataset_test_file}")
    logger.info(f"SFS LightGBM file: {sfs_results_excel_file}")
    logger.info(f"Output file: {output_excel_file}")
    logger.info(f"Labels column: {labels_column}")
    logger.info(f"Using SRC_PATH = {SRC_PATH}")

    get_combination_list_with_accuracy_lightgbm(
        train_excel_file=dataset_train_file,
        test_excel_file=dataset_test_file,
        sfs_results_excel_file=sfs_results_excel_file,
        output_excel_file=output_excel_file,
        group_column=labels_column,
        participant_column=participant_column,

        feature_range=(5, 20),

        n_estimators_set=lightgbm_config.get("n_estimators", [50, 100, 150]),
        max_depth_set=lightgbm_config.get("max_depth", [-1, 2, 3, 4]),
        learning_rate_set=lightgbm_config.get("learning_rate", [0.01, 0.05, 0.1]),
        num_leaves_set=lightgbm_config.get("num_leaves", [7, 15, 31]),
        subsample_set=lightgbm_config.get("subsample", [0.8, 1.0]),
        colsample_bytree_set=lightgbm_config.get("colsample_bytree", [0.8, 1.0]),
        min_child_samples_set=lightgbm_config.get("min_child_samples", [2, 5, 10]),
        reg_alpha_set=lightgbm_config.get("reg_alpha", [0.0]),
        reg_lambda_set=lightgbm_config.get("reg_lambda", [0.0]),

        n_splits=lightgbm_config.get("n_splits", 10),
        random_state=lightgbm_config.get("random_state", 0),
        max_candidates=lightgbm_config.get("max_candidates", 1000),

        logger=logger,
    )

    logger.info(
        ">>>>>>>>>> Completed Phase 4.2: LightGBM Model Combinations <<<<<<<<<<"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in Phase 4.2 LightGBM: {e}")
        raise e
