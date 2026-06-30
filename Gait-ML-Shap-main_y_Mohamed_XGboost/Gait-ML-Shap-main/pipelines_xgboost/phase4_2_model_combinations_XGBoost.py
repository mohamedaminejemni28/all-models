import os
import sys

# Add NEW copied project src directory to Python path FIRST
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from gaitml import logger
from gaitml.utils.helpers import read_yaml

from gaitml.models.get_model_combinations_with_accuracy_xgboost import (
    get_combination_list_with_accuracy_xgboost
)


def main():
    logger.info(
        ">>>>>>>>>> Starting Phase 4.2: XGBoost Model Combinations <<<<<<<<<<"
    )

    config_path = "configs/config.yaml"
    config = read_yaml(config_path)

    dataset_train_file = config["SFS"]["dataset_train_file"]
    dataset_test_file = config["SFS"]["dataset_test_file"]
    labels_column = config["SFS"]["labels_column"]

    sfs_results_excel_file = config["data"]["sfs_results_excel_file_xgboost"]
    output_excel_file = config["data"]["combination_results_file_xgboost"]

    logger.info(f"Train file: {dataset_train_file}")
    logger.info(f"Test file: {dataset_test_file}")
    logger.info(f"SFS XGBoost file: {sfs_results_excel_file}")
    logger.info(f"Output file: {output_excel_file}")
    logger.info(f"Labels column: {labels_column}")
    logger.info(f"Using SRC_PATH = {SRC_PATH}")

    get_combination_list_with_accuracy_xgboost(
        train_excel_file=dataset_train_file,
        test_excel_file=dataset_test_file,
        sfs_results_excel_file=sfs_results_excel_file,
        output_excel_file=output_excel_file,
        group_column=labels_column,

        feature_range=(5, 20),
        n_estimators_set=[50, 100, 150],
        max_depth_set=[2, 3, 4],
        learning_rate_set=[0.01, 0.05, 0.1],
        subsample_set=[0.8, 1.0],
        colsample_bytree_set=[0.8, 1.0],

        n_splits=10,
        random_state=0,
        logger=logger
    )

    logger.info(
        ">>>>>>>>>> Completed Phase 4.2: XGBoost Model Combinations <<<<<<<<<<"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in Phase 4.2 XGBoost: {e}")
        raise e