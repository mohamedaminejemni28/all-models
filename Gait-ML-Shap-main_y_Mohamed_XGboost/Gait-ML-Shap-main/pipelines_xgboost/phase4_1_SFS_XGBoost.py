import os
import sys

# Add NEW copied project src directory to Python path FIRST
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from gaitml import logger
from gaitml.utils.helpers import read_yaml
from gaitml.features.sequential_feature_selection_xgboost import (
    sequential_feature_selection_xgboost
)


def main():
    logger.info(">>>>>>>>>> Starting Phase 4.1: XGBoost Sequential Feature Selection <<<<<<<<<<")

    config_path = "configs/config.yaml"
    config = read_yaml(config_path)

    dataset_train_file = config["SFS"]["dataset_train_file"]
    labels_column = config["SFS"]["labels_column"]
    classes_dict = config["SFS"]["classes"]

    output_SFS_scores_file = config["data"]["sfs_results_excel_file_xgboost"]

    logger.info(f"Train file: {dataset_train_file}")
    logger.info(f"Output XGBoost SFS file: {output_SFS_scores_file}")
    logger.info(f"Working with classes_dict = {classes_dict}")
    logger.info(f"Using SRC_PATH = {SRC_PATH}")

    sequential_feature_selection_xgboost(
        input_excel_file=dataset_train_file,
        output_excel_file=output_SFS_scores_file,
        group_column=labels_column,
        classes_dict=classes_dict,
        n_features=50,
        n_estimators_set=[50, 100],
        max_depth_set=[2, 3],
        learning_rate_set=[0.05, 0.1],
        kfold_n_splits=10,
        random_state=0
    )

    logger.info(">>>>>>>>>> Completed Phase 4.1: XGBoost Sequential Feature Selection <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in Phase 4.1 XGBoost: {e}")
        raise e