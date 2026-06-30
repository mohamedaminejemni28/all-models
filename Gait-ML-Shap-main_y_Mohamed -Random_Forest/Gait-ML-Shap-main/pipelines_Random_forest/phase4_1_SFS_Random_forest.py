import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from gaitml import logger
from gaitml.utils.helpers import read_yaml
from gaitml.features.sequential_feature_selection_random_forest import (
    sequential_feature_selection_random_forest
)


def main():
    logger.info(">>>>>>>>>> Starting Phase 4.1: Random Forest Sequential Feature Selection <<<<<<<<<<")

    config_path = "configs/config.yaml"
    config = read_yaml(config_path)

    dataset_train_file = config["SFS"]["dataset_train_file"]
    labels_column = config["SFS"]["labels_column"]
    classes_dict = config["SFS"]["classes"]

    output_SFS_scores_file = (
        config["data"].get("sfs_results_excel_file_random_forest")
        or config["data"].get("sfs_results_excel_file_Random_forest")
        or config["SFS"].get("Phase4_1_SFS_scores_file")
    )

    logger.info(f"Train file: {dataset_train_file}")
    logger.info(f"Output Random Forest SFS file: {output_SFS_scores_file}")
    logger.info(f"Working with classes_dict = {classes_dict}")
    logger.info(f"Using SRC_PATH = {SRC_PATH}")

    sequential_feature_selection_random_forest(
        input_excel_file=dataset_train_file,
        output_excel_file=output_SFS_scores_file,
        group_column=labels_column,
        classes_dict=classes_dict,

        # This will create rows for 5, 6, 7, 8, 9, 10 features
        n_features=19,

        # This gives several rows but still acceptable
        n_estimators_set=[50, 100],
        max_depth_set=[2, 3],

        learning_rate_set=None,
        kfold_n_splits=10,
        random_state=0
    )

    logger.info(">>>>>>>>>> Completed Phase 4.1: Random Forest Sequential Feature Selection <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in Phase 4.1 Random Forest: {e}")
        raise e