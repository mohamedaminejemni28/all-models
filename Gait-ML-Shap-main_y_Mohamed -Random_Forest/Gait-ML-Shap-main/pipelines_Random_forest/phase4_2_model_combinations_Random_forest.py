import os
import sys

# Add project src directory to Python path first
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from gaitml import logger
from gaitml.utils.helpers import read_yaml

from gaitml.models.get_model_combinations_with_accuracy_random_forest import (
    get_combination_list_with_accuracy_random_forest
)


def main():
    logger.info(
        ">>>>>>>>>> Starting Phase 4.2: Random Forest Model Combinations <<<<<<<<<<"
    )

    config_path = "configs/config.yaml"
    config = read_yaml(config_path)

    dataset_train_file = config["SFS"]["dataset_train_file"]
    dataset_test_file = config["SFS"]["dataset_test_file"]
    labels_column = config["SFS"]["labels_column"]

    sfs_results_excel_file = (
        config["data"].get("sfs_results_excel_file_random_forest")
        or config["data"].get("sfs_results_excel_file_Random_forest")
        or config["SFS"].get("Phase4_1_SFS_scores_file")
    )

    output_excel_file = config["data"]["combination_results_file_random_forest"]

    logger.info(f"Train file: {dataset_train_file}")
    logger.info(f"Test file: {dataset_test_file}")
    logger.info(f"SFS Random Forest file: {sfs_results_excel_file}")
    logger.info(f"Output file: {output_excel_file}")
    logger.info(f"Labels column: {labels_column}")
    logger.info(f"Using SRC_PATH = {SRC_PATH}")

    get_combination_list_with_accuracy_random_forest(
        train_excel_file=dataset_train_file,
        test_excel_file=dataset_test_file,
        sfs_results_excel_file=sfs_results_excel_file,
        output_excel_file=output_excel_file,
        group_column=labels_column,

        # Use all SFS rows from 1 feature to 19 features
        feature_range=(1, 19),

        # Random Forest parameter cases
        n_estimators_set=[50, 100, 150],
        max_depth_set=[2, 3, 4],
        min_samples_split_set=[2, 5],
        min_samples_leaf_set=[1, 2],
        max_features_set=["sqrt", "log2"],

        # 10-fold cross-validation
        n_splits=10,
        random_state=0,
        logger=logger
    )

    logger.info(
        ">>>>>>>>>> Completed Phase 4.2: Random Forest Model Combinations <<<<<<<<<<"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in Phase 4.2 Random Forest: {e}")
        raise e