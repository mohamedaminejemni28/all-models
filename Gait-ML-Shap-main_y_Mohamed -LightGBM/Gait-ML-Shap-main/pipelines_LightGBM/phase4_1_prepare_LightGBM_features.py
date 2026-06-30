import os
import sys

# Add project src directory to Python path first
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from gaitml import logger
from gaitml.utils.helpers import read_yaml
from gaitml.features.sequential_feature_selection_lightgbm import (
    sequential_feature_selection_lightgbm,
)


def main():
    logger.info(">>>>>>>>>> Starting Phase 4.1: LightGBM Sequential Feature Selection <<<<<<<<<<")

    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    config = read_yaml(config_path)

    dataset_train_file = os.path.join(
        PROJECT_ROOT,
        config["SFS"]["dataset_train_file"],
    )

    labels_column = config["SFS"]["labels_column"]
    classes_dict = config["SFS"]["classes"]

    output_SFS_scores_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["sfs_results_excel_file_lightgbm"],
    )

    lightgbm_config = config.get("LightGBM", {})

    participant_column = lightgbm_config.get("participant_column", "Participant")

    feature_counts = lightgbm_config.get(
        "feature_counts",
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    )

    logger.info(f"Train file: {dataset_train_file}")
    logger.info(f"Output LightGBM SFS file: {output_SFS_scores_file}")
    logger.info(f"Working with classes_dict = {classes_dict}")
    logger.info(f"Using SRC_PATH = {SRC_PATH}")
    logger.info(f"Feature counts: {feature_counts}")

    sequential_feature_selection_lightgbm(
        input_excel_file=dataset_train_file,
        output_excel_file=output_SFS_scores_file,
        group_column=labels_column,
        classes_dict=classes_dict,
        participant_column=participant_column,
        feature_counts=feature_counts,

        n_estimators_set=lightgbm_config.get("n_estimators", [50, 100, 150]),
        max_depth_set=lightgbm_config.get("max_depth", [-1, 2, 3, 4]),
        learning_rate_set=lightgbm_config.get("learning_rate", [0.01, 0.05, 0.1]),
        num_leaves_set=lightgbm_config.get("num_leaves", [7, 15, 31]),
        subsample_set=lightgbm_config.get("subsample", [0.8, 1.0]),
        colsample_bytree_set=lightgbm_config.get("colsample_bytree", [0.8, 1.0]),
        min_child_samples_set=lightgbm_config.get("min_child_samples", [2, 5, 10]),
        reg_alpha_set=lightgbm_config.get("reg_alpha", [0.0]),
        reg_lambda_set=lightgbm_config.get("reg_lambda", [0.0]),

        kfold_n_splits=lightgbm_config.get("n_splits", 5),
        random_state=lightgbm_config.get("random_state", 42),
        max_candidates=lightgbm_config.get("max_candidates", 50),
        save_partial=True,
    )

    logger.info(">>>>>>>>>> Completed Phase 4.1: LightGBM Sequential Feature Selection <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in Phase 4.1 LightGBM: {e}")
        raise e
