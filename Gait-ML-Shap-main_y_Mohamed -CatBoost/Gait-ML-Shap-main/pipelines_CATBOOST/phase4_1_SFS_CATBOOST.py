from gaitml import logger
from gaitml.features.sequential_feature_selection_catboost import (
    sequential_feature_selection_catboost,
)
from pipeline_common import SRC_PATH, load_config, prepare_catboost_config, project_path


def main():
    logger.info(">>>>>>>>>> Starting Phase 4.1: CatBoost Sequential Feature Selection <<<<<<<<<<")

    config, config_path = load_config()
    logger.info(f"Using config: {config_path}")

    catboost_config = prepare_catboost_config(config, logger=logger)
    feature_counts = catboost_config.get("feature_counts", [5, 6, 10])

    dataset_train_file = project_path(config["SFS"]["dataset_train_file"])
    output_sfs_scores_file = project_path(config["data"]["sfs_results_excel_file_catboost"])

    logger.info(f"Train file: {dataset_train_file}")
    logger.info(f"Output CatBoost SFS file: {output_sfs_scores_file}")
    logger.info(f"Working with classes_dict = {config['SFS']['classes']}")
    logger.info(f"Using SRC_PATH = {SRC_PATH}")
    logger.info(f"Feature counts: {feature_counts}")

    sequential_feature_selection_catboost(
        input_excel_file=dataset_train_file,
        output_excel_file=output_sfs_scores_file,
        group_column=config["SFS"].get("labels_column", "Group"),
        classes_dict=config["SFS"].get("classes"),
        participant_column=catboost_config.get("participant_column", "Participant"),
        feature_counts=feature_counts,
        iterations_set=catboost_config.get("iterations", [50, 100]),
        depth_set=catboost_config.get("depth", [2, 3]),
        learning_rate_set=catboost_config.get("learning_rate", [0.03, 0.1]),
        l2_leaf_reg_set=catboost_config.get("l2_leaf_reg", [1, 3]),
        bagging_temperature_set=catboost_config.get("bagging_temperature", [0.0]),
        random_strength_set=catboost_config.get("random_strength", [0.0]),
        kfold_n_splits=catboost_config.get("n_splits", 3),
        random_state=catboost_config.get("random_state", 42),
        max_candidates=catboost_config.get("max_candidates", 20),
        save_partial=True,
    )

    logger.info(">>>>>>>>>> Completed Phase 4.1: CatBoost Sequential Feature Selection <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in Phase 4.1 CatBoost: {e}")
        raise
