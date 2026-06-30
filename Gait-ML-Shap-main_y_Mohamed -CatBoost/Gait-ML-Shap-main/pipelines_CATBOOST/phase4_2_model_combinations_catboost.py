from gaitml import logger
from gaitml.models.get_model_combinations_with_accuracy_catboost import (
    get_combination_list_with_accuracy_catboost,
)
from pipeline_common import SRC_PATH, load_config, prepare_catboost_config, project_path


def main():
    logger.info(">>>>>>>>>> Starting Phase 4.2: CatBoost Model Combinations <<<<<<<<<<")

    config, config_path = load_config()
    logger.info(f"Using config: {config_path}")

    catboost_config = prepare_catboost_config(config, logger=logger)
    feature_counts = catboost_config.get("feature_counts", [5, 6, 10])

    dataset_train_file = project_path(config["SFS"]["dataset_train_file"])
    dataset_test_file = project_path(config["SFS"]["dataset_test_file"])
    sfs_results_excel_file = project_path(config["data"]["sfs_results_excel_file_catboost"])
    output_excel_file = project_path(config["data"]["combination_results_file_catboost"])

    logger.info(f"Train file: {dataset_train_file}")
    logger.info(f"Test file: {dataset_test_file}")
    logger.info(f"SFS CatBoost file: {sfs_results_excel_file}")
    logger.info(f"Output file: {output_excel_file}")
    logger.info(f"Labels column: {config['SFS'].get('labels_column', 'Group')}")
    logger.info(f"Participant column: {catboost_config.get('participant_column', 'Participant')}")
    logger.info(f"Using SRC_PATH = {SRC_PATH}")

    get_combination_list_with_accuracy_catboost(
        train_excel_file=dataset_train_file,
        test_excel_file=dataset_test_file,
        sfs_results_excel_file=sfs_results_excel_file,
        output_excel_file=output_excel_file,
        group_column=config["SFS"].get("labels_column", "Group"),
        participant_column=catboost_config.get("participant_column", "Participant"),
        feature_range=(min(feature_counts), max(feature_counts) + 1),
        iterations_set=catboost_config.get("iterations", [50, 100]),
        depth_set=catboost_config.get("depth", [2, 3]),
        learning_rate_set=catboost_config.get("learning_rate", [0.03, 0.1]),
        l2_leaf_reg_set=catboost_config.get("l2_leaf_reg", [1, 3]),
        bagging_temperature_set=catboost_config.get("bagging_temperature", [0.0]),
        random_strength_set=catboost_config.get("random_strength", [0.0]),
        n_splits=catboost_config.get("n_splits", 3),
        random_state=catboost_config.get("random_state", 42),
        max_candidates=catboost_config.get("max_candidates", 8),
        logger=logger,
    )

    logger.info(">>>>>>>>>> Completed Phase 4.2: CatBoost Model Combinations <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in Phase 4.2 CatBoost: {e}")
        raise
