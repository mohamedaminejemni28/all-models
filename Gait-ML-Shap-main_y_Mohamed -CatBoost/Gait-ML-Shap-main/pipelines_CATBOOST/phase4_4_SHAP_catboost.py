from gaitml import logger
from gaitml.models.get_SHAP_data_CatBoost import get_SHAP_data_CatBoost
from pipeline_common import SRC_PATH, load_config, project_path


def main():
    logger.info(">>>>>>>>>> Starting Phase 4.4: CatBoost SHAP <<<<<<<<<<")

    config, config_path = load_config()
    logger.info(f"Using config: {config_path}")

    catboost_config = config.get("CatBoost", {})
    scores_file = project_path(config["data"]["scores_file_top3_catboost"])
    train_file = project_path(config["SFS"]["dataset_train_file"])
    test_file = project_path(config["SFS"]["dataset_test_file"])
    output_file = project_path(config["data"]["output_file_catboost"])
    output_dir = project_path(config["data"]["output_dir_catboost"])

    logger.info(f"Scores Top3 CatBoost file: {scores_file}")
    logger.info(f"Train file: {train_file}")
    logger.info(f"Test file: {test_file}")
    logger.info(f"Output SHAP Excel file: {output_file}")
    logger.info(f"Output SHAP image directory: {output_dir}")
    logger.info(f"Labels column: {config['SFS'].get('labels_column', 'Group')}")
    logger.info(f"Participant column: {catboost_config.get('participant_column', 'Participant')}")
    logger.info(f"Using SRC_PATH = {SRC_PATH}")

    get_SHAP_data_CatBoost(
        scores_file=scores_file,
        train_file=train_file,
        test_file=test_file,
        output_file=output_file,
        output_dir=output_dir,
        labels_column=config["SFS"].get("labels_column", "Group"),
        participant_column=catboost_config.get("participant_column", "Participant"),
    )

    logger.info(">>>>>>>>>> Completed Phase 4.4: CatBoost SHAP <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in Phase 4.4 CatBoost SHAP: {e}")
        raise
