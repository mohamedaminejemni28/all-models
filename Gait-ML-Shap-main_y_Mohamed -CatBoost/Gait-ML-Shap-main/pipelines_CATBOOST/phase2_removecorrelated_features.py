from gaitml import logger
from gaitml.data.remove_correlated_features import remove_correlated_features
from pipeline_common import load_config, project_path


def main():
    logger.info(">>>>>>>>>> Starting Phase 2: Removing Correlated Features <<<<<<<<<<")

    config, config_path = load_config()
    logger.info(f"Using config: {config_path}")

    remove_correlated_features(
        project_path(config["remove_correlated_features"]["highly_correlated_features_file"]),
        project_path(config["remove_correlated_features"]["input_excel_file"]),
        project_path(config["remove_correlated_features"]["output_excel_file"]),
    )

    logger.info(">>>>>>>>>> Completed Phase 2: Removing Correlated Features <<<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
