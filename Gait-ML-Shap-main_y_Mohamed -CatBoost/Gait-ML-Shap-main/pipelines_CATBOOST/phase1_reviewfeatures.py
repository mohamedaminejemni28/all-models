from gaitml import logger
from gaitml.data.get_filtered_dataset import get_filtered_datasets
from pipeline_common import load_config, project_path


def main():
    logger.info(">>>>>>>>>> Starting Phase 1: Reviewing Features <<<<<<<<<<")

    config, config_path = load_config()
    logger.info(f"Using config: {config_path}")

    feature_names_file = project_path(config["data"]["feature_names_file"])
    variant_file = project_path(config["data"]["variant_file"])
    invariant_file = project_path(config["data"]["invariant_file"])
    variant_output = project_path(config["data"]["variant_output"])
    invariant_output = project_path(config["data"]["invariant_output"])

    get_filtered_datasets(
        feature_names_file,
        variant_file,
        invariant_file,
        variant_output,
        invariant_output,
    )

    logger.info(">>>>>>>>>> Completed Phase 1: Reviewing Features <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
