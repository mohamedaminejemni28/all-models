from gaitml import logger
from gaitml.data.create_train_val_test_split import create_train_test_split
from pipeline_common import load_config, project_path


def main():
    logger.info(">>>>>>>>>> Starting Phase 3: Splitting Datasets <<<<<<<<<<")

    config, config_path = load_config()
    logger.info(f"Using config: {config_path}")

    create_train_test_split(
        input_excel_file=project_path(config["split_datasets"]["input_file"]),
        output_train_file=project_path(config["split_datasets"]["output_train_file"]),
        output_test_file=project_path(config["split_datasets"]["output_test_file"]),
        group_column=config["SFS"].get("labels_column", "Group"),
        test_size=0.2,
        stratify=True,
        random_state=0,
    )

    logger.info(">>>>>>>>>> Completed Phase 3: Splitting Datasets <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
