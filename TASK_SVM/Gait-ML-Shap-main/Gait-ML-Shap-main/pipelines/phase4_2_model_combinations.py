import os
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from gaitml import logger
from gaitml.utils.helpers import read_yaml
from gaitml.models.get_model_combinations_with_accuracy import (
    get_combination_list_with_accuracy,
)


def main():
    logger.info("Starting Phase 4.2: Model Combinations")

    config_path = "configs/config.yaml"
    config = read_yaml(config_path)

    dataset_train_file = config["SFS"]["dataset_train_file"]
    dataset_test_file = config["SFS"]["dataset_test_file"]
    sfs_results_excel_file = config["data"]["sfs_results_excel_file"]
    combination_results_file = config["data"]["combination_results_file"]

    labels_column = config["SFS"].get("labels_column", "Group")

    get_combination_list_with_accuracy(
        train_excel_file=dataset_train_file,
        test_excel_file=dataset_test_file,
        sfs_results_excel_file=sfs_results_excel_file,
        output_excel_file=combination_results_file,
        group_column=labels_column,
    )

    logger.info("Phase 4.2: Model Combinations completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e