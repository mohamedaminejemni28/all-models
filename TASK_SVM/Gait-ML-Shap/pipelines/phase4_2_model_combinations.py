import os
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gaitml import logger
from gaitml.utils.helpers import read_yaml
from gaitml.models.get_model_combinations_with_accuracy import get_combination_list_with_accuracy

def main():
    logger.info("Starting Phase 4.2: Model Combinations")
    config_path = "configs/config.yaml"
    config = read_yaml(config_path)
    variant_output = config["data"]["variant_output"]
    invariant_output = config["data"]["invariant_output"]
    sfs_results_excel_file = config["data"]["sfs_results_excel_file"]

    get_combination_list_with_accuracy(variant_output, invariant_output, sfs_results_excel_file)

    logger.info("Phase 4.2: Model Combinations completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e
