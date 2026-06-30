import os
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gaitml import logger
from gaitml.utils.helpers import read_yaml
from gaitml.models.get_RBF_Scores import get_RBF_scores

def main():
    logger.info("Starting Phase 4.3: Scores RBF")
    config_path = "configs/config.yaml"
    config = read_yaml(config_path)
    sfs_results_file = config["data"]["sfs_results_file"]
    combination_results_file = config["data"]["combination_results_file"]
    output_file = config["data"]["output_file"]

    get_RBF_scores(sfs_results_file, combination_results_file, output_file)

    logger.info("Phase 4.3: Scores RBF completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e