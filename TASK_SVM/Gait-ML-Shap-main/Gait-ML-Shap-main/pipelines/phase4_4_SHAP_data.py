import os
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gaitml import logger
from gaitml.utils.helpers import read_yaml
from gaitml.models.get_SHAP_data import get_SHAP_data

def main():
    logger.info("Starting Phase 4.4: SHAP Data")
    config_path = "configs/config.yaml"
    config = read_yaml(config_path)
    scores_file = config["data"]["scores_file_top3"]
    train_file = config["data"]["train_file"]
    test_file = config["data"]["test_file"]
    output_file = config["data"]["output_file"]
    output_dir = config["data"]["output_dir"]

    get_SHAP_data(scores_file, train_file, test_file, output_file, output_dir)

    logger.info("Phase 4.4: SHAP Data completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e

