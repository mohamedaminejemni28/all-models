import os
import sys

# Add project src directory to Python path first
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from gaitml import logger
from gaitml.utils.helpers import read_yaml
from gaitml.models.get_Random_forest_scores import get_Random_forest_scores
def main():
    logger.info(">>>>>>>>>> Starting Phase 4.3: Scores Random Forest <<<<<<<<<<")

    config_path = "configs/config.yaml"
    config = read_yaml(config_path)

    sfs_results_file = (
        config["data"].get("sfs_results_excel_file_random_forest")
        or config["data"].get("sfs_results_excel_file_Random_forest")
        or config["SFS"].get("Phase4_1_SFS_scores_file")
    )

    combination_results_file = config["data"]["combination_results_file_random_forest"]
    output_file = config["data"]["scores_file_random_forest"]

    logger.info(f"SFS Random Forest file: {sfs_results_file}")
    logger.info(f"Combination Random Forest file: {combination_results_file}")
    logger.info(f"Output scores file: {output_file}")
    logger.info(f"Using SRC_PATH = {SRC_PATH}")

    get_Random_forest_scores(
        sfs_results_file=sfs_results_file,
        combination_results_file=combination_results_file,
        output_file=output_file,
        logger=logger
    )

    logger.info(">>>>>>>>>> Completed Phase 4.3: Scores Random Forest <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in Phase 4.3 Random Forest: {e}")
        raise e