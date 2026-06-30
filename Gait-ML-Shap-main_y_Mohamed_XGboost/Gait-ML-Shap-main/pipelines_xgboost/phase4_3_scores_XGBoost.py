import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from gaitml import logger
from gaitml.utils.helpers import read_yaml
from gaitml.models.get_XGBoost_Scores import get_XGBoost_scores


def main():
    logger.info(">>>>>>>>>> Starting Phase 4.3: Scores XGBoost <<<<<<<<<<")


    config_path = "configs/config.yaml"
    config = read_yaml(config_path)


    sfs_results_file = config["data"]["sfs_results_excel_file_xgboost"]
    combination_results_file = config["data"]["combination_results_file_xgboost"]
    output_file = config["data"]["scores_file_xgboost"]

    logger.info(f"SFS XGBoost file: {sfs_results_file}")
    logger.info(f"Combination XGBoost file: {combination_results_file}")
    logger.info(f"Output scores file: {output_file}")
    logger.info(f"Using SRC_PATH = {SRC_PATH}")


    get_XGBoost_scores(
        sfs_results_file=sfs_results_file,
        combination_results_file=combination_results_file,
        output_file=output_file,
        logger=logger
    )

    logger.info(">>>>>>>>>> Completed Phase 4.3: Scores XGBoost <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in Phase 4.3 XGBoost: {e}")
        raise e