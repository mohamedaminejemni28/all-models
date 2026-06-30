import os
import sys

# Add NEW copied project src directory to Python path FIRST
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from gaitml import logger
from gaitml.utils.helpers import read_yaml

from gaitml.models.get_LightGBM_Scores import (
    get_lightgbm_scores,
)


def main():
    logger.info(">>>>>>>>>> Starting Phase 4.3: LightGBM Scores <<<<<<<<<<")

    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    config = read_yaml(config_path)

    combination_results_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["combination_results_file_lightgbm"],
    )

    output_scores_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["scores_file_lightgbm"],
    )

    output_top3_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["scores_file_top3_lightgbm"],
    )

    lightgbm_config = config.get("LightGBM", {})

    top3_feature_counts = lightgbm_config.get(
        "top3_feature_counts",
        [5, 6, 10],
    )

    logger.info(f"LightGBM Step 2 file: {combination_results_file}")
    logger.info(f"Output scores file: {output_scores_file}")
    logger.info(f"Output Top 3 file: {output_top3_file}")
    logger.info(f"Top 3 feature counts: {top3_feature_counts}")
    logger.info(f"Using SRC_PATH = {SRC_PATH}")

    get_lightgbm_scores(
        combination_results_file=combination_results_file,
        output_scores_file=output_scores_file,
        output_top3_file=output_top3_file,
        top3_feature_counts=top3_feature_counts,
        logger=logger,
    )

    logger.info(">>>>>>>>>> Completed Phase 4.3: LightGBM Scores <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in Phase 4.3 LightGBM: {e}")
        raise e
