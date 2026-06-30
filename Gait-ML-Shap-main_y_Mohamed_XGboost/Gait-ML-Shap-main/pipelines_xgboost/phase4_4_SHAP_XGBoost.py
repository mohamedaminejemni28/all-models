import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from gaitml import logger
from gaitml.utils.helpers import read_yaml
from gaitml.models.get_SHAP_data_XGBoost import get_SHAP_data_XGBoost


def main():
    logger.info(">>>>>>>>>> Starting Phase 4.4: XGBoost SHAP <<<<<<<<<<")

    config_path = "configs/config.yaml"
    config = read_yaml(config_path)

    scores_file = config["data"]["scores_file_top3_xgboost"]
    train_file = config["SFS"]["dataset_train_file"]
    test_file = config["SFS"]["dataset_test_file"]
    output_file = config["data"]["output_file"]
    output_dir = config["data"]["output_dir"]
    labels_column = config["SFS"]["labels_column"]

    get_SHAP_data_XGBoost(
        scores_file=scores_file,
        train_file=train_file,
        test_file=test_file,
        output_file=output_file,
        output_dir=output_dir,
        labels_column=labels_column
    )

    logger.info(">>>>>>>>>> Completed Phase 4.4: XGBoost SHAP <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e