import os
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gaitml import logger
from gaitml.utils.helpers import read_yaml
from gaitml.features.sequential_feature_selection import sequential_feature_selection

def main():
    logger.info(">>>>>>>>>> Starting Phase 4.1: Sequential Feature Selection <<<<<<<<<<")

    config_path = "configs/config.yaml"
    config = read_yaml(config_path)
    dataset_train_file = config["SFS"]["dataset_train_file"]
    output_SFS_scores_file = config["SFS"]["Phase4_1_SFS_scores_file"]
    labels_column = config["SFS"]["labels_column"] 
    classes_dict = config["SFS"]["classes"]
    logger.info(f"Working with classes_dict = {classes_dict}")

    logger.info("Executing sequencial seature selection...")
    sequential_feature_selection(
        input_excel_file = dataset_train_file,
        output_excel_file = output_SFS_scores_file,
        group_column = labels_column, # Name of the target column.
        classes_dict = classes_dict, # Dictionary for label encoding classes (target variables)
        n_features = 50, #  Number of features to select.
        c_set = [1, 10], # List of C values for SVM.
        gamma_set = [0.1, 1, 10], # List of gamma values for SVM.
        kfold_n_splits = 10, # Number of folds for StratifiedKFold.
        random_state = 0
    )  
    
    logger.info(">>>>>>>>>> Completed Phase 4.1: Sequential Feature Selection <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")     
        raise e
