from gaitml import logger
from gaitml.models.get_CatBoost_Scores import get_CatBoost_scores
from pipeline_common import load_config, project_path


def main():
    logger.info(">>>>>>>>>> Starting Phase 4.3: CatBoost Scores <<<<<<<<<<")

    config, config_path = load_config()
    logger.info(f"Using config: {config_path}")

    catboost_config = config.get("CatBoost", {})
    sfs_results_file = project_path(config["data"]["sfs_results_excel_file_catboost"])
    combination_results_file = project_path(config["data"]["combination_results_file_catboost"])
    output_file = project_path(config["data"]["scores_file_catboost"])
    output_top3_file = project_path(config["data"]["scores_file_top3_catboost"])

    logger.info(f"SFS CatBoost file: {sfs_results_file}")
    logger.info(f"Combination CatBoost file: {combination_results_file}")
    logger.info(f"Scores CatBoost file: {output_file}")
    logger.info(f"Top3 Scores CatBoost file: {output_top3_file}")

    get_CatBoost_scores(
        sfs_results_file=sfs_results_file,
        combination_results_file=combination_results_file,
        output_file=output_file,
        output_top3_file=output_top3_file,
        top3_feature_counts=catboost_config.get("top3_feature_counts", [5, 6, 10]),
        logger=logger,
    )

    logger.info(">>>>>>>>>> Completed Phase 4.3: CatBoost Scores <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in Phase 4.3 CatBoost Scores: {e}")
        raise
