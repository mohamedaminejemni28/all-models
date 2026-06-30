"""Run the full CNN gait-classification pipeline."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from gaitml import logger
from gaitml.utils.helpers import read_yaml


def main():
    parser = argparse.ArgumentParser(description="Run CNN experiments for gait datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "autism_2024", "young_old_2024", "flatfoot_control_older_2024"],
        default="all",
        help="Dataset to run. Use 'all' for every dataset in configs/cnn_config.yaml.",
    )
    parser.add_argument("--config", default="configs/cnn_config.yaml", help="CNN config YAML path.")
    args = parser.parse_args()

    config = read_yaml(os.path.join(PROJECT_ROOT, args.config))
    datasets = list(config["datasets"]) if args.dataset == "all" else [args.dataset]

    try:
        from gaitml.deep_learning.cnn_experiment import run_dataset
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "PyTorch is required for the CNN pipeline. Install dependencies with: "
                "pip install -r requirements.txt"
            ) from exc
        raise

    for dataset_name in datasets:
        logger.info(f">>>>> Running CNN pipeline for {dataset_name} <<<<<")
        scores_path, top3_path = run_dataset(dataset_name, config, PROJECT_ROOT)
        logger.info(f"Saved CNN scores: {scores_path}")
        logger.info(f"Saved CNN Top 3: {top3_path}")


if __name__ == "__main__":
    main()
