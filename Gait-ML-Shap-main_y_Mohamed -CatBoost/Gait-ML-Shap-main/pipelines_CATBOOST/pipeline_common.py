from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"


def ensure_src_path() -> None:
    src = str(SRC_PATH)
    if src not in sys.path:
        sys.path.insert(0, src)


ensure_src_path()

from gaitml.utils.helpers import read_yaml


def resolve_config_path(default_name: str = "config.yaml") -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()

    config_value = args.config or os.environ.get("CATBOOST_CONFIG") or default_name
    config_path = Path(config_value)

    if not config_path.is_absolute():
        if config_path.parts and config_path.parts[0].lower() == "configs":
            config_path = PROJECT_ROOT / config_path
        else:
            config_path = PROJECT_ROOT / "configs" / config_path

    return config_path


def load_config(default_name: str = "config.yaml") -> tuple[dict, Path]:
    config_path = resolve_config_path(default_name)
    return read_yaml(str(config_path)), config_path


def project_path(value: str | os.PathLike) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def full_run_enabled() -> bool:
    value = os.environ.get("CATBOOST_FULL_RUN", "").strip().lower()
    return value in {"1", "true", "yes", "y", "full"}


def prepare_catboost_config(config: dict, logger=None) -> dict:
    catboost_config = deepcopy(config.get("CatBoost", {}))

    if full_run_enabled():
        if logger:
            logger.info("CATBOOST_FULL_RUN enabled: using full CatBoost grid from config.")
        return catboost_config

    top_counts = catboost_config.get("top3_feature_counts", [5, 6, 10])
    catboost_config["feature_counts"] = [
        int(count) for count in top_counts if int(count) > 0
    ]
    catboost_config["iterations"] = catboost_config.get("iterations", [50, 100])[:2]
    catboost_config["depth"] = catboost_config.get("depth", [2, 3])[:2]
    catboost_config["learning_rate"] = catboost_config.get(
        "learning_rate", [0.03, 0.1]
    )[:2]
    catboost_config["l2_leaf_reg"] = catboost_config.get("l2_leaf_reg", [1, 3])[:2]
    catboost_config["bagging_temperature"] = catboost_config.get(
        "bagging_temperature", [0.0]
    )[:1]
    catboost_config["random_strength"] = catboost_config.get(
        "random_strength", [0.0]
    )[:1]
    catboost_config["n_splits"] = min(int(catboost_config.get("n_splits", 5)), 3)
    catboost_config["max_candidates"] = min(
        int(catboost_config.get("max_candidates", 50)), 8
    )

    if logger:
        logger.warning(
            "Quick CatBoost mode is active. Set CATBOOST_FULL_RUN=true for the full grid."
        )
        logger.info(
            "Quick grid: feature_counts=%s, iterations=%s, depth=%s, "
            "learning_rate=%s, max_candidates=%s, n_splits=%s",
            catboost_config["feature_counts"],
            catboost_config["iterations"],
            catboost_config["depth"],
            catboost_config["learning_rate"],
            catboost_config["max_candidates"],
            catboost_config["n_splits"],
        )

    return catboost_config
