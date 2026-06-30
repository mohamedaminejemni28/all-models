from __future__ import annotations

import os
from pathlib import Path


PLATFORM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLATFORM_ROOT.parent
DEFAULT_MLFLOW_DB = REPO_ROOT / "mlflow.db"
DEFAULT_MLRUNS_DIR = REPO_ROOT / "mlruns"


def default_tracking_uri() -> str:
    """Return the local MLflow tracking URI used by this repository."""
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    if DEFAULT_MLFLOW_DB.exists():
        return f"sqlite:///{DEFAULT_MLFLOW_DB.as_posix()}"

    return f"file:///{DEFAULT_MLRUNS_DIR.as_posix()}"


APP_TITLE = "XGBoost MLflow Results Platform"
XGBOOST_TOKEN = "xgboost"

