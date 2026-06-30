from __future__ import annotations

import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


def artifact_uri_to_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None

    path = unquote(parsed.path)
    if parsed.netloc:
        path = f"//{parsed.netloc}{path}"

    # Windows file URIs often parse as /C:/Users/...
    if len(path) >= 4 and path[0] == "/" and path[2] == ":":
        path = path[1:]

    return Path(path)


def list_artifacts_recursive(
    client: MlflowClient,
    run_id: str,
    path: str | None = None,
) -> list[dict]:
    items: list[dict] = []
    for artifact in client.list_artifacts(run_id, path):
        if artifact.is_dir:
            items.extend(list_artifacts_recursive(client, run_id, artifact.path))
        else:
            items.append(
                {
                    "path": artifact.path,
                    "file_name": Path(artifact.path).name,
                    "size": artifact.file_size,
                }
            )
    return items


def get_local_artifact_path(
    run_artifact_uri: str,
    artifact_path: str,
    run_id: str,
) -> Path | None:
    root = artifact_uri_to_path(run_artifact_uri)
    if root is not None:
        candidate = root / artifact_path
        if candidate.exists():
            return candidate

    try:
        downloaded = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=tempfile.mkdtemp(prefix="xgb_mlflow_artifacts_"),
        )
        return Path(downloaded)
    except Exception:
        return None


def find_artifacts(artifacts: list[dict], suffixes: tuple[str, ...], contains: str = "") -> list[dict]:
    contains_lower = contains.lower()
    matches = []
    for artifact in artifacts:
        artifact_path = artifact["path"].lower()
        if not artifact_path.endswith(suffixes):
            continue
        if contains_lower and contains_lower not in artifact_path:
            continue
        matches.append(artifact)
    return matches


def read_excel_artifact(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)

