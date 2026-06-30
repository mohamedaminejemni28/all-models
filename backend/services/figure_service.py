"""
figure_service.py

Scans SHAP and time-variable plot directories for available PNG figures
and returns structured lists with URLs that the frontend can display.
"""

import logging
from pathlib import Path

from utils.file_paths import (
    DATASETS,
    MODEL_META,
    get_shap_figures_dir,
    get_time_variable_plots_dir,
)

logger = logging.getLogger(__name__)


def _scan_png_files(directory: Path) -> list[str]:
    """Return sorted list of PNG filenames in a directory."""
    if not directory.exists():
        return []
    return sorted([f.name for f in directory.iterdir() if f.suffix.lower() == ".png"])


def get_figures_for_model(model_id: str) -> dict:
    """
    Collect all available figures (SHAP + scatter plots) for a model
    across all datasets.
    """
    figures = []

    for ds_id in DATASETS:
        # SHAP summary plots
        shap_dir = get_shap_figures_dir(model_id, ds_id)
        for filename in _scan_png_files(shap_dir):
            figures.append({
                "filename": filename,
                "url": f"/api/figures/{model_id}/{ds_id}/shap/{filename}",
                "category": "shap",
                "dataset": ds_id,
            })

        # Time-variable scatter plots
        scatter_dir = get_time_variable_plots_dir(model_id, ds_id)
        for filename in _scan_png_files(scatter_dir):
            figures.append({
                "filename": filename,
                "url": f"/api/figures/{model_id}/{ds_id}/scatter/{filename}",
                "category": "scatter",
                "dataset": ds_id,
            })

    return {
        "model_id": model_id,
        "figures": figures,
    }


def get_figure_path(model_id: str, dataset_id: str, category: str, filename: str) -> Path | None:
    """
    Resolve a figure request to an absolute file path.
    Returns None if the file doesn't exist.
    """
    if category == "shap":
        base_dir = get_shap_figures_dir(model_id, dataset_id)
    elif category == "scatter":
        base_dir = get_time_variable_plots_dir(model_id, dataset_id)
    else:
        return None

    file_path = base_dir / filename

    # Security: ensure the resolved path is inside the expected directory
    try:
        file_path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        logger.warning(f"Path traversal attempt: {filename}")
        return None

    if file_path.exists() and file_path.is_file():
        return file_path

    return None
