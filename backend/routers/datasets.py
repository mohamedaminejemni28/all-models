"""
routers/datasets.py

API endpoints for dataset and experiment metadata:
  GET /api/datasets      — list all datasets with class info
  GET /api/experiments   — list all available experiment result files
"""

from fastapi import APIRouter

from services.data_loader import get_dataset_info_list, get_experiment_files

router = APIRouter(prefix="/api", tags=["datasets"])


@router.get("/datasets")
def list_datasets():
    """Return info about all datasets with which models have results."""
    return get_dataset_info_list()


@router.get("/experiments")
def list_experiments():
    """Return a flat list of all experiment result files with metadata."""
    return get_experiment_files()
