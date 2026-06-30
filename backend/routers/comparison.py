"""
routers/comparison.py

API endpoint for side-by-side model comparison:
  GET /api/comparison?dataset=autism
"""

from fastapi import APIRouter, Query, HTTPException

from services.data_loader import get_comparison_data
from utils.file_paths import DATASETS

router = APIRouter(prefix="/api", tags=["comparison"])

VALID_DATASETS = set(DATASETS.keys())


@router.get("/comparison")
def compare_models(dataset: str = Query(default="autism")):
    """
    Return a side-by-side comparison of the best model run from each
    of all configured models for the given dataset.
    """
    if dataset not in VALID_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset '{dataset}'. Valid: {list(VALID_DATASETS)}",
        )
    return get_comparison_data(dataset)
