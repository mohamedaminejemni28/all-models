"""
routers/models.py

API endpoints for individual model data:
  GET /api/models                          — list all models
  GET /api/models/{model_name}/metrics     — Top-3 metrics per dataset
  GET /api/models/{model_name}/features    — SFS selected features
  GET /api/models/{model_name}/figures     — SHAP & scatter plot URLs
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from services.data_loader import get_model_summaries, get_top3_metrics, get_sfs_features
from services.figure_service import get_figures_for_model, get_figure_path
from utils.file_paths import MODEL_META, DATASETS

router = APIRouter(prefix="/api", tags=["models"])

VALID_MODELS = set(MODEL_META.keys())
VALID_DATASETS = set(DATASETS.keys())


def _validate_model(model_name: str) -> str:
    """Normalise and validate model name."""
    key = model_name.lower().replace("-", "_").replace(" ", "_")
    if key not in VALID_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Valid: {list(VALID_MODELS)}")
    return key


@router.get("/models")
def list_models():
    """Return summary metadata for all configured models."""
    return get_model_summaries()


@router.get("/models/{model_name}/metrics")
def model_metrics(model_name: str, dataset: str = Query(default="autism")):
    """
    Return Top-3 best model runs for a given model and dataset.
    Query param `dataset` can be: autism, young_old, flatfoot.
    """
    model_id = _validate_model(model_name)
    if dataset not in VALID_DATASETS:
        raise HTTPException(status_code=400, detail=f"Invalid dataset. Valid: {list(VALID_DATASETS)}")
    return get_top3_metrics(model_id, dataset)


@router.get("/models/{model_name}/features")
def model_features(model_name: str, dataset: str = Query(default="autism")):
    """Return SFS-selected features for a model and dataset."""
    model_id = _validate_model(model_name)
    if dataset not in VALID_DATASETS:
        raise HTTPException(status_code=400, detail=f"Invalid dataset. Valid: {list(VALID_DATASETS)}")
    return get_sfs_features(model_id, dataset)


@router.get("/models/{model_name}/figures")
def model_figures(model_name: str):
    """Return all available SHAP and scatter plot figure URLs for a model."""
    model_id = _validate_model(model_name)
    return get_figures_for_model(model_id)


@router.get("/figures/{model_name}/{dataset_id}/{category}/{filename}")
def serve_figure(model_name: str, dataset_id: str, category: str, filename: str):
    """Serve a specific figure file (PNG) by model, dataset, category, and filename."""
    model_id = _validate_model(model_name)
    if dataset_id not in VALID_DATASETS:
        raise HTTPException(status_code=400, detail="Invalid dataset")
    if category not in ("shap", "scatter"):
        raise HTTPException(status_code=400, detail="Category must be 'shap' or 'scatter'")

    path = get_figure_path(model_id, dataset_id, category, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Figure not found")

    return FileResponse(path, media_type="image/png")
