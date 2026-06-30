"""
Pydantic response schemas for the API.
"""

from pydantic import BaseModel
from typing import Optional


class ModelSummary(BaseModel):
    id: str
    name: str
    short_name: str
    description: str
    color: str
    icon: str
    hyperparameters: list[str]
    datasets_available: list[str]


class MetricRow(BaseModel):
    name_model: Optional[str] = None
    num_features: Optional[int] = None
    cv_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    f1: Optional[float] = None
    mcc: Optional[float] = None
    npv: Optional[float] = None
    ppv: Optional[float] = None
    likelihood_ratio: Optional[float] = None
    confusion_matrix: Optional[str] = None
    features: Optional[str] = None
    # Hyperparameters (present depending on model)
    c_value: Optional[float] = None
    gamma_value: Optional[float] = None
    n_estimators: Optional[int] = None
    max_depth: Optional[int] = None
    learning_rate: Optional[float] = None
    subsample: Optional[float] = None
    colsample_bytree: Optional[float] = None
    min_samples_split: Optional[int] = None
    min_samples_leaf: Optional[int] = None
    max_features: Optional[str] = None
    cv_split: Optional[int] = None
    random_state: Optional[int] = None


class MetricsResponse(BaseModel):
    model_id: str
    model_name: str
    dataset: str
    dataset_name: str
    top3: list[MetricRow]
    total_runs: Optional[int] = None


class FeatureEntry(BaseModel):
    rank: int
    name: str
    score: Optional[float] = None


class FeatureResponse(BaseModel):
    model_id: str
    dataset: str
    features: list[FeatureEntry]
    sheet_name: Optional[str] = None


class FigureItem(BaseModel):
    filename: str
    url: str
    category: str  # "shap" or "scatter"
    dataset: str


class FigureResponse(BaseModel):
    model_id: str
    figures: list[FigureItem]


class ComparisonEntry(BaseModel):
    model_id: str
    model_name: str
    color: str
    cv_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    f1: Optional[float] = None
    mcc: Optional[float] = None
    npv: Optional[float] = None
    ppv: Optional[float] = None
    num_features: Optional[int] = None
    model_run_name: Optional[str] = None


class ComparisonResponse(BaseModel):
    dataset: str
    dataset_name: str
    models: list[ComparisonEntry]


class DatasetInfo(BaseModel):
    id: str
    name: str
    prefix: str
    description: str
    classes: dict[str, str]
    models_available: list[str]


class ExperimentFile(BaseModel):
    filename: str
    file_type: str
    model: str
    dataset: str
    category: str
    size_bytes: Optional[int] = None
