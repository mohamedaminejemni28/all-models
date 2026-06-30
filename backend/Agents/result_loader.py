"""Read model summaries for the agent from the existing backend services."""

from services.data_loader import get_comparison_data, get_dataset_info_list
from utils.file_paths import DATASETS, MODEL_META


def normalize_dataset(dataset: str | None) -> str:
    """Return a valid dataset id, falling back to flatfoot."""
    if dataset in DATASETS:
        return dataset
    return "flatfoot"


def load_available_datasets() -> list[dict]:
    """Return datasets and the models that have files available."""
    return get_dataset_info_list()


def load_model_comparison(dataset: str | None = None) -> dict:
    """Return the existing comparison payload for one dataset."""
    dataset_id = normalize_dataset(dataset)
    return get_comparison_data(dataset_id)


def get_model_label(model_id: str) -> str:
    """Return a readable model name."""
    return MODEL_META.get(model_id, {}).get("name", model_id)

