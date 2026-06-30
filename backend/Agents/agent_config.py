"""Configuration used by the GaitML research assistant."""

DEFAULT_DATASET = "young_old"

AGENT_NAME = "GaitML Research Assistant"

SUPPORTED_INTENTS = {
    "available_models": "List models with available result files.",
    "best_model": "Find the strongest model for a dataset using existing metrics.",
    "compare_models": "Compare the best run from each model for a dataset.",
}

