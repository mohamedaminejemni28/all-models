"""Simple first version of the GaitML research assistant."""

import re

from .agent_config import AGENT_NAME, DEFAULT_DATASET
from .result_loader import load_available_datasets, load_model_comparison, normalize_dataset
from .langchain_gait_agent import GeminiRagUnavailable, answer_with_gemini


DATASET_ALIASES = {
    "autism": ["autism", "asd"],
    "flatfoot": ["flatfoot", "flat foot", "pes planus", "older foot"],
    "young_old": ["young", "older", "young old", "young_old", "age"],
}

RANK_WORDS = {
    "best": 1,
    "top": 1,
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "runner up": 2,
    "runner-up": 2,
    "third": 3,
    "3rd": 3,
}


class GaitModelAgent:
    """Answer questions using existing project result files."""

    def __init__(self, default_dataset: str = DEFAULT_DATASET):
        self.default_dataset = normalize_dataset(default_dataset)

    def summarize_available_models(self) -> str:
        datasets = load_available_datasets()
        if not datasets:
            return "No dataset result files were found."

        lines = [f"{AGENT_NAME} found these available results:"]
        for dataset in datasets:
            models = dataset.get("models_available", [])
            model_text = ", ".join(models) if models else "no models found"
            lines.append(f"- {dataset['name']} ({dataset['id']}): {model_text}")

        return "\n".join(lines)

    def compare_models(self, dataset: str | None = None) -> str:
        comparison = load_model_comparison(dataset or self.default_dataset)
        dataset_name = comparison["dataset_name"]
        models = self._ranked_models(comparison)

        if not models:
            return f"No model comparison data was found for {dataset_name}."

        lines = [f"Model comparison for {dataset_name}:"]
        for index, model in enumerate(models, start=1):
            cv = self._format_metric(model.get("cv_accuracy"))
            test = self._format_metric(model.get("test_accuracy"))
            f1 = self._format_metric(model.get("f1"))
            features = model.get("num_features")
            feature_text = f"{features} features" if features is not None else "features unavailable"
            lines.append(
                f"{index}. {model['model_name']}: CV accuracy {cv}, test accuracy {test}, F1 {f1}, {feature_text}"
            )

        return "\n".join(lines)

    def get_ranked_model(self, dataset: str | None = None, rank: int = 1) -> str:
        comparison = load_model_comparison(dataset or self.default_dataset)
        models = self._ranked_models(comparison)
        dataset_name = comparison["dataset_name"]

        if not models:
            return f"No CV accuracy values were found for {dataset_name}."

        if rank > len(models):
            return f"I found only {len(models)} ranked models for {dataset_name}."

        model = models[rank - 1]
        next_model = models[rank] if rank < len(models) else None
        cv = self._format_metric(model.get("cv_accuracy"))
        test = self._format_metric(model.get("test_accuracy"))
        f1 = self._format_metric(model.get("f1"))
        sensitivity = self._format_metric(model.get("sensitivity"))
        specificity = self._format_metric(model.get("specificity"))
        features = model.get("num_features")
        run_name = model.get("name_model")

        rank_label = self._rank_label(rank)
        answer = f"The {rank_label} model for {dataset_name} is {model['model_name']}"
        if run_name:
            answer += f" ({run_name})"
        answer += f" with CV accuracy {cv}"
        if features is not None:
            answer += f" using {features} selected features"
        answer += "."
        answer += (
            f" Supporting metrics: test accuracy {test}, F1 {f1}, "
            f"sensitivity {sensitivity}, specificity {specificity}."
        )

        if next_model:
            answer += (
                f" The next ranked model is {next_model['model_name']} "
                f"with CV accuracy {self._format_metric(next_model.get('cv_accuracy'))}."
            )

        return answer

    def get_best_model(self, dataset: str | None = None) -> str:
        return self.get_ranked_model(dataset=dataset, rank=1)

    def answer(self, question: str, dataset: str | None = None) -> str:
        return self.answer_structured(question, dataset)["answer"]

    def answer_structured(self, question: str, dataset: str | None = None) -> dict:
        question_lower = question.lower()
        dataset_id = self._detect_dataset(question_lower, dataset)
        rank = self._detect_rank(question_lower)

        if "available" in question_lower or "datasets" in question_lower:
            return {
                "answer": self.summarize_available_models(),
                "dataset": None,
                "intent": "available_models",
            }

        if "compare" in question_lower or "comparison" in question_lower:
            return {
                "answer": self.compare_models(dataset_id),
                "dataset": dataset_id,
                "intent": "compare_models",
            }

        if self._is_ranking_question(question_lower):
            return {
                "answer": self.get_ranked_model(dataset_id, rank),
                "dataset": dataset_id,
                "intent": "ranked_model",
            }

        try:
            return {
                "answer": answer_with_gemini(question),
                "dataset": dataset_id,
                "intent": "gemini_rag",
            }
        except GeminiRagUnavailable as exc:
            return {
                "answer": (
                    "Gemini RAG is not ready yet: "
                    f"{exc} I can still answer exact ranking questions like "
                    "who is the second best model for autism?"
                ),
                "dataset": dataset_id,
                "intent": "gemini_unavailable",
            }
        except Exception as exc:
            return {
                "answer": (
                    "Gemini is configured, but the request failed while contacting Google. "
                    f"Technical detail: {exc}. Exact ranking questions still work locally."
                ),
                "dataset": dataset_id,
                "intent": "gemini_error",
            }

    @staticmethod
    def _ranked_models(comparison: dict) -> list[dict]:
        models = [
            model for model in comparison.get("models", [])
            if model.get("cv_accuracy") is not None
        ]
        return sorted(
            models,
            key=lambda model: (
                model.get("cv_accuracy") or -1,
                model.get("f1") or -1,
                model.get("test_accuracy") or -1,
            ),
            reverse=True,
        )

    def _detect_dataset(self, question_lower: str, dataset: str | None = None) -> str:
        if dataset:
            return normalize_dataset(dataset)
        for dataset_id, aliases in DATASET_ALIASES.items():
            if any(alias in question_lower for alias in aliases):
                return normalize_dataset(dataset_id)
        return self.default_dataset

    @staticmethod
    def _detect_rank(question_lower: str) -> int:
        priority_phrases = [
            ("runner-up", 2),
            ("runner up", 2),
            ("second", 2),
            ("2nd", 2),
            ("third", 3),
            ("3rd", 3),
            ("first", 1),
            ("1st", 1),
            ("best", 1),
            ("top", 1),
        ]
        for phrase, rank in priority_phrases:
            if phrase in question_lower:
                return rank
        match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", question_lower)
        if match:
            return max(1, int(match.group(1)))
        return 1

    @staticmethod
    def _is_ranking_question(question_lower: str) -> bool:
        rank_terms = ["best", "top", "highest", "second", "third", "rank", "runner"]
        model_terms = ["model", "classifier", "algorithm"]
        return any(term in question_lower for term in rank_terms) and any(
            term in question_lower for term in model_terms
        )

    @staticmethod
    def _rank_label(rank: int) -> str:
        labels = {1: "best", 2: "second-best", 3: "third-best"}
        return labels.get(rank, f"rank #{rank}")

    @staticmethod
    def _format_metric(value: float | None) -> str:
        if value is None:
            return "unavailable"
        return f"{value * 100:.1f}%"
