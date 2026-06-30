"""Gemini + LangChain assistant for normal GaitML questions.

Exact model ranking questions stay handled by the deterministic GaitModelAgent.
This module handles general explanations with Gemini and a small local knowledge
retriever, so it does not depend on remote embedding models.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

BLOCKED_LOCAL_PROXY = "http://127.0.0.1:9"
PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

SYSTEM_GUIDANCE = """
You are the GaitML Research Assistant. Answer clearly and naturally.
Use the retrieved context when it is relevant. If the question asks for exact
model rankings, metrics, or result values, say that deterministic result tools
should be used rather than guessing. Keep answers concise but useful for a
researcher writing a gait-analysis report or thesis.
""".strip()


class GeminiRagUnavailable(RuntimeError):
    """Raised when Gemini cannot be used in the current environment."""


def _clear_blocked_local_proxy() -> None:
    for name in PROXY_ENV_NAMES:
        if os.environ.get(name) == BLOCKED_LOCAL_PROXY:
            os.environ.pop(name, None)


def _get_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or next(
        (value for key, value in os.environ.items() if key.lstrip("\ufeff") == "GOOGLE_API_KEY"),
        None,
    )


def _load_documents() -> list[dict[str, str]]:
    documents = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        documents.append({"content": path.read_text(encoding="utf-8"), "source": path.name})
    return documents


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_+-]+", text.lower()) if len(token) > 2}


def _format_context(question: str, documents: list[dict[str, str]]) -> str:
    question_tokens = _tokens(question)
    scored = []
    for document in documents:
        content_tokens = _tokens(document["content"])
        score = len(question_tokens & content_tokens)
        scored.append((score, document))

    selected = [document for score, document in sorted(scored, key=lambda item: item[0], reverse=True)[:4]]
    if not selected:
        selected = documents[:4]

    return "\n\n".join(
        f"Source: {document['source']}\n{document['content']}" for document in selected
    )


@lru_cache(maxsize=1)
def _build_chain():
    _clear_blocked_local_proxy()
    api_key = _get_api_key()
    if not api_key:
        raise GeminiRagUnavailable("GOOGLE_API_KEY is not set.")

    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise GeminiRagUnavailable(
            "LangChain Gemini dependencies are not installed. Run pip install -r requirements.txt."
        ) from exc

    documents = _load_documents()
    if not documents:
        raise GeminiRagUnavailable("No knowledge documents were found.")

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=api_key,
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_GUIDANCE),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"),
    ])

    return (
        {
            "context": RunnableLambda(lambda question: _format_context(question, documents)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def answer_with_gemini(question: str) -> str:
    chain = _build_chain()
    return chain.invoke(question)


def is_available() -> bool:
    try:
        _build_chain()
        return True
    except GeminiRagUnavailable:
        return False

