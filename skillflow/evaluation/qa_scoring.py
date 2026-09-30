from __future__ import annotations

import re
import string
from collections.abc import Sequence

def normalize_searchqa_answer(value: object) -> str:
    text = str(value or "").casefold()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def searchqa_exact_match(prediction: object, golden_answers: Sequence[object]) -> bool:
    normalized = normalize_searchqa_answer(prediction)
    return any(
        normalized == normalize_searchqa_answer(answer)
        for answer in golden_answers
    )


def format_retrieved_documents(payload: object) -> str:
    if isinstance(payload, str):
        return payload.strip()
    docs = payload
    if isinstance(docs, dict) and "result" in docs:
        docs = docs["result"]
    while isinstance(docs, list) and len(docs) == 1 and isinstance(docs[0], list):
        docs = docs[0]
    if not isinstance(docs, list):
        raise RuntimeError("SearchQA retrieval response has no document list")
    rendered: list[str] = []
    for index, item in enumerate(docs, start=1):
        if isinstance(item, dict) and isinstance(item.get("document"), dict):
            item = item["document"]
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        content = str(item.get("contents") or item.get("text") or "").strip()
        if not title and not content:
            continue
        heading = f"Doc {index}: {title}" if title else f"Doc {index}"
        rendered.append(f"{heading}\n{content}".strip())
    return "\n\n".join(rendered)

