import os
from functools import cache

from transformers import pipeline

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@cache
def _embedder():
    model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return pipeline("feature-extraction", model=model)


def embed_text_en(text_en: str) -> list[float]:
    token_vectors = _embedder()(text_en)[0]
    count = len(token_vectors)
    width = len(token_vectors[0])
    return [
        sum(token[index] for token in token_vectors) / count for index in range(width)
    ]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return dot / (left_norm * right_norm)


def average_similarity(texts: list[str]) -> float | None:
    vectors = [embed_text_en(text) for text in texts if text]
    if len(vectors) < 2:
        return None
    total = 0.0
    pairs = 0
    for i, left in enumerate(vectors):
        for right in vectors[i + 1 :]:
            total += _cosine(left, right)
            pairs += 1
    return total / pairs
