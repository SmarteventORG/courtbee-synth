from functools import cache

from transformers import pipeline

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@cache
def _embedder():
    return pipeline("feature-extraction", model=MODEL)


def embed_text_en(text_en: str) -> list[float]:
    token_vectors = _embedder()(text_en)[0]
    count = len(token_vectors)
    width = len(token_vectors[0])
    return [
        sum(token[index] for token in token_vectors) / count for index in range(width)
    ]
