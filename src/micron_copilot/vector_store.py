"""Dependency-free TF-IDF vector store with cosine retrieval."""
from __future__ import annotations

import math
import re
from collections import Counter

from .corpus import Document

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class VectorStore:
    """In-memory TF-IDF index. Deterministic; no external embedding service."""

    def __init__(self, documents: list[Document]):
        self.documents = list(documents)
        self._by_id = {d.doc_id: d for d in self.documents}
        df: Counter[str] = Counter()
        self._tf: list[Counter[str]] = []
        for doc in self.documents:
            counts = Counter(tokenize(f"{doc.title} {doc.text}"))
            self._tf.append(counts)
            df.update(counts.keys())
        n = max(len(self.documents), 1)
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self._vectors = [self._vectorise(c) for c in self._tf]

    def _vectorise(self, counts: Counter[str]) -> dict[str, float]:
        vec = {t: (1 + math.log(c)) * self._idf.get(t, 1.0)
               for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def get(self, doc_id: str) -> Document | None:
        return self._by_id.get(doc_id)

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        q = self._vectorise(Counter(tokenize(query)))
        scored = []
        for doc, vec in zip(self.documents, self._vectors):
            score = sum(q.get(t, 0.0) * v for t, v in vec.items()
                        if t in q)
            if score > 0:
                scored.append((doc, round(score, 6)))
        scored.sort(key=lambda x: (-x[1], x[0].doc_id))
        return scored[:k]

    def __len__(self) -> int:
        return len(self.documents)
