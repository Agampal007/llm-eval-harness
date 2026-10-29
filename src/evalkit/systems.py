"""RAG systems under test.

A system is anything with `answer(question) -> Response`. Bring your own with
`evalkit run --system my_module:factory`, where `factory(corpus: list[Doc])` returns a system.
"""

from __future__ import annotations

import importlib
import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .data import Doc
from .text import content_tokens, sentences

ABSTAIN_TEXT = "I don't know."


@dataclass
class Response:
    answer: str
    retrieved: list[str] = field(default_factory=list)  # ranked document ids
    contexts: list[str] = field(default_factory=list)  # texts the answer may draw on
    cited: list[str] = field(default_factory=list)  # document ids the answer cites
    abstained: bool = False


class System(Protocol):
    def answer(self, question: str) -> Response: ...


class BM25Index:
    """Okapi BM25 over whole documents (Lucene-style idf), small enough to stay pure Python."""

    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self._tf = [Counter(content_tokens(d.title + " " + d.text)) for d in docs]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / len(docs)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = content_tokens(query)
        out = []
        for tf, length in zip(self._tf, self._len, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self._avg)
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        s = self.scores(query)
        order = sorted(range(len(s)), key=lambda i: (-s[i], i))
        return [(i, s[i]) for i in order[:k]]

    def coverage(self, query: str, doc_index: int) -> float:
        """Share of the query's content terms that appear in the document."""
        terms = set(content_tokens(query))
        if not terms:
            return 0.0
        return len(terms & set(self._tf[doc_index])) / len(terms)


class BM25Extractive:
    """BM25 retrieval, then the best-matching sentence from the top documents as the answer.

    Abstains when the top document covers too little of the question's vocabulary. The default
    threshold was chosen by looking at this repo's golden set, so treat it as tuned on it.
    """

    name = "bm25-extractive"

    def __init__(self, docs: list[Doc], k: int = 3, min_coverage: float = 0.4):
        self.docs, self.k, self.min_coverage = docs, k, min_coverage
        self.index = BM25Index(docs)

    def answer(self, question: str) -> Response:
        ranked = self.index.rank(question, self.k)
        top_doc = ranked[0][0]
        retrieved = [self.docs[i].id for i, _ in ranked]
        contexts = [self.docs[i].text for i, _ in ranked]
        if self.index.coverage(question, top_doc) < self.min_coverage:
            return Response(ABSTAIN_TEXT, retrieved, contexts, [], abstained=True)
        q_terms = content_tokens(question)
        best, best_score, best_doc = "", -1.0, top_doc
        for rank, (i, _) in enumerate(ranked):
            for sent in sentences(self.docs[i].text):
                present = set(content_tokens(sent))
                score = sum(self.index.idf.get(t, 0.0) for t in q_terms if t in present) - 0.01 * rank
                if score > best_score:
                    best, best_score, best_doc = sent, score, i
        return Response(best, retrieved, contexts, [self.docs[best_doc].id])


class BM25FirstSentence(BM25Extractive):
    """Same retrieval, but a weaker answerer: always the first sentence of the top document."""

    name = "bm25-first-sentence"

    def answer(self, question: str) -> Response:
        base = super().answer(question)
        if base.abstained:
            return base
        top = self.docs[self.index.rank(question, 1)[0][0]]
        return Response(sentences(top.text)[0], base.retrieved, base.contexts, [top.id])


class RandomRetrieval:
    """Control: random documents, random sentence. A metric that cannot tell this from BM25 is broken."""

    name = "random-retrieval"

    def __init__(self, docs: list[Doc], k: int = 3, seed: int = 0):
        self.docs, self.k = docs, k
        self.rng = random.Random(seed)

    def answer(self, question: str) -> Response:
        picked = self.rng.sample(self.docs, self.k)
        sent = self.rng.choice(sentences(picked[0].text))
        return Response(sent, [d.id for d in picked], [d.text for d in picked], [picked[0].id])


class CannedAnswer:
    """Control: ignores the corpus and always returns a fluent claim that nothing supports."""

    name = "canned-answer"

    def __init__(self, docs: list[Doc]):
        self.docs = docs
        self.index = BM25Index(docs)

    def answer(self, question: str) -> Response:
        ranked = self.index.rank(question, 3)
        return Response(
            "It is widely used across modern cloud platforms and is generally considered best practice.",
            [self.docs[i].id for i, _ in ranked],
            [self.docs[i].text for i, _ in ranked],
            [self.docs[ranked[0][0]].id],
        )


SYSTEMS: dict[str, Callable[[list[Doc]], System]] = {
    "bm25-extractive": BM25Extractive,
    "bm25-first-sentence": BM25FirstSentence,
    "random-retrieval": RandomRetrieval,
    "canned-answer": CannedAnswer,
}


def build_system(spec: str, docs: list[Doc]) -> System:
    """Resolve a built-in name or a `module:factory` path into a system."""
    if spec in SYSTEMS:
        return SYSTEMS[spec](docs)
    if ":" in spec:
        module_name, _, factory_name = spec.partition(":")
        factory = getattr(importlib.import_module(module_name), factory_name)
        return factory(docs)
    raise KeyError(f"unknown system {spec!r}; built-ins: {', '.join(SYSTEMS)} (or use module:factory)")


def system_label(spec: str) -> str:
    return spec.replace(":", "-").replace(".", "-")
