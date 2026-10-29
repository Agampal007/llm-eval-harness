"""Per-example scoring and aggregation with bootstrap confidence intervals.

Metrics, all in [0, 1] and all higher-is-better:

hit@1, hit@3   a gold document is among the first 1 / 3 retrieved (answerable questions)
mrr            reciprocal rank of the first gold document in the top 5, else 0
contains_ref   the answer contains the reference answer as a run of tokens
ref_recall     share of the reference's tokens that appear in the answer
faithfulness   share of the answer's content words found in the retrieved contexts. It measures
               grounding, not correctness: a copied but irrelevant sentence is fully faithful.
citation_precision   share of cited documents that are gold documents
abstain_accuracy     abstained exactly when the question is unanswerable (all questions)
"""

from __future__ import annotations

import math

import numpy as np

from .data import Example
from .systems import Response
from .text import contains_sequence, content_tokens, tokens

METRICS = (
    "hit@1",
    "hit@3",
    "mrr",
    "contains_ref",
    "ref_recall",
    "faithfulness",
    "citation_precision",
    "abstain_accuracy",
)


def score_example(example: Example, response: Response) -> dict[str, float | None]:
    scores: dict[str, float | None] = dict.fromkeys(METRICS)
    scores["abstain_accuracy"] = float(response.abstained == (not example.answerable))

    if not response.abstained:
        answer_terms = content_tokens(response.answer)
        context_terms = {t for text in response.contexts for t in tokens(text)}
        if answer_terms:
            scores["faithfulness"] = sum(t in context_terms for t in answer_terms) / len(answer_terms)

    if not example.answerable:
        return scores

    gold = set(example.gold_docs)
    rank = next((i for i, doc_id in enumerate(response.retrieved, start=1) if doc_id in gold), None)
    scores["hit@1"] = float(rank == 1)
    scores["hit@3"] = float(rank is not None and rank <= 3)
    scores["mrr"] = 1.0 / rank if rank is not None and rank <= 5 else 0.0

    if response.abstained:
        scores["contains_ref"] = 0.0
        scores["ref_recall"] = 0.0
        return scores

    answer_tokens, ref_tokens = tokens(response.answer), tokens(example.reference or "")
    scores["contains_ref"] = float(contains_sequence(answer_tokens, ref_tokens))
    if ref_tokens:
        answer_set = set(answer_tokens)
        scores["ref_recall"] = sum(t in answer_set for t in ref_tokens) / len(ref_tokens)
    scores["citation_precision"] = (
        sum(c in gold for c in response.cited) / len(response.cited) if response.cited else 0.0
    )
    return scores


def bootstrap_ci(
    values: list[float], n_resamples: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean. Deterministic for a given seed."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_resamples, arr.size))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def aggregate(per_example: list[dict[str, float | None]]) -> dict[str, dict[str, float | int]]:
    """Mean, 95% bootstrap interval and sample size per metric; None values are skipped."""
    summary: dict[str, dict[str, float | int]] = {}
    for name in METRICS:
        values = [s[name] for s in per_example if s[name] is not None]
        if not values:
            summary[name] = {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
            continue
        low, high = bootstrap_ci(values)
        summary[name] = {"mean": float(np.mean(values)), "ci_low": low, "ci_high": high, "n": len(values)}
    return summary
