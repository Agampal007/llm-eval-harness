"""Turn the SQuAD v2.0 dev set into an evalkit corpus and golden set.

SQuAD v2.0 (Rajpurkar, Jia and Liang, 2018) pairs paragraphs from Wikipedia with crowd-written
questions. About half the questions are unanswerable: they were written to look answerable from a
paragraph that does not contain the answer. That makes it a real test of abstention, which the
bundled golden set (4 unanswerable questions) is not.
"""

from __future__ import annotations

import random
import re

from .text import contains_sequence, tokens

SQUAD_V2_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
SQUAD_V2_DEV_SHA256 = "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_sample(
    squad: dict, n_articles: int, n_answerable: int, n_unanswerable: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """Sample articles, use all their paragraphs as the corpus, then sample questions from them.

    Answerable questions are kept only when their first gold answer appears as a run of tokens in
    the paragraph: SQuAD answers are character spans, and a span cut mid-word (for example
    "Norman" inside "Normans") could never score on `contains_ref`.

    Returns (corpus rows, golden rows) in the JSONL shapes that `evalkit.data` loads.
    """
    rng = random.Random(seed)
    articles = sorted(rng.sample(squad["data"], n_articles), key=lambda a: a["title"])

    corpus: list[dict] = []
    answerable: list[dict] = []
    unanswerable: list[dict] = []
    for article in articles:
        slug = _slug(article["title"])
        for p_index, paragraph in enumerate(article["paragraphs"]):
            doc_id = f"{slug}-p{p_index}"
            corpus.append(
                {"id": doc_id, "title": article["title"].replace("_", " "), "text": paragraph["context"]}
            )
            context_tokens = tokens(paragraph["context"])
            for qa in paragraph["qas"]:
                if qa["is_impossible"]:
                    unanswerable.append(
                        {"id": qa["id"], "question": qa["question"].strip(), "answerable": False}
                    )
                    continue
                reference = qa["answers"][0]["text"].strip()
                if not contains_sequence(context_tokens, tokens(reference)):
                    continue
                answerable.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"].strip(),
                        "reference": reference,
                        "gold_docs": [doc_id],
                        "answerable": True,
                    }
                )

    if len(answerable) < n_answerable or len(unanswerable) < n_unanswerable:
        raise ValueError(
            f"not enough questions in {n_articles} articles: "
            f"{len(answerable)} answerable, {len(unanswerable)} unanswerable"
        )
    golden = rng.sample(answerable, n_answerable) + rng.sample(unanswerable, n_unanswerable)
    golden.sort(key=lambda row: row["id"])
    return corpus, golden
