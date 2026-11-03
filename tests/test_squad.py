from pathlib import Path

import pytest

from evalkit.data import load_corpus, load_golden
from evalkit.squad import build_sample
from evalkit.text import contains_sequence, tokens

SAMPLE = Path(__file__).resolve().parent.parent / "datasets" / "squad-v2-sample"


def _qa(qid, question, answer=None):
    if answer is None:
        return {"id": qid, "question": question, "answers": [], "is_impossible": True}
    return {"id": qid, "question": question, "answers": [{"text": answer}], "is_impossible": False}


FAKE_SQUAD = {
    "data": [
        {
            "title": "Normandy_Region",
            "paragraphs": [
                {
                    "context": "The Normans gave their name to Normandy in the 10th century.",
                    "qas": [
                        _qa("a1", "When did they name Normandy?", "10th century"),
                        _qa("a2", "Who named it?", "Norman"),  # span cut mid-word: dropped
                        _qa("u1", "When did the Normans leave Normandy?"),
                    ],
                },
                {
                    "context": "Rollo was the first ruler of Normandy.",
                    "qas": [
                        _qa("a3", "Who was the first ruler?", "Rollo"),
                        _qa("u2", "Who was the last ruler?"),
                    ],
                },
            ],
        }
    ]
}


def test_build_sample_shapes_rows_for_the_loaders():
    corpus, golden = build_sample(FAKE_SQUAD, n_articles=1, n_answerable=2, n_unanswerable=2, seed=0)

    assert [d["id"] for d in corpus] == ["normandy-region-p0", "normandy-region-p1"]
    assert corpus[0]["title"] == "Normandy Region"
    by_id = {row["id"]: row for row in golden}
    assert set(by_id) == {"a1", "a3", "u1", "u2"}
    assert by_id["a1"] == {
        "id": "a1",
        "question": "When did they name Normandy?",
        "reference": "10th century",
        "gold_docs": ["normandy-region-p0"],
        "answerable": True,
    }
    assert by_id["u2"] == {"id": "u2", "question": "Who was the last ruler?", "answerable": False}


def test_build_sample_drops_answers_that_split_a_word():
    _, golden = build_sample(FAKE_SQUAD, n_articles=1, n_answerable=2, n_unanswerable=0, seed=0)
    assert "a2" not in {row["id"] for row in golden}


def test_build_sample_refuses_to_return_fewer_questions_than_asked():
    with pytest.raises(ValueError, match="not enough questions"):
        build_sample(FAKE_SQUAD, n_articles=1, n_answerable=3, n_unanswerable=0, seed=0)


def test_committed_squad_sample_is_balanced_and_consistent():
    docs = {d.id: d for d in load_corpus(SAMPLE / "corpus.jsonl")}
    examples = load_golden(SAMPLE / "golden.jsonl")
    assert len(examples) == 500 and sum(not e.answerable for e in examples) == 250
    for e in examples:
        if not e.answerable:
            assert e.gold_docs == () and e.reference is None
            continue
        (gold,) = e.gold_docs
        assert contains_sequence(tokens(docs[gold].text), tokens(e.reference)), e.id
