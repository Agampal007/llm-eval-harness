import json

import pytest

from evalkit.data import DEFAULT_CORPUS, DEFAULT_GOLDEN, load_corpus, load_golden
from evalkit.text import contains_sequence, content_tokens, sentences, tokens


def test_tokens_and_content_tokens():
    assert tokens("The B-tree, version 3.13!") == ["the", "b", "tree", "version", "3", "13"]
    assert content_tokens("What is the B-tree index?") == ["tree", "index"]


def test_sentences_do_not_split_on_decimals():
    text = "Python 3.13 added a build. It is experimental!"
    assert sentences(text) == ["Python 3.13 added a build.", "It is experimental!"]


def test_contains_sequence_needs_contiguous_run():
    hay = ["a", "new", "read", "only", "layer"]
    assert contains_sequence(hay, ["read", "only"])
    assert not contains_sequence(hay, ["new", "only"])
    assert not contains_sequence(hay, [])


def test_bundled_golden_set_is_internally_consistent():
    docs = {d.id: d for d in load_corpus()}
    examples = load_golden()
    assert len(examples) == 40 and sum(not e.answerable for e in examples) == 4
    for e in examples:
        if not e.answerable:
            assert e.gold_docs == ()
            continue
        assert set(e.gold_docs) <= set(docs), e.id
        gold_tokens = tokens(" ".join(docs[g].text for g in e.gold_docs))
        assert contains_sequence(gold_tokens, tokens(e.reference)), f"{e.id}: reference not in gold doc"


def test_loader_reports_bad_json_with_line_number(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a", "text": "ok"}\nnot json\n')
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        load_corpus(bad)


def test_loader_rejects_duplicate_ids_and_incomplete_answerable(tmp_path):
    dup = tmp_path / "dup.jsonl"
    dup.write_text(json.dumps({"id": "a", "text": "x"}) + "\n" + json.dumps({"id": "a", "text": "y"}) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(dup)
    golden = tmp_path / "g.jsonl"
    golden.write_text(json.dumps({"id": "q", "question": "?", "answerable": True}) + "\n")
    with pytest.raises(ValueError, match="need a reference"):
        load_golden(golden)


def test_default_paths_exist():
    assert DEFAULT_CORPUS.exists() and DEFAULT_GOLDEN.exists()
