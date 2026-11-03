from pathlib import Path

import pytest

from evalkit.data import DEFAULT_CORPUS, DEFAULT_GOLDEN, Doc, load_corpus, load_golden
from evalkit.llm import LLMRag, ResponseCache, build_messages, is_abstention, make, parse_reply
from evalkit.systems import ABSTAIN_TEXT

DOCS = [
    Doc("rhine-p0", "Rhine", "The Rhine is a European river that begins in the Swiss canton of Graubunden."),
    Doc("oxygen-p0", "Oxygen", "Oxygen is a chemical element with symbol O and atomic number 8."),
    Doc("warsaw-p0", "Warsaw", "Warsaw is the capital and largest city of Poland."),
]


class FakeChat:
    """Returns canned replies and records every request it receives."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []

    def __call__(self, payload: dict) -> str:
        self.calls.append(payload)
        return self.reply


def _system(tmp_path, reply: str):
    chat = FakeChat(reply)
    return LLMRag(DOCS, chat, ResponseCache(tmp_path / "cache.jsonl"), model="test-model", k=2), chat


def test_parse_reply_reads_answer_and_source():
    assert parse_reply("ANSWER: atomic number 8\nSOURCE: [oxygen-p0]") == ("atomic number 8", "oxygen-p0")
    assert parse_reply("answer: Poland\nsource: none") == ("Poland", None)
    assert parse_reply("Just some text") == ("Just some text", None)


def test_abstention_is_recognised_with_or_without_a_full_stop():
    assert is_abstention("UNANSWERABLE")
    assert is_abstention(" unanswerable. ")
    assert not is_abstention("Unanswerable questions are tricky")


def test_prompt_contains_passage_ids_and_the_question():
    user = build_messages("What is the atomic number of oxygen?", DOCS[1:2])[1]["content"]
    assert "[oxygen-p0] Oxygen" in user and user.endswith("Question: What is the atomic number of oxygen?")


def test_answer_cites_the_source_and_uses_retrieved_passages(tmp_path):
    system, chat = _system(tmp_path, "ANSWER: 8\nSOURCE: oxygen-p0")
    response = system.answer("What is the atomic number of oxygen?")
    assert response.answer == "8" and response.cited == ["oxygen-p0"]
    assert response.retrieved[0] == "oxygen-p0" and len(response.retrieved) == 2
    assert not response.abstained
    assert chat.calls[0]["temperature"] == 0


def test_unanswerable_reply_becomes_an_abstention(tmp_path):
    system, _ = _system(tmp_path, "ANSWER: UNANSWERABLE\nSOURCE: none")
    response = system.answer("What is the boiling point of oxygen?")
    assert response.abstained and response.answer == ABSTAIN_TEXT and response.cited == []


def test_cache_replays_without_calling_the_model(tmp_path):
    system, chat = _system(tmp_path, "ANSWER: Poland\nSOURCE: warsaw-p0")
    system.answer("Which country is Warsaw the capital of?")
    assert len(chat.calls) == 1

    offline = LLMRag(DOCS, None, ResponseCache(tmp_path / "cache.jsonl"), model="test-model", k=2)
    assert offline.answer("Which country is Warsaw the capital of?").answer == "Poland"
    with pytest.raises(RuntimeError, match="no cached response"):
        offline.answer("A question that was never asked")


def test_cache_key_changes_with_the_model(tmp_path):
    payload = {"model": "a", "messages": []}
    assert ResponseCache.key(payload) != ResponseCache.key({**payload, "model": "b"})


def test_make_reads_its_configuration_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALKIT_LLM_MODEL", "Org/Some-Model 7B")
    monkeypatch.setenv("EVALKIT_LLM_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    system = make(DOCS)
    assert system.chat is None
    assert system.cache.path.as_posix() == "responses/org-some-model-7b.jsonl"


ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "corpus, golden",
    [
        (DEFAULT_CORPUS, DEFAULT_GOLDEN),
        (ROOT / "datasets/squad-v2-sample/corpus.jsonl", ROOT / "datasets/squad-v2-sample/golden.jsonl"),
    ],
)
def test_committed_cache_replays_every_question(corpus, golden, monkeypatch):
    # CI scores the LLM system from this cache, so it must cover both golden sets exactly.
    monkeypatch.setenv("EVALKIT_LLM_OFFLINE", "1")
    monkeypatch.setenv("EVALKIT_LLM_CACHE", str(ROOT / "responses/qwen-qwen3.8-27b.jsonl"))
    system = make(load_corpus(corpus))
    responses = [system.answer(e.question) for e in load_golden(golden)]
    assert any(r.abstained for r in responses) and any(not r.abstained for r in responses)
