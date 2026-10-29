import json
import textwrap

import pytest

from evalkit import cli
from evalkit.data import Doc, load_corpus
from evalkit.systems import ABSTAIN_TEXT, BM25Extractive, BM25Index, build_system

DOCS = [
    Doc("pets", "Pets", "Cats purr when content. Dogs bark at strangers."),
    Doc("space", "Space", "Mars has two small moons. Jupiter has dozens of moons."),
    Doc("food", "Food", "Bread is baked from flour and water."),
]


def test_bm25_ranks_the_matching_document_first():
    index = BM25Index(DOCS)
    top, _ = index.rank("how many moons does Mars have", 1)[0]
    assert DOCS[top].id == "space"


def test_bm25_ties_break_by_document_order():
    ranked = BM25Index(DOCS).rank("zebra", 3)
    assert [i for i, _ in ranked] == [0, 1, 2]


def test_extractive_system_picks_the_best_sentence_and_cites_it():
    r = BM25Extractive(DOCS).answer("What do dogs do to strangers?")
    assert r.answer == "Dogs bark at strangers." and r.cited == ["pets"] and not r.abstained


def test_extractive_system_abstains_when_nothing_matches():
    r = BM25Extractive(DOCS).answer("Who painted the Mona Lisa?")
    assert r.abstained and r.answer == ABSTAIN_TEXT and r.cited == []


def test_build_system_errors_are_helpful():
    with pytest.raises(KeyError, match="built-ins"):
        build_system("nope", DOCS)


def test_custom_system_plugin(tmp_path, monkeypatch):
    (tmp_path / "my_system.py").write_text(
        textwrap.dedent(
            """
            from evalkit.systems import Response

            class Echo:
                def __init__(self, docs):
                    self.docs = docs

                def answer(self, question):
                    return Response("echo", [self.docs[0].id], [self.docs[0].text], [self.docs[0].id])

            def make(docs):
                return Echo(docs)
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert build_system("my_system:make", DOCS).answer("anything").answer == "echo"


def test_cli_run_writes_reports(tmp_path, capsys):
    assert cli.main(["run", "--system", "bm25-extractive", "--out", str(tmp_path)]) == 0
    assert "contains_ref" in capsys.readouterr().out
    data = json.loads((tmp_path / "bm25-extractive.json").read_text())
    assert data["n_examples"] == 40 and len(data["examples"]) == 40


def test_cli_leaderboard_orders_known_controls(tmp_path, capsys):
    assert cli.main(["leaderboard", "--out", str(tmp_path)]) == 0
    table = (tmp_path / "leaderboard.md").read_text()
    rows = {line.split("|")[1].strip(): line for line in table.splitlines()[2:]}
    assert set(rows) == {"bm25-extractive", "bm25-first-sentence", "random-retrieval", "canned-answer"}
    best = float(rows["bm25-extractive"].split("|")[5])  # contains_ref
    assert (
        best
        > float(rows["bm25-first-sentence"].split("|")[5])
        > float(rows["random-retrieval"].split("|")[5])
    )
    assert float(rows["canned-answer"].split("|")[7]) < 0.2  # faithfulness catches ungrounded answers


def test_gate_passes_on_own_baseline_and_fails_on_regression(tmp_path, capsys):
    baseline = tmp_path / "base.json"
    assert cli.main(["baseline", "--system", "bm25-extractive", "--out", str(baseline)]) == 0
    assert cli.main(["gate", "--system", "bm25-extractive", "--baseline", str(baseline)]) == 0
    # A weaker answerer measured against the stronger system's baseline must fail the gate.
    assert cli.main(["gate", "--system", "bm25-first-sentence", "--baseline", str(baseline)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_gate_tolerance_allows_small_drops(tmp_path):
    baseline = tmp_path / "base.json"
    cli.main(["baseline", "--system", "bm25-extractive", "--out", str(baseline)])
    data = json.loads(baseline.read_text())
    data["metrics"]["contains_ref"] += 0.015  # pretend the baseline was slightly higher
    baseline.write_text(json.dumps(data))
    assert cli.main(["gate", "--baseline", str(baseline), "--tolerance", "0.02"]) == 0
    assert cli.main(["gate", "--baseline", str(baseline), "--tolerance", "0.01"]) == 1


def test_bundled_corpus_loads_for_every_builtin():
    docs = load_corpus()
    for name in ("bm25-extractive", "bm25-first-sentence", "random-retrieval", "canned-answer"):
        assert build_system(name, docs).answer("What is a pod?").answer
