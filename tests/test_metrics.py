import math

import pytest

from evalkit.data import Example
from evalkit.metrics import aggregate, bootstrap_ci, score_example
from evalkit.systems import ABSTAIN_TEXT, Response

ANSWERABLE = Example("q1", "What is X?", "a new layer", ("d1",), True)
UNANSWERABLE = Example("u1", "Who knows?", None, (), False)


def test_perfect_response():
    r = Response("Each step makes a new layer.", ["d1", "d2"], ["Each step makes a new layer today."], ["d1"])
    s = score_example(ANSWERABLE, r)
    assert s["hit@1"] == s["hit@3"] == s["mrr"] == 1.0
    assert s["contains_ref"] == 1.0 and s["ref_recall"] == 1.0
    assert s["faithfulness"] == 1.0 and s["citation_precision"] == 1.0
    assert s["abstain_accuracy"] == 1.0


def test_rank_two_gives_half_mrr_and_misses_hit_at_one():
    s = score_example(ANSWERABLE, Response("x", ["d2", "d1", "d3"], [], ["d2"]))
    assert (s["hit@1"], s["hit@3"], s["mrr"]) == (0.0, 1.0, 0.5)
    assert s["citation_precision"] == 0.0


def test_gold_outside_top_five_scores_zero_mrr():
    s = score_example(ANSWERABLE, Response("x", ["a", "b", "c", "d", "e", "d1"], [], []))
    assert s["mrr"] == 0.0 and s["hit@3"] == 0.0


def test_partial_reference_recall_and_no_citations():
    s = score_example(ANSWERABLE, Response("There is a layer.", ["d1"], [], []))
    assert s["contains_ref"] == 0.0
    assert s["ref_recall"] == pytest.approx(2 / 3)  # "a" and "layer" present, "new" missing
    assert s["citation_precision"] == 0.0


def test_faithfulness_drops_for_unsupported_claims():
    r = Response("Widely adopted best practice everywhere.", ["d1"], ["unrelated text"], ["d1"])
    assert score_example(ANSWERABLE, r)["faithfulness"] == 0.0


def test_abstaining_on_answerable_question_is_penalised():
    s = score_example(ANSWERABLE, Response(ABSTAIN_TEXT, ["d1"], [], [], abstained=True))
    assert s["abstain_accuracy"] == 0.0 and s["contains_ref"] == 0.0
    assert s["faithfulness"] is None and s["citation_precision"] is None
    assert s["hit@1"] == 1.0  # retrieval is scored independently of the abstention decision


def test_unanswerable_questions_only_score_abstention_and_faithfulness():
    ok = score_example(UNANSWERABLE, Response(ABSTAIN_TEXT, ["d1"], [], [], abstained=True))
    assert ok["abstain_accuracy"] == 1.0 and ok["hit@1"] is None and ok["contains_ref"] is None
    wrong = score_example(UNANSWERABLE, Response("Something confident.", ["d1"], ["other"], ["d1"]))
    assert wrong["abstain_accuracy"] == 0.0 and wrong["faithfulness"] == 0.0


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    low, high = bootstrap_ci(values)
    assert (low, high) == bootstrap_ci(values)
    assert low <= sum(values) / len(values) <= high
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)
    assert all(math.isnan(x) for x in bootstrap_ci([]))


def test_aggregate_skips_missing_values():
    from evalkit.metrics import METRICS

    rows = [{**dict.fromkeys(METRICS), "hit@1": v} for v in (1.0, None, 0.0)]
    summary = aggregate(rows)
    assert summary["hit@1"]["mean"] == 0.5 and summary["hit@1"]["n"] == 2
    assert summary["mrr"]["n"] == 0 and math.isnan(summary["mrr"]["mean"])
