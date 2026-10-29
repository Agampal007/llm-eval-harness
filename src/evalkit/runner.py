"""Run a system over a golden set and produce JSON and Markdown reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .data import Example
from .metrics import METRICS, aggregate, score_example
from .systems import Response, System


@dataclass
class ExampleResult:
    example: Example
    response: Response
    scores: dict[str, float | None]


def run(system: System, examples: list[Example]) -> list[ExampleResult]:
    results = []
    for example in examples:
        response = system.answer(example.question)
        results.append(ExampleResult(example, response, score_example(example, response)))
    return results


def summarise(system_name: str, results: list[ExampleResult]) -> dict[str, Any]:
    return {
        "system": system_name,
        "n_examples": len(results),
        "n_answerable": sum(r.example.answerable for r in results),
        "metrics": aggregate([r.scores for r in results]),
        "examples": [
            {
                "id": r.example.id,
                "question": r.example.question,
                "reference": r.example.reference,
                "response": asdict(r.response),
                "scores": r.scores,
            }
            for r in results
        ],
    }


def _fmt(stat: dict[str, float]) -> str:
    if stat["n"] == 0:
        return "n/a"
    return f"{stat['mean']:.3f} ({stat['ci_low']:.2f}-{stat['ci_high']:.2f})"


def metrics_table(summary: dict[str, Any]) -> str:
    lines = ["| Metric | Mean (95% CI) | n |", "| --- | --- | --- |"]
    for name in METRICS:
        stat = summary["metrics"][name]
        lines.append(f"| {name} | {_fmt(stat)} | {stat['n']} |")
    return "\n".join(lines)


def leaderboard_table(summaries: list[dict[str, Any]]) -> str:
    header = ["System", *METRICS]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for s in summaries:
        cells = [s["system"]]
        for name in METRICS:
            stat = s["metrics"][name]
            cells.append("n/a" if stat["n"] == 0 else f"{stat['mean']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def failures(summary: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    """Answerable questions the system got wrong, for quick debugging."""
    bad = [
        e
        for e in summary["examples"]
        if e["scores"]["contains_ref"] == 0.0 or e["scores"]["abstain_accuracy"] == 0.0
    ]
    return bad[:limit]
