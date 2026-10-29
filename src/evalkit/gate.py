"""Regression gate: compare a run against a stored baseline and fail on drops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GATED_METRICS = ("hit@1", "mrr", "contains_ref", "faithfulness", "citation_precision", "abstain_accuracy")


@dataclass(frozen=True)
class GateRow:
    metric: str
    baseline: float
    current: float
    delta: float
    passed: bool


def make_baseline(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": summary["system"],
        "metrics": {name: round(summary["metrics"][name]["mean"], 6) for name in GATED_METRICS},
    }


def save_baseline(summary: dict[str, Any], path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(make_baseline(summary), indent=2) + "\n", encoding="utf-8")


def check(summary: dict[str, Any], baseline: dict[str, Any], tolerance: float = 0.02) -> list[GateRow]:
    """A metric fails when it falls more than `tolerance` below its baseline. Improvements pass."""
    rows = []
    for name in GATED_METRICS:
        if name not in baseline["metrics"]:
            continue
        base = float(baseline["metrics"][name])
        current = float(summary["metrics"][name]["mean"])
        delta = current - base
        rows.append(GateRow(name, base, current, delta, delta >= -tolerance - 1e-12))
    return rows


def gate_table(rows: list[GateRow], tolerance: float) -> str:
    lines = [
        f"Regression gate (tolerance {tolerance:.3f})",
        "",
        "| Metric | Baseline | Current | Change | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        status = "pass" if r.passed else "FAIL"
        lines.append(f"| {r.metric} | {r.baseline:.3f} | {r.current:.3f} | {r.delta:+.3f} | {status} |")
    return "\n".join(lines)
