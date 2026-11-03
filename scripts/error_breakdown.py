"""Where does a system go wrong? Splits each question into one outcome.

    evalkit run --system bm25-extractive --corpus ... --golden ... --out results/bm25
    python scripts/error_breakdown.py results/bm25/bm25-extractive.json [more summaries ...]

Answerable questions: correct (the answer contains the reference), abstained, gold passage not
retrieved, or wrong answer despite retrieving the gold passage. Unanswerable questions: abstained
(correct) or answered.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ANSWERABLE = ("correct", "abstained", "gold not retrieved", "wrong answer, gold retrieved")
UNANSWERABLE = ("abstained", "answered")


def outcome(example: dict) -> tuple[bool, str]:
    response, scores = example["response"], example["scores"]
    answerable = example["reference"] is not None
    if not answerable:
        return False, "abstained" if response["abstained"] else "answered"
    if response["abstained"]:
        return True, "abstained"
    if scores["contains_ref"] == 1.0:
        return True, "correct"
    if scores["hit@3"] == 0.0:
        return True, "gold not retrieved"
    return True, "wrong answer, gold retrieved"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("summaries", nargs="+", type=Path, help="JSON written by evalkit run --out")
    args = parser.parse_args()

    header = [
        "System",
        *(f"answerable: {o}" for o in ANSWERABLE),
        *(f"unanswerable: {o}" for o in UNANSWERABLE),
    ]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")
    for path in args.summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        counts = Counter(outcome(e) for e in summary["examples"])
        n_ans = sum(v for (a, _), v in counts.items() if a)
        n_unans = sum(v for (a, _), v in counts.items() if not a)
        cells = [summary["system"]]
        cells += [f"{counts[(True, o)] / n_ans:.3f}" for o in ANSWERABLE]
        cells += [f"{counts[(False, o)] / n_unans:.3f}" for o in UNANSWERABLE]
        print("| " + " | ".join(cells) + " |")
    print(f"\nShares of {n_ans} answerable and {n_unans} unanswerable questions (last summary).")


if __name__ == "__main__":
    main()
