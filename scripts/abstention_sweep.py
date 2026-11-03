"""How well does bm25-extractive's coverage threshold separate answerable from unanswerable questions?

    python scripts/abstention_sweep.py [--corpus PATH --golden PATH]

For each threshold, prints the share of unanswerable questions the system abstains on (catch rate),
the share of answerable questions it wrongly abstains on, and abstain_accuracy over both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evalkit.data import load_corpus, load_golden
from evalkit.systems import BM25Index

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "datasets" / "squad-v2-sample"
THRESHOLDS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=SAMPLE / "corpus.jsonl")
    parser.add_argument("--golden", type=Path, default=SAMPLE / "golden.jsonl")
    args = parser.parse_args()

    docs, examples = load_corpus(args.corpus), load_golden(args.golden)
    index = BM25Index(docs)
    # bm25-extractive abstains when coverage of its top document falls below the threshold.
    coverage = [(e.answerable, index.coverage(e.question, index.rank(e.question, 1)[0][0])) for e in examples]
    n_ans = sum(a for a, _ in coverage)
    n_unans = len(coverage) - n_ans

    print(f"{len(coverage)} questions: {n_ans} answerable, {n_unans} unanswerable\n")
    print("| Threshold | Unanswerable caught | Answerable wrongly abstained | abstain_accuracy |")
    print("| --- | --- | --- | --- |")
    for t in THRESHOLDS:
        caught = sum(1 for a, c in coverage if not a and c < t)
        wrong = sum(1 for a, c in coverage if a and c < t)
        accuracy = (caught + n_ans - wrong) / len(coverage)
        print(f"| {t:.1f} | {caught / n_unans:.3f} | {wrong / n_ans:.3f} | {accuracy:.3f} |")


if __name__ == "__main__":
    main()
