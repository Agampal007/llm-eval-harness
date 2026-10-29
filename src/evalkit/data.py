"""Corpus and golden-set loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CORPUS = DATA_DIR / "corpus.jsonl"
DEFAULT_GOLDEN = DATA_DIR / "golden.jsonl"


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    reference: str | None  # short gold answer; None when the question is unanswerable
    gold_docs: tuple[str, ...]
    answerable: bool


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON ({error.msg})") from error
    return rows


def load_corpus(path: Path | str = DEFAULT_CORPUS) -> list[Doc]:
    docs = [Doc(id=r["id"], title=r.get("title", ""), text=r["text"]) for r in _read_jsonl(Path(path))]
    if len({d.id for d in docs}) != len(docs):
        raise ValueError(f"{path}: duplicate document ids")
    return docs


def load_golden(path: Path | str = DEFAULT_GOLDEN) -> list[Example]:
    examples = []
    for r in _read_jsonl(Path(path)):
        answerable = bool(r["answerable"])
        if answerable and not (r.get("reference") and r.get("gold_docs")):
            raise ValueError(f"{r['id']}: answerable examples need a reference and gold_docs")
        examples.append(
            Example(
                id=r["id"],
                question=r["question"],
                reference=r.get("reference"),
                gold_docs=tuple(r.get("gold_docs", [])),
                answerable=answerable,
            )
        )
    if len({e.id for e in examples}) != len(examples):
        raise ValueError(f"{path}: duplicate example ids")
    return examples
