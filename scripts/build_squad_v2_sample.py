"""Download SQuAD v2.0 dev and write the sample in datasets/squad-v2-sample/.

    python scripts/build_squad_v2_sample.py

The download is checked against a pinned SHA-256 and cached in data/ (not committed). The output is
deterministic: the same seed gives byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from evalkit.squad import SQUAD_V2_DEV_SHA256, SQUAD_V2_DEV_URL, build_sample

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "dev-v2.0.json"
OUT = ROOT / "datasets" / "squad-v2-sample"

N_ARTICLES = 12
N_ANSWERABLE = 250
N_UNANSWERABLE = 250
SEED = 2026


def download() -> bytes:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SQUAD_V2_DEV_URL, timeout=60) as response:
            RAW.write_bytes(response.read())
    raw = RAW.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SQUAD_V2_DEV_SHA256:
        raise SystemExit(f"checksum mismatch for {RAW}: {digest}")
    return raw


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    squad = json.loads(download())
    corpus, golden = build_sample(squad, N_ARTICLES, N_ANSWERABLE, N_UNANSWERABLE, SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "corpus.jsonl", corpus)
    write_jsonl(OUT / "golden.jsonl", golden)
    titles = sorted({row["title"] for row in corpus})
    print(f"{len(corpus)} paragraphs from {len(titles)} articles: {', '.join(titles)}")
    print(f"{len(golden)} questions ({N_ANSWERABLE} answerable, {N_UNANSWERABLE} unanswerable) -> {OUT}")


if __name__ == "__main__":
    main()
