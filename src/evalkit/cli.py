"""Command line interface: evalkit run | leaderboard | gate | baseline | systems."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import gate as gate_mod
from .data import DEFAULT_CORPUS, DEFAULT_GOLDEN, load_corpus, load_golden
from .runner import leaderboard_table, metrics_table, run, summarise
from .systems import SYSTEMS, build_system


def _evaluate(spec: str, corpus_path: Path, golden_path: Path) -> dict:
    docs, examples = load_corpus(corpus_path), load_golden(golden_path)
    return summarise(spec, run(build_system(spec, docs), examples))


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="corpus JSONL (default: bundled)")
    parser.add_argument(
        "--golden", type=Path, default=DEFAULT_GOLDEN, help="golden set JSONL (default: bundled)"
    )


def cmd_run(args: argparse.Namespace) -> int:
    summary = _evaluate(args.system, args.corpus, args.golden)
    print(
        f"## {summary['system']}  ({summary['n_examples']} questions, {summary['n_answerable']} answerable)\n"
    )
    print(metrics_table(summary))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        stem = args.system.replace(":", "-").replace(".", "-")
        (args.out / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (args.out / f"{stem}.md").write_text(metrics_table(summary) + "\n", encoding="utf-8")
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    names = args.systems or list(SYSTEMS)
    summaries = [_evaluate(name, args.corpus, args.golden) for name in names]
    table = leaderboard_table(summaries)
    print(table)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "leaderboard.md").write_text(table + "\n", encoding="utf-8")
        for s in summaries:
            (args.out / f"{s['system']}.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    summary = _evaluate(args.system, args.corpus, args.golden)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    rows = gate_mod.check(summary, baseline, args.tolerance)
    print(gate_mod.gate_table(rows, args.tolerance))
    failed = [r.metric for r in rows if not r.passed]
    if failed:
        print(f"\nGate failed: {', '.join(failed)} regressed beyond tolerance.", file=sys.stderr)
        return 1
    print("\nGate passed.")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    summary = _evaluate(args.system, args.corpus, args.golden)
    gate_mod.save_baseline(summary, args.out)
    print(f"Wrote baseline for {args.system} to {args.out}")
    return 0


def cmd_systems(_: argparse.Namespace) -> int:
    for name in SYSTEMS:
        print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalkit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="evaluate one system")
    p.add_argument("--system", default="bm25-extractive", help="built-in name or module:factory")
    p.add_argument("--out", type=Path, help="directory for JSON and Markdown reports")
    _add_data_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("leaderboard", help="evaluate several systems side by side")
    p.add_argument("--systems", nargs="*", help="default: all built-ins")
    p.add_argument("--out", type=Path)
    _add_data_args(p)
    p.set_defaults(func=cmd_leaderboard)

    p = sub.add_parser("gate", help="fail (exit 1) if a system regressed against a baseline")
    p.add_argument("--system", default="bm25-extractive")
    p.add_argument("--baseline", type=Path, default=Path("baselines/bm25-extractive.json"))
    p.add_argument("--tolerance", type=float, default=0.02)
    _add_data_args(p)
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("baseline", help="write the current scores as the new baseline")
    p.add_argument("--system", default="bm25-extractive")
    p.add_argument("--out", type=Path, default=Path("baselines/bm25-extractive.json"))
    _add_data_args(p)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("systems", help="list built-in systems")
    p.set_defaults(func=cmd_systems)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
