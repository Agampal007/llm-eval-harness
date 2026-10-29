# llm-eval-harness

[![CI](https://github.com/Agampal007/llm-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/Agampal007/llm-eval-harness/actions/workflows/ci.yml)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Agampal007/llm-eval-harness)

A small evaluation harness for RAG systems. It scores a system on **retrieval**, **answer quality**, **grounding** and **knowing when to abstain**, reports bootstrap confidence intervals, and includes a **regression gate** that fails a CI build when quality drops below a stored baseline.

It runs with only NumPy, needs no API keys, and everything is deterministic.

## Run it from GitHub

- **In the browser:** click **Open in GitHub Codespaces** above. The container installs the package and prints the leaderboard on startup, so you can run `evalkit run` straight away.
- **From the Actions tab:** open **Run an evaluation**, press **Run workflow**, pick a system, and read its scores in the job summary.
- **On every push:** the **CI** workflow runs the tests on Python 3.10-3.13, prints a leaderboard of all built-in systems to the job summary, and runs the regression gate.

## Run it locally

```bash
pip install -e ".[dev]"
evalkit leaderboard          # all built-in systems side by side
evalkit run --system bm25-extractive
evalkit gate                 # exit code 1 if scores fell below baselines/bm25-extractive.json
pytest
```

## What it measures

Scores are in [0, 1] and higher is better. Answerable questions are scored on the first six metrics; abstention is scored on all questions.

| Metric | Meaning |
| --- | --- |
| `hit@1`, `hit@3`, `mrr` | A gold document is retrieved at rank 1 / within 3 / reciprocal rank in the top 5 |
| `contains_ref` | The answer contains the reference answer as a run of tokens |
| `ref_recall` | Share of the reference answer's tokens that appear in the answer |
| `faithfulness` | Share of the answer's content words found in the retrieved context. This measures grounding, not correctness |
| `citation_precision` | Share of cited documents that are gold documents |
| `abstain_accuracy` | The system abstained exactly when the question has no answer in the corpus |

Each metric is reported as a mean with a 95% percentile-bootstrap interval (2,000 resamples, fixed seed).

## Built-in systems and what they show

The bundled golden set has 40 questions over 12 short technical documents: 36 answerable and 4 that have no answer in the corpus. Two of the four systems are deliberate controls; a harness that cannot tell them apart from a real system is broken.

| System | hit@1 | mrr | contains_ref | faithfulness | citation_precision | abstain_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| `bm25-extractive` | 1.000 | 1.000 | 0.944 | 1.000 | 1.000 | 1.000 |
| `bm25-first-sentence` | 1.000 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 |
| `random-retrieval` (control) | 0.056 | 0.120 | 0.056 | 1.000 | 0.056 | 0.900 |
| `canned-answer` (control) | 1.000 | 1.000 | 0.000 | 0.045 | 1.000 | 0.900 |

What each row tells you:

- `bm25-first-sentence` has the same retrieval as `bm25-extractive` but a weaker answerer, so only the answer metrics move. The metrics separate retrieval faults from answer faults.
- `random-retrieval` copies real sentences from real documents, so its faithfulness is 1.000 even though its answers are wrong. Faithfulness alone would not catch it; `contains_ref` and `hit@k` do.
- `canned-answer` retrieves well but returns a fluent claim that nothing in the context supports. Faithfulness (0.045) catches it.
- `bm25-extractive` misses two questions (`q17`, `q23`) because its sentence selector picks a related sentence from the right document. These are real failures and the baseline keeps them.

Per-metric intervals for `bm25-extractive` (`evalkit run`): `contains_ref` is 0.944 (95% CI 0.86-1.00, n = 36).

## Regression gate

`evalkit baseline` stores the current scores in `baselines/<system>.json`. `evalkit gate` re-runs the system and fails with exit code 1 if any gated metric falls more than `--tolerance` (default 0.02) below its baseline. Improvements pass. The test suite checks that a weaker system measured against the stronger system's baseline fails the gate.

## Evaluate your own system

Write a factory that takes the corpus and returns an object with `answer(question) -> Response`:

```python
# my_rag.py
from evalkit.systems import Response


class MyRAG:
    def __init__(self, docs):
        self.docs = docs

    def answer(self, question):
        ...  # call your retriever and generator here
        return Response(
            answer="...",
            retrieved=["doc-id", ...],
            contexts=["text shown to the model", ...],
            cited=["doc-id"],
            abstained=False,
        )


def make(docs):
    return MyRAG(docs)
```

```bash
evalkit run --system my_rag:make
evalkit run --system my_rag:make --corpus my_corpus.jsonl --golden my_golden.jsonl
```

Corpus rows are `{"id", "title", "text"}`; golden rows are `{"id", "question", "reference", "gold_docs", "answerable"}`.

## Limitations

- **The bundled data is small and easy.** 12 documents on distinct topics make retrieval trivial (hit@1 is 1.000), so it cannot rank good retrievers against each other. It is there to exercise the harness; use `--corpus` and `--golden` for a real evaluation.
- **The abstention threshold was chosen after looking at this golden set**, so `abstain_accuracy = 1.000` is optimistic, and with only four unanswerable questions it is a weak estimate.
- **Answer metrics are lexical.** `contains_ref` needs the reference tokens to appear in order, so a correct paraphrase scores 0. Faithfulness is word overlap with the context, not entailment. An LLM judge would handle both better, at the cost of determinism and API keys.
- **The reference answers were written by the author** from the same documents the systems read.

## Layout

```
src/evalkit/
  text.py      tokenisation and sentence splitting
  data.py      corpus and golden-set loaders with validation
  systems.py   BM25 index, built-in systems, module:factory plug-in loader
  metrics.py   per-example scoring and bootstrap aggregation
  runner.py    run a system, build Markdown reports
  gate.py      baseline and regression check
  cli.py       evalkit run | leaderboard | gate | baseline | systems
  data/        bundled corpus and golden set
baselines/     stored scores used by the CI gate
tests/         27 tests, including a golden-set consistency check
```

## License

MIT
