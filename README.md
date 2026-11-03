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

# the harder SQuAD v2.0 sample (see below)
evalkit leaderboard --corpus datasets/squad-v2-sample/corpus.jsonl --golden datasets/squad-v2-sample/golden.jsonl
python scripts/abstention_sweep.py
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

## A harder test: SQuAD v2.0

The bundled set is too easy to rank systems, so the repo also ships a sample of the
[SQuAD v2.0](https://rajpurkar.github.io/SQuAD-explorer/) dev set: 500 questions over 423
Wikipedia paragraphs from 12 articles. Half the questions are **unanswerable**: crowd workers wrote
them to look answerable from a paragraph that does not contain the answer. Details, licence and the
rebuild script are in [`datasets/squad-v2-sample`](datasets/squad-v2-sample/README.md).

| System | hit@1 | mrr | contains_ref | faithfulness | citation_precision | abstain_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| `bm25-extractive` | 0.812 | 0.861 | 0.732 | 1.000 | 0.849 | 0.522 |
| `bm25-first-sentence` | 0.812 | 0.861 | 0.264 | 1.000 | 0.833 | 0.522 |
| `random-retrieval` (control) | 0.000 | 0.002 | 0.000 | 1.000 | 0.000 | 0.500 |
| `canned-answer` (control) | 0.812 | 0.861 | 0.000 | 0.101 | 0.812 | 0.500 |

95% intervals for `bm25-extractive` (n = 250 answerable): hit@1 0.76-0.86, `contains_ref`
0.68-0.78; `abstain_accuracy` 0.47-0.56 (n = 500).

- **Retrieval is no longer trivial.** hit@1 falls from 1.000 to 0.812 because neighbouring
  paragraphs of one article share vocabulary.
- **The controls still separate.** Random retrieval scores near zero and the canned answer's
  faithfulness drops to 0.101, so the metrics behave on real data, not just on the toy set.
- **The abstainer is close to a coin flip.** `bm25-extractive` abstains when its top document
  covers too little of the question's vocabulary. That reached 1.000 on the bundled set but 0.522
  here.

`python scripts/abstention_sweep.py` asks whether a better threshold would fix it:

| Coverage threshold | Unanswerable caught | Answerable wrongly abstained | abstain_accuracy |
| --- | --- | --- | --- |
| 0.4 (default) | 0.088 | 0.044 | 0.522 |
| 0.6 | 0.288 | 0.204 | 0.542 |
| 0.7 | 0.480 | 0.400 | 0.540 |
| 0.9 | 0.820 | 0.728 | 0.546 |

No threshold does better than 0.548. Catching more unanswerable questions costs almost as many
wrong abstentions, because SQuAD v2's unanswerable questions reuse the paragraph's words on
purpose. Telling them apart needs a model that reads the passage.

## An LLM reader: what it fixes and what it does not

`evalkit.llm` keeps the same BM25 top-3 retrieval but hands the passages to an LLM. The model is
told to answer with the shortest exact span from the passages and cite the passage id, or to reply
`UNANSWERABLE` if the passages do not contain the answer. Because retrieval is identical,
any difference from `bm25-extractive` comes from the reader.

The run below used **Qwen3.8-27B** (4-bit, via LM Studio on a laptop RTX 5090) at temperature 0
with reasoning off, about 3 seconds per question. Any OpenAI-compatible endpoint works (LM Studio,
Ollama, vLLM or a hosted API); see the `EVALKIT_LLM_*` variables in
[`src/evalkit/llm.py`](src/evalkit/llm.py). Every reply is cached in
[`responses/`](responses/), keyed by a hash of the exact request, so CI replays and scores this run
with no GPU or API key, and a test fails if the cache stops covering both golden sets.

```bash
evalkit run --system evalkit.llm:make --corpus datasets/squad-v2-sample/corpus.jsonl --golden datasets/squad-v2-sample/golden.jsonl --out results/llm
EVALKIT_LLM_OFFLINE=1 evalkit run --system evalkit.llm:make ...   # replay from the cache only
python scripts/error_breakdown.py results/bm25/bm25-extractive.json results/llm/evalkit-llm-make.json
```

| System (SQuAD v2.0 sample) | contains_ref | ref_recall | faithfulness | citation_precision | abstain_accuracy |
| --- | --- | --- | --- | --- | --- |
| `bm25-extractive` | 0.732 | 0.760 | 1.000 | 0.849 | 0.522 |
| **LLM reader** | **0.760** | **0.821** | 0.994 | **0.956** | **0.794** |

95% intervals for the LLM reader: `contains_ref` 0.71-0.81 (n = 250), `abstain_accuracy`
0.76-0.83 (n = 500). Retrieval metrics are identical by construction (hit@1 0.812).

Where each system's answers end up (`scripts/error_breakdown.py`):

| System | Correct | Abstained | Gold passage not retrieved | Wrong, gold retrieved | Unanswerable: abstained | Unanswerable: answered |
| --- | --- | --- | --- | --- | --- | --- |
| `bm25-extractive` | 0.732 | 0.044 | 0.060 | 0.164 | 0.088 | 0.912 |
| LLM reader | 0.760 | 0.092 | 0.016 | 0.132 | 0.680 | 0.320 |

The first four columns are shares of the 250 answerable questions; the last two of the 250
unanswerable ones.

- **Abstention is the big win.** The LLM declines 68% of unanswerable questions against 9% for the
  lexical rule, lifting `abstain_accuracy` from 0.522 to 0.794. It still answers 32% of them,
  usually with a plausible span from a passage on the same topic.
- **Its extra abstentions on answerable questions are mostly justified.** It abstained on 23
  answerable questions; in 13 of them BM25 had not retrieved the gold passage, so the passages it
  saw really did not contain the answer. That leaves 10 (4%) where it refused with the answer in
  front of it.
- **Answer quality is better than `contains_ref` shows.** Of the 33 answers scored wrong despite
  the gold passage being retrieved, 25 are a shorter span inside the reference ("Donald Davies"
  against a reference that is a whole clause about him). Counting those, the reader would reach
  about 0.86. The extractive baseline returns whole sentences, which this metric rewards. The same
  bias shows on the bundled set, where the LLM's `contains_ref` (0.861) is below the baseline's
  (0.944) with perfect abstention.
- **Citations improve.** 95.6% of the passages the LLM cites are the gold passage, against 84.9%
  for the passage the baseline takes its sentence from.

Next: a judge that credits correct short or paraphrased answers, checked against hand labels, so
the answer metric stops favouring long spans.

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

- **The bundled data is small and easy.** 12 documents on distinct topics make retrieval trivial (hit@1 is 1.000), so it cannot rank good retrievers against each other. It is there to exercise the harness; use the SQuAD v2.0 sample or your own `--corpus` and `--golden` for a real evaluation.
- **The abstention threshold was chosen after looking at this golden set**, so `abstain_accuracy = 1.000` is optimistic, and with only four unanswerable questions it is a weak estimate.
- **Answer metrics are lexical.** `contains_ref` needs the reference tokens to appear in order, so a correct paraphrase or a correct but shorter span scores 0 (25 of the LLM reader's 33 "wrong" answers on SQuAD). Faithfulness is word overlap with the context, not entailment. An LLM judge would handle both better, at the cost of determinism and API keys.
- **The reference answers were written by the author** from the same documents the systems read.

## Layout

```
src/evalkit/
  text.py      tokenisation and sentence splitting
  data.py      corpus and golden-set loaders with validation
  systems.py   BM25 index, built-in systems, module:factory plug-in loader
  metrics.py   per-example scoring and bootstrap aggregation
  runner.py    run a system, build Markdown reports
  squad.py     SQuAD v2.0 -> corpus and golden set
  llm.py       LLM reader over any OpenAI-compatible endpoint, with a response cache
  gate.py      baseline and regression check
  cli.py       evalkit run | leaderboard | gate | baseline | systems
  data/        bundled corpus and golden set
baselines/     stored scores used by the CI gate
datasets/      SQuAD v2.0 sample (CC BY-SA 4.0)
responses/     cached LLM replies that CI replays
scripts/       sample builder, abstention threshold sweep, error breakdown
tests/         41 tests, including consistency checks for both golden sets and the cache
```

## License

MIT
