# SQuAD v2.0 sample

500 questions over 423 Wikipedia paragraphs, taken from the SQuAD v2.0 dev set:
250 answerable and 250 unanswerable.

- **Articles (12, chosen with seed 2026):** Civil disobedience, Construction, Force, Harvard
  University, Islamism, Oxygen, Packet switching, Prime number, Private school, Rhine, Steam
  engine, Warsaw.
- **Corpus:** every paragraph of those articles, one document each (`<article>-p<index>`).
- **Answerable questions:** the reference is the first gold answer. One of the 2,164 answerable
  questions in these articles was dropped because its answer span cut a word in half, which
  `contains_ref` could never match.
- **Unanswerable questions:** written by crowd workers to look answerable from a paragraph that
  does not contain the answer. They have no `reference` and no `gold_docs`.

Rebuild with `python scripts/build_squad_v2_sample.py`. It downloads the dev set, checks its
SHA-256 and writes byte-identical files.

## Source and licence

Rajpurkar, Jia and Liang, "Know What You Don't Know: Unanswerable Questions for SQuAD", ACL 2018.
<https://rajpurkar.github.io/SQuAD-explorer/>

SQuAD v2.0 is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
These files are a reformatted subset of it and are distributed under the same licence. The MIT
licence of this repository does not apply to them.
