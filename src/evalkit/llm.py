"""A RAG system that reads the retrieved passages with an LLM.

Retrieval is the same BM25 top-k as `bm25-extractive`, so comparing the two isolates the reader.
The LLM is any OpenAI-compatible chat endpoint (LM Studio, Ollama, vLLM, or a hosted API). Every
response is cached in a JSONL file keyed by a hash of the exact request, so a committed cache
replays the evaluation without a model, and CI can score it with no GPU or API key.

    evalkit run --system evalkit.llm:make --corpus ... --golden ...

Configuration comes from environment variables:

    EVALKIT_LLM_BASE_URL   default http://localhost:1234/v1 (LM Studio)
    EVALKIT_LLM_MODEL      default qwen/qwen3.8-27b
    EVALKIT_LLM_API_KEY    sent as a bearer token if set
    EVALKIT_LLM_CACHE      default responses/<model>.jsonl
    EVALKIT_LLM_OFFLINE    if "1", never call the model; a cache miss is an error
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .data import Doc
from .systems import ABSTAIN_TEXT, BM25Index, Response

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "qwen/qwen3.8-27b"
UNANSWERABLE = "UNANSWERABLE"

SYSTEM_PROMPT = f"""You answer questions using only the passages provided.
If the passages do not contain the answer, reply with {UNANSWERABLE}. Do not guess, and do not use
knowledge from outside the passages.
Otherwise, give the shortest exact answer copied from the passages (a few words, not a sentence)
and the id of the passage it comes from.
Reply in exactly this format:
ANSWER: <answer or {UNANSWERABLE}>
SOURCE: <passage id, or none>"""

_ANSWER = re.compile(r"^\s*ANSWER:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_SOURCE = re.compile(r"^\s*SOURCE:\s*(.*)$", re.IGNORECASE | re.MULTILINE)

Chat = Callable[[dict], str]


def build_messages(question: str, passages: list[Doc]) -> list[dict]:
    blocks = "\n\n".join(f"[{d.id}] {d.title}\n{d.text}" for d in passages)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Passages:\n\n{blocks}\n\nQuestion: {question}"},
    ]


def parse_reply(text: str) -> tuple[str, str | None]:
    """(answer, source id or None). An unparseable reply is taken as the whole text, unsourced."""
    answer_match = _ANSWER.search(text)
    answer = answer_match.group(1).strip() if answer_match else text.strip()
    source_match = _SOURCE.search(text)
    source = source_match.group(1).strip().strip("[]") if source_match else ""
    if not source or source.lower() == "none":
        source = None
    return answer, source


def is_abstention(answer: str) -> bool:
    return answer.strip().strip(".").upper() == UNANSWERABLE


def openai_chat(base_url: str, api_key: str | None = None, timeout: float = 300.0) -> Chat:
    """A Chat that posts to `{base_url}/chat/completions` and returns the message content."""

    def chat(payload: dict) -> str:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
        return body["choices"][0]["message"]["content"] or ""

    return chat


class ResponseCache:
    """Append-only JSONL of {key, response}; the key is a SHA-256 of the canonical request."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._entries: dict[str, str] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self._entries[row["key"]] = row["response"]

    @staticmethod
    def key(payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def get(self, payload: dict) -> str | None:
        return self._entries.get(self.key(payload))

    def put(self, payload: dict, response: str) -> None:
        key = self.key(payload)
        self._entries[key] = response
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")


class LLMRag:
    """BM25 top-k retrieval, then an LLM that answers from the passages or abstains."""

    name = "llm-rag"

    def __init__(
        self,
        docs: list[Doc],
        chat: Chat | None,
        cache: ResponseCache,
        model: str,
        k: int = 3,
        max_tokens: int = 200,
    ):
        self.docs, self.chat, self.cache, self.model = docs, chat, cache, model
        self.k, self.max_tokens = k, max_tokens
        self.index = BM25Index(docs)

    def payload(self, question: str, passages: list[Doc]) -> dict:
        return {
            "model": self.model,
            "messages": build_messages(question, passages),
            "temperature": 0,
            "seed": 0,
            "max_tokens": self.max_tokens,
            "reasoning_effort": "none",
        }

    def answer(self, question: str) -> Response:
        ranked = [self.docs[i] for i, _ in self.index.rank(question, self.k)]
        payload = self.payload(question, ranked)
        reply = self.cache.get(payload)
        if reply is None:
            if self.chat is None:
                raise RuntimeError(
                    f"no cached response for {question!r} in {self.cache.path} and EVALKIT_LLM_OFFLINE=1"
                )
            reply = self.chat(payload)
            self.cache.put(payload, reply)

        retrieved = [d.id for d in ranked]
        contexts = [d.text for d in ranked]
        answer, source = parse_reply(reply)
        if is_abstention(answer):
            return Response(ABSTAIN_TEXT, retrieved, contexts, [], abstained=True)
        return Response(answer, retrieved, contexts, [source] if source else [])


def make(docs: list[Doc]) -> LLMRag:
    """Factory for `--system evalkit.llm:make`, configured from EVALKIT_LLM_* variables."""
    model = os.environ.get("EVALKIT_LLM_MODEL", DEFAULT_MODEL)
    slug = re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-")
    cache = ResponseCache(os.environ.get("EVALKIT_LLM_CACHE", f"responses/{slug}.jsonl"))
    chat = None
    if os.environ.get("EVALKIT_LLM_OFFLINE") != "1":
        base_url = os.environ.get("EVALKIT_LLM_BASE_URL", DEFAULT_BASE_URL)
        chat = openai_chat(base_url, os.environ.get("EVALKIT_LLM_API_KEY"))
    return LLMRag(docs, chat, cache, model)
