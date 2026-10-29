"""Answer agent: write a short answer using only the retrieved passages.

The model never writes URLs. It cites passage numbers ([1], [2]) and the code
maps those numbers to the page links, so a link can never be invented.

Plain text with inline [n] citations, not JSON: my first version asked for
{"answer": "...", "sources": [...]} and qwen2.5:3b often copied the "..."
placeholder literally or returned one bare word. Prose is what small models
are best at, and the citations are easy to parse back out.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .. import config
from ..llm import LLM
from ..vectorstore import Hit

log = logging.getLogger("agent.answerer")

NOT_FOUND = "NOT_FOUND"

# The answer and checker prompts both START with this system prompt + the passages, and put their own
# task AFTER them. Ollama keeps the processed prompt of the previous call, so the checker reuses the
# ~1,200 passage tokens the answer agent just read instead of reading them again
# (measured: checker prompt 3.7 s -> 0.2 s). Any change here must keep the two prefixes identical.
PASSAGES_SYSTEM = ("You are part of the AIS website assistant. You work only from the numbered passages "
                   "from the AIS website below. Do not use outside knowledge.")

ANSWER_TASK = f"""TASK: answer a website visitor's question using only the passages above.

Rules:
- Write 1 to 3 short, complete sentences in plain English, as if talking to a website visitor.
- If the question asks about several things (for example offices, services or industries), name every one the passages mention.
- After each sentence, cite the passage number(s) it comes from in square brackets, like [2].
- Every fact must come from the passages. Never guess names, numbers, prices, clients or awards.
- If the passages do not answer the question, reply with exactly: {NOT_FOUND}

Example of the format only (about a different company): if asked "Where are Acme's offices?", a good reply is
Acme's head office is in Leeds, England [2]. It also has offices in Dublin and Lyon [4].

The visitor asks: {{question}}
Reply with the answer only."""

# Newline-prefixed on purpose: a bare "Question:" also matched when the model began its reply by echoing
# the question, which produced an empty answer (seen in evaluation).
STOP_SEQUENCES = ["\nQuestion:", "\nThe visitor asks", "\nPassages:"]
CITATION_RE = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
# Small models sometimes explain the passages instead of saying NOT_FOUND
META_RE = re.compile(r"(passages?|provided (text|information|context)) (do|does|did) not|"
                     r"not (mentioned|specified|stated|provided) in the (passages?|text)", re.IGNORECASE)


@dataclass
class Draft:
    text: str
    sources: list[Hit] = field(default_factory=list)  # the passages the answer cites

    @property
    def found(self) -> bool:
        return self.text != NOT_FOUND


def format_passages(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] (page: {h.title})\n{h.text}" for i, h in enumerate(hits, start=1))


def passages_prompt(hits: list[Hit], task: str) -> str:
    """Passages first, task last: the shared prefix that makes the checker's call cheap."""
    return f"Passages:\n{format_passages(hits)}\n\n{task}"


def parse_answer(raw: str, hits: list[Hit]) -> Draft:
    """Split the model's text into the clean answer and the passages it cited."""
    # Small models sometimes echo the question first ("Question: ...\nAnswer: ..."): drop that,
    # then cut off any further invented "Question:" pair (safety net if a stop sequence was missed).
    raw = re.sub(r"^\s*Question:.*?\n", "", raw.strip())
    raw = re.sub(r"^\s*(Answer|Reply):", "", raw).strip()
    raw = re.split(r"\n\s*(?:Question|Passages):", raw)[0].strip()
    if NOT_FOUND in raw.upper().replace(" ", "_") or META_RE.search(raw):
        return Draft(NOT_FOUND)

    # Keep the first few sentences. Done before reading citations, so a cut sentence's source is dropped too.
    # Splits only before a capital letter, so "888.455.6662" and "2.5 million" stay whole.
    sentences = SENTENCE_RE.split(raw)
    if len(sentences) > config.MAX_ANSWER_SENTENCES:
        log.info("answer had %d sentences; keeping %d", len(sentences), config.MAX_ANSWER_SENTENCES)
        raw = " ".join(sentences[: config.MAX_ANSWER_SENTENCES])

    cited: list[Hit] = []
    for group in CITATION_RE.findall(raw):
        for n in group.split(","):
            idx = int(n) - 1
            if 0 <= idx < len(hits) and hits[idx] not in cited:
                cited.append(hits[idx])
    text = re.sub(r"\s+([.,;:!?])", r"\1", CITATION_RE.sub("", raw))
    text = re.sub(r"[,;:]+([.!?])", r"\1", text).strip()  # "interested in [1],." -> "interested in."

    if len(re.findall(r"[A-Za-z0-9]+", text)) < 2:  # "...", "" or a lone word is not an answer
        log.info("answer agent returned no usable text: %r", raw)
        return Draft(NOT_FOUND)
    if not cited:
        # The model forgot to cite. Use the best match rather than show no link;
        # the checker still has to confirm the answer against the passages.
        log.info("answer had no valid citations; using the top passage")
        cited = [hits[0]]
    return Draft(text, cited)


class AnswerAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, question: str, hits: list[Hit]) -> Draft:
        prompt = passages_prompt(hits, ANSWER_TASK.format(question=question))
        # The prompt ends "Question: ...\nAnswer:", so a small model may carry on and invent the next
        # "Question:/Answer:" pair itself (seen in testing). Stopping there also saves generation time.
        return parse_answer(self.llm.chat(PASSAGES_SYSTEM, prompt, stop=STOP_SEQUENCES), hits)
