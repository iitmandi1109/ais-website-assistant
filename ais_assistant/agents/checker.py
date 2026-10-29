"""Checker agent: is every claim in the draft answer supported by the passages?

Two layers:
  1. a deterministic check: every number in the answer (phone numbers, years,
     percentages, counts) must appear in the passages. Small models are worst at
     exactly this kind of detail, and a string check cannot be fooled.
  2. an LLM judge with a separate prompt that sees only the passages and the
     answer (not the conversation), so it judges the text, not the intent.

If either layer fails, or the judge itself errors, the answer is rejected:
saying "I don't know" is cheaper than a wrong fact about the company.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..llm import LLM, LLMError
from ..vectorstore import Hit
from .answerer import PASSAGES_SYSTEM, passages_prompt

log = logging.getLogger("agent.checker")

# Same system prompt and passages as the answer agent, task after them: see PASSAGES_SYSTEM.
CHECK_TASK = """TASK: you are a strict fact checker. Below is an answer written from the passages above.
Split the answer into its factual claims and check each one against the passages.

A claim is supported only if the passages state it or clearly imply it.
Rewording is fine. Extra facts, names, numbers or places that are not in the passages are NOT supported.
Polite phrases such as "you can find more on the website" are not claims.

Return JSON: {"unsupported_claims": ["..."], "supported": true or false}
"supported" is true only if "unsupported_claims" is empty.

Answer to check:
{answer}

Return JSON only."""

# hyphens included so "888-455-6662" is checked as one number, not three short ones
NUMBER_RE = re.compile(r"\d[\d,.\-]*\d|\d")


@dataclass
class Verdict:
    supported: bool
    problems: list[str] = field(default_factory=list)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def unsupported_numbers(answer: str, hits: list[Hit]) -> list[str]:
    """Numbers in the answer that appear nowhere in the passages.

    Compared digits-only, so "888-455-6662" matches "888.455.6662" and "1,400" matches "1400".
    """
    passage_digits = " ".join(_digits(tok) for h in hits for tok in NUMBER_RE.findall(h.text))
    missing = []
    for tok in NUMBER_RE.findall(answer):
        digits = _digits(tok)
        if digits and digits not in passage_digits:
            missing.append(tok)
    return missing


class CheckerAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, answer: str, hits: list[Hit]) -> Verdict:
        if len(re.findall(r"[A-Za-z]{2,}", answer)) < 2:
            # "no claims" must not count as "all claims supported"
            return Verdict(False, ["answer has no real content"])
        bad_numbers = unsupported_numbers(answer, hits)
        if bad_numbers:
            return Verdict(False, [f"number not in sources: {n}" for n in bad_numbers])

        # .replace, not .format: the task text contains literal JSON braces
        prompt = passages_prompt(hits, CHECK_TASK.replace("{answer}", answer))
        try:
            data = self.llm.chat_json(PASSAGES_SYSTEM, prompt)
        except LLMError as exc:
            log.warning("Checker failed (%s); rejecting the answer to be safe", exc)
            return Verdict(False, [f"checker unavailable: {exc}"])

        claims = [str(c) for c in data.get("unsupported_claims") or [] if str(c).strip()]
        supported = data.get("supported") is True and not claims
        return Verdict(supported, claims)
