"""Orchestrator: passes work between the agents, keeps the conversation, handles failures.

The flow is a small LangGraph state graph:

    route --+-- small_talk ---------------------------------------> END
            +-- off_topic ----------------------------------------> END
            +-- retrieve --(no hits)----------------> unknown ----> END
                    |                                    ^
                    +--> answer --(NOT_FOUND)------------+
                            |                            |
                            +--> check --(unsupported)---+
                                    |
                                    +--(supported)--> respond --> END

Every node writes its decision to `trace`, which the CLI prints with --trace
and the evaluation saves, so each answer can be explained after the fact.
Any exception inside a node becomes a polite reply instead of a crash.
"""

from __future__ import annotations

import logging
import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from . import config
from .agents import AnswerAgent, CheckerAgent, Draft, Intent, RetrievalAgent, Route, RouterAgent, Verdict
from .llm import LLM, LLMError
from .vectorstore import Hit, VectorStore

log = logging.getLogger("orchestrator")

UNKNOWN_TEXT = ("I'm sorry, I couldn't find that on the AIS website, so I'd rather not guess. "
                f"The AIS team can help you directly: {config.CONTACT_URL}")
SMALL_TALK_TEXT = ("Hi! I'm the AIS website assistant. I can answer questions about AIS: "
                   "its services, the industries it serves, its offices, leadership and careers.")
OFF_TOPIC_TEXT = ("Sorry, I can only help with questions about AIS and its website. "
                  "For example, you could ask what services AIS offers or where its offices are.")
ERROR_TEXT = ("Sorry, I'm having a technical problem and can't answer right now. Please try again "
              f"in a moment, or contact AIS directly: {config.CONTACT_URL}")


@dataclass
class Reply:
    status: str                                   # answered | unknown | small_talk | off_topic | error
    text: str
    sources: list[tuple[str, str]] = field(default_factory=list)  # (title, url)
    query: str = ""                               # the standalone question that was searched
    trace: list[dict] = field(default_factory=list)

    def as_markdown(self) -> str:
        if not self.sources:
            return self.text
        links = ", ".join(f"[{title}]({url})" for title, url in self.sources)
        return f"{self.text}\n\nSource: {links}"


class ChatState(TypedDict, total=False):
    message: str
    history: list[tuple[str, str]]
    route: Route
    hits: list[Hit]
    draft: Draft
    verdict: Verdict
    reply: Reply
    trace: Annotated[list[dict], operator.add]   # every node appends, nothing overwrites


def _unique_sources(hits: list[Hit]) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    for h in hits:
        seen.setdefault(h.url, h.title)
    return [(title, url) for url, title in seen.items()]


class Orchestrator:
    def __init__(self, llm: LLM | None = None, store: VectorStore | None = None):
        llm = self.llm = llm or LLM()
        self.router = RouterAgent(llm)
        self.retriever = RetrievalAgent(store or VectorStore())
        self.answerer = AnswerAgent(llm)
        self.checker = CheckerAgent(llm)
        self.graph = self._build_graph()

    # --- graph ------------------------------------------------------------------

    def _build_graph(self):
        g = StateGraph(ChatState)
        g.add_node("route", self._node("router", self._route))
        g.add_node("retrieve", self._node("retriever", self._retrieve))
        g.add_node("answer", self._node("answerer", self._answer))
        g.add_node("check", self._node("checker", self._check))
        g.add_node("respond", self._respond)
        g.add_node("unknown", self._unknown)
        g.add_node("small_talk", lambda s: {"reply": Reply("small_talk", SMALL_TALK_TEXT)}) # type: ignore
        g.add_node("off_topic", lambda s: {"reply": Reply("off_topic", OFF_TOPIC_TEXT)}) # type: ignore

        g.add_edge(START, "route")
        g.add_conditional_edges("route", self._after_route)
        g.add_conditional_edges("retrieve", lambda s: END if "reply" in s else ("answer" if s["hits"] else "unknown"))
        g.add_conditional_edges("answer", lambda s: END if "reply" in s else ("check" if s["draft"].found else "unknown"))
        g.add_conditional_edges("check", lambda s: END if "reply" in s else ("respond" if s["verdict"].supported else "unknown"))
        for terminal in ("respond", "unknown", "small_talk", "off_topic"):
            g.add_edge(terminal, END)
        return g.compile()

    @staticmethod
    def _after_route(state: ChatState) -> str:
        if "reply" in state:
            return END
        return {Intent.SMALL_TALK: "small_talk", Intent.OFF_TOPIC: "off_topic"}.get(
            state["route"].intent, "retrieve") # type: ignore

    def _node(self, agent: str, fn: Callable[[ChatState], tuple[dict, str]]):
        """Wrap an agent step: time it, log its decision, and turn any failure into a reply."""
        def run(state: ChatState) -> dict:
            start = time.perf_counter()
            try:
                update, decision = fn(state)
            except Exception as exc:  # noqa: BLE001 - one bad step must never crash the chat
                log.exception("%s failed", agent)
                kind = "model unavailable or timed out" if isinstance(exc, LLMError) else "unexpected error"
                update, decision = {"reply": Reply("error", ERROR_TEXT)}, f"FAILED ({kind}): {exc}"
            ms = round((time.perf_counter() - start) * 1000)
            log.info("[%s] %s (%d ms)", agent, decision, ms)
            return {**update, "trace": [{"agent": agent, "decision": decision, "ms": ms}]}
        return run

    # --- agent steps --------------------------------------------------------------

    def _route(self, state: ChatState):
        route = self.router.run(state["message"], state.get("history", [])) # type: ignore
        return {"route": route}, f"intent={route.intent.value} query={route.query!r} ({route.reason})"

    def _retrieve(self, state: ChatState):
        hits = self.retriever.run(state["route"].query) # type: ignore
        found = ", ".join(f"{h.score:.2f} {h.url}" for h in hits) or "nothing above threshold"
        return {"hits": hits}, f"{len(hits)} passages: {found}"

    def _answer(self, state: ChatState):
        draft = self.answerer.run(state["route"].query, state["hits"]) # type: ignore
        decision = "not in passages" if not draft.found else f"draft={draft.text!r}"
        return {"draft": draft}, decision

    def _check(self, state: ChatState):
        verdict = self.checker.run(state["draft"].text, state["hits"]) # type: ignore
        decision = "supported" if verdict.supported else f"REJECTED: {verdict.problems}"
        return {"verdict": verdict}, decision

    @staticmethod
    def _respond(state: ChatState) -> dict:
        draft = state["draft"] # type: ignore
        return {"reply": Reply("answered", draft.text, _unique_sources(draft.sources))}

    @staticmethod
    def _unknown(state: ChatState) -> dict:
        return {"reply": Reply("unknown", UNKNOWN_TEXT, [("Contact Us", config.CONTACT_URL)])}

    # --- public -------------------------------------------------------------------

    def warm_up(self) -> None:
        """Load both models before the first visitor arrives. Without this the first question
        also pays for loading the embedding model, building BM25 and loading the LLM (~3-8 s)."""
        self.retriever.run("AIS services")
        try:
            self.llm.chat("Reply with OK.", "OK")
        except LLMError as exc:
            log.warning("LLM warm-up failed (is Ollama running?): %s", exc)

    def run(self, message: str, history: list[tuple[str, str]] | None = None) -> Reply:
        message = message.strip()
        if not message:
            return Reply("small_talk", SMALL_TALK_TEXT)
        state = self.graph.invoke({"message": message, "history": history or [], "trace": []})
        reply: Reply = state["reply"]
        reply.trace = state["trace"]
        reply.query = state["route"].query if "route" in state else message
        return reply


class Conversation:
    """One visitor's chat. Holds the history; the orchestrator itself is shared and stateless."""

    def __init__(self, orchestrator: Orchestrator, max_turns: int = config.HISTORY_TURNS):
        self.orchestrator = orchestrator
        self.max_turns = max_turns
        self.history: list[tuple[str, str]] = []

    def ask(self, message: str) -> Reply:
        reply = self.orchestrator.run(message, self.history[-self.max_turns:])
        if reply.status != "error":  # a failed turn should not confuse the next follow-up
            self.history.append((message, reply.text))
        return reply
