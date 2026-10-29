"""Retrieval agent: find the passages most likely to answer the question.

No LLM here on purpose: search is fast, deterministic and easy to test.

Hybrid search: vector search (meaning) and BM25 (exact words) each return a
ranked list, merged with reciprocal rank fusion (RRF). RRF uses ranks, not
scores, so the two very different score scales never need to be compared.

Its one judgement is "is anything relevant enough?". If the best vector match
is below MIN_RELEVANCE we return nothing, and the orchestrator says
"I don't know" instead of letting the answer agent improvise from weak matches.
"""

from __future__ import annotations

import logging

from .. import config
from ..keyword_search import BM25
from ..vectorstore import Hit, VectorStore

log = logging.getLogger("agent.retriever")

CANDIDATES = 20   # how deep each search looks before fusing
RRF_K = 60        # standard RRF constant: dampens the gap between rank 1 and rank 2
MAX_PER_PAGE = 2  # near-identical job-listing pages otherwise fill every slot (seen on "offices in India")


class RetrievalAgent:
    def __init__(self, store: VectorStore, k: int = config.TOP_K,
                 min_relevance: float = config.MIN_RELEVANCE):
        self.store = store
        self.k = k
        self.min_relevance = min_relevance
        self._chunks: list[Hit] | None = None
        self._bm25: BM25 | None = None

    def _keyword_index(self) -> tuple[list[Hit], BM25]:
        if self._bm25 is None:  # built lazily from the vector store, so both always match
            self._chunks = self.store.all_chunks()
            self._bm25 = BM25([f"{c.title}\n{c.text}" for c in self._chunks])
        return self._chunks, self._bm25

    def run(self, query: str) -> list[Hit]:
        dense = self.store.search(query, k=CANDIDATES)
        best = dense[0].score if dense else 0.0
        if best < self.min_relevance:
            log.info("query=%r best=%.3f below threshold", query, best)
            return []

        chunks, bm25 = self._keyword_index()
        keyword = [chunks[i] for i, _ in bm25.top(query, CANDIDATES)]

        fused: dict[tuple[str, str], float] = {}
        by_key: dict[tuple[str, str], Hit] = {}
        for ranked in (dense, keyword):
            for rank, hit in enumerate(ranked):
                key = (hit.url, hit.text)
                fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
                if key not in by_key or hit.score > by_key[key].score:
                    by_key[key] = hit  # prefer the copy that carries the cosine score
        top, per_page = [], {}
        for key in sorted(fused, key=fused.get, reverse=True):
            page = key[0].split("?")[0]  # featured-jobs?tag=x is the same page as featured-jobs
            if per_page.get(page, 0) < MAX_PER_PAGE:
                per_page[page] = per_page.get(page, 0) + 1
                top.append(key)
            if len(top) == self.k:
                break
        log.info("query=%r best=%.3f dense=%d keyword=%d kept=%d", query, best, len(dense), len(keyword), len(top))
        return [by_key[key] for key in top]
