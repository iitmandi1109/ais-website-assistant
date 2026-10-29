"""BM25 keyword search over the same chunks as the vector index.

Why: on the real data, dense search alone missed exact-term questions. "Who is
the CEO?" ranked ten other leadership bios above the one containing "Chairman
and CEO", and "How many employees?" never surfaced "1400 Employees Worldwide".
Keywords catch exactly these; embeddings catch paraphrases. The retrieval agent
uses both. ~350 chunks fit in memory easily, so this is plain Python.
"""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "at", "by", "from",
    "is", "are", "was", "were", "be", "do", "does", "did", "what", "which", "who", "whom",
    "how", "when", "where", "why", "can", "i", "you", "your", "it", "its", "this", "that",
    "there", "their", "they", "me", "my", "we", "our", "us", "have", "has", "any", "about", "ais",
}


def stem(word: str) -> str:
    """Very light stemming: drop a plural "s" so "offices" matches "Office:" on the contact page.
    A full stemmer (Porter) would also merge "certified"/"certifications", but over-merges short
    words; plural-only was enough for the misses seen in evaluation."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def tokenize(text: str) -> list[str]:
    # "ais" is a stopword here: it is on every page, so it carries no signal
    return [stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.docs), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        """(document index, score) of the k best matches with a score above zero."""
        terms = tokenize(query)
        scores = []
        for i, (doc, length) in enumerate(zip(self.docs, self.lengths)):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            if s > 0:
                scores.append((i, s))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]
