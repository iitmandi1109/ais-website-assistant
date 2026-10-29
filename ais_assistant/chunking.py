"""Split crawled pages into small passages ("chunks") for the vector index.

Why chunk at all? An embedding is one vector per piece of text. A whole page
averages many topics into one blurry vector; a ~900 character chunk is about
one idea, so a question lands close to the chunk that answers it, and the
answer agent receives a short, focused passage instead of a long page.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from . import config


@dataclass
class Chunk:
    id: str
    url: str
    title: str
    text: str                                        # what the answer agent will read
    also_at: list[str] = field(default_factory=list)  # other URLs with identical text

    @property
    def embed_text(self) -> str:
        """What we embed: the page title gives every chunk its context
        ("Who We Serve | Banking"), even when the chunk itself never says it."""
        return f"{self.title}\n{self.text}"


def clean_title(title: str) -> str:
    """'AIS Info | Who We Serve | Banking' -> 'Who We Serve | Banking'."""
    return re.sub(r"^AIS Info\s*\|\s*", "", title).strip() or title


def _split_long_line(line: str, max_chars: int) -> list[str]:
    """Break an over-long paragraph at sentence ends, packing sentences up to max_chars."""
    pieces: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", line):
        while len(sentence) > max_chars:  # one giant sentence: hard cut (rare)
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def split_text(text: str, max_chars: int = config.CHUNK_MAX_CHARS,
               min_chars: int = config.CHUNK_MIN_CHARS,
               overlap_max: int = config.OVERLAP_MAX_CHARS) -> list[str]:
    """Pack consecutive lines into chunks of at most ~max_chars.

    * Lines are never split mid-way unless a single line is longer than max_chars.
    * If the previous chunk ended on a short line (typically a heading) it is
      repeated at the start of the next chunk, so the heading keeps its body.
    * A small leftover at the end is merged back into the previous chunk, so a
      chunk can exceed max_chars by up to min_chars (never a tiny orphan chunk).
    """
    units: list[str] = []
    for line in text.split("\n"):
        if line.strip():
            units.extend([line] if len(line) <= max_chars else _split_long_line(line, max_chars))

    chunks: list[list[str]] = []   # each chunk is a list of lines
    new_lines: list[int] = []      # how many lines in each chunk are new (not carried overlap)
    current: list[str] = []
    carried = 0
    size = 0
    for unit in units:
        if len(current) > carried and size + len(unit) + 1 > max_chars:
            chunks.append(current)
            new_lines.append(len(current) - carried)
            tail = current[-1]
            current, carried = ([tail], 1) if len(tail) <= overlap_max else ([], 0)
            size = sum(len(l) + 1 for l in current)
        current.append(unit)
        size += len(unit) + 1
    if len(current) > carried:
        chunks.append(current)
        new_lines.append(len(current) - carried)

    # Merge a tiny final chunk into the one before it (its repeated heading is dropped).
    if len(chunks) > 1:
        fresh = chunks[-1][len(chunks[-1]) - new_lines[-1]:]  # the lines that are not carried overlap
        if sum(len(line) for line in fresh) < min_chars:
            chunks.pop()
            chunks[-1].extend(fresh)
    return ["\n".join(c) for c in chunks]


def _fingerprint(text: str) -> str:
    """Hash of the text ignoring case and spacing, to spot copies of the same passage."""
    return hashlib.sha1(re.sub(r"\s+", " ", text).lower().encode()).hexdigest()


def build_chunks(pages: list[dict]) -> list[Chunk]:
    """Turn crawled pages (dicts from pages.jsonl) into de-duplicated chunks.

    The same product often appears under two URLs (e.g. /expertise/technology/bk-tripack
    and /expertise/loan-servicing/bk-tripack). We index the text once and remember the
    other URLs in `also_at`, so search results are not filled with copies.
    """
    chunks: list[Chunk] = []
    first_seen: dict[str, Chunk] = {}
    for page in pages:
        lines = page["text"].split("\n")
        if lines and lines[0] == page["title"]:   # the <title> was captured as the first line
            lines = lines[1:]
        body = "\n".join(lines)
        if len(body) < config.MIN_PAGE_CHARS // 2:  # e.g. a page that was only a form
            continue

        title = clean_title(page["title"])
        url_key = hashlib.sha1(page["url"].encode()).hexdigest()[:10]
        for i, text in enumerate(split_text(body)):
            fp = _fingerprint(text)
            if fp in first_seen:
                if page["url"] not in first_seen[fp].also_at:
                    first_seen[fp].also_at.append(page["url"])
                continue
            chunk = Chunk(f"{url_key}-{i:02d}", page["url"], title, text)
            first_seen[fp] = chunk
            chunks.append(chunk)
    return chunks
