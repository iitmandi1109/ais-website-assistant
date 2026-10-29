"""Build the search index from the saved website copy.

    python -m ais_assistant.index

Reads data/raw/pages.jsonl (made by the crawler), splits it into chunks,
embeds them, and stores them in data/chroma/. Takes about a minute on a laptop CPU.
"""

import logging
import statistics

from .chunking import build_chunks
from .crawler import load_pages
from .vectorstore import VectorStore

log = logging.getLogger("index")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pages = load_pages()
    chunks = build_chunks(pages)
    sizes = [len(c.text) for c in chunks]
    log.info("%d pages -> %d chunks (median %d chars)", len(pages), len(chunks), statistics.median(sizes))
    VectorStore().build(chunks)


if __name__ == "__main__":
    main()
