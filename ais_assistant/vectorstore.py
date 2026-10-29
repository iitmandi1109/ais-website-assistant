"""Local embeddings + local vector database.

    Embedding model : BAAI/bge-small-en-v1.5 (open source, ~130 MB, runs on CPU)
    Vector database : Chroma, stored in data/chroma/ (a folder on disk)

Nothing here calls a hosted service. The only network use is the one-time
download of the embedding model from Hugging Face on first run.

Try it from the command line once the index is built:

    python -m ais_assistant.vectorstore "which industries does AIS serve"
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # must be set before the HF libraries load

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from . import config
from .chunking import Chunk

log = logging.getLogger("vectorstore")


@dataclass
class Hit:
    """One search result, as the retrieval agent will pass it on."""
    text: str
    url: str
    title: str
    score: float  # cosine similarity: 1.0 = identical meaning, ~0 = unrelated


class VectorStore:
    def __init__(self, path=config.VECTOR_DIR):
        # Telemetry off: the data-governance rule is that nothing leaves the machine.
        self._client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:  # lazy: loading takes a few seconds
            log.info("Loading embedding model %s", config.EMBEDDING_MODEL)
            try:
                # Already downloaded: load from disk without contacting Hugging Face at all.
                self._model = SentenceTransformer(config.EMBEDDING_MODEL, local_files_only=True)
            except OSError:
                log.info("Model not cached yet; downloading it once from Hugging Face")
                self._model = SentenceTransformer(config.EMBEDDING_MODEL)
        return self._model

    def _embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if is_query:  # bge expects this instruction in front of search queries only
            texts = [config.QUERY_PREFIX + t for t in texts]
        # normalize_embeddings=True -> unit-length vectors, so cosine similarity is a dot product
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                 show_progress_bar=False).tolist()

    def build(self, chunks: list[Chunk]) -> None:
        """(Re)create the collection from scratch. Safe to run repeatedly."""
        try:
            self._client.delete_collection(config.COLLECTION_NAME)
        except Exception:  # noqa: BLE001 - chromadb raises a different error per version when absent
            pass
        collection = self._client.create_collection(
            config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=self._embed([c.embed_text for c in chunks]),
            metadatas=[{"url": c.url, "title": c.title} for c in chunks],
        )
        log.info("Indexed %d chunks", collection.count())

    def all_chunks(self) -> list[Hit]:
        """Every stored passage (score 0), for the keyword index."""
        data = self._client.get_collection(config.COLLECTION_NAME).get(include=["documents", "metadatas"])
        return [Hit(text=doc, url=meta["url"], title=meta["title"], score=0.0)
                for doc, meta in zip(data["documents"], data["metadatas"])]

    def search(self, query: str, k: int = config.TOP_K) -> list[Hit]:
        collection = self._client.get_collection(config.COLLECTION_NAME)
        result = collection.query(query_embeddings=self._embed([query], is_query=True), n_results=k)
        return [
            Hit(text=doc, url=meta["url"], title=meta["title"], score=1.0 - dist)
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('usage: python -m ais_assistant.vectorstore "your question"')
    query = " ".join(sys.argv[1:])
    for rank, hit in enumerate(VectorStore().search(query), start=1):
        snippet = hit.text.replace("\n", " ")[:160]
        print(f"{rank}. score={hit.score:.3f}  {hit.url}\n   [{hit.title}] {snippet}...\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
