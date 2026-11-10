from pathlib import Path

import chromadb

from schemas import ChunkRecord, RetrievalHit
from utils import flatten_metadata


class ChromaVectorStore:
    def __init__(self, persist_directory: str | Path, collection_name: str, embedder):
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self.embedder = embedder

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(self, chunks: list[ChunkRecord], batch_size: int = 32) -> None:
        if not chunks:
            raise ValueError("No chunks to index.")

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [item.text for item in batch]
            embeddings = self.embedder.embed_texts(texts)

            self.collection.upsert(
                ids=[item.id for item in batch],
                documents=texts,
                metadatas=[flatten_metadata(item.metadata) for item in batch],
                embeddings=embeddings,
            )

    def query(self, query_text: str, n_results: int = 10) -> list[RetrievalHit]:
        if self.count() == 0:
            return []

        query_embedding = self.embedder.embed_query(query_text)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[RetrievalHit] = []
        for doc, meta, distance in zip(docs, metas, distances):
            hits.append(
                RetrievalHit(
                    text=doc,
                    metadata=meta or {},
                    distance=distance,
                )
            )
        return hits


def delete_collection(persist_directory: str | Path, collection_name: str) -> None:
    client = chromadb.PersistentClient(path=str(persist_directory))
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass