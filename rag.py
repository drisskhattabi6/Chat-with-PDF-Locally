import numpy as np
from rank_bm25 import BM25Okapi

from schemas import ChatResult, RetrievalHit
from utils import simple_tokenize, strip_think_tags


SYSTEM_PROMPT = """You are a grounded PDF question-answering assistant.

Rules:
1. Use only the provided source context.
2. If the answer is incomplete or unsupported by the sources, say so clearly.
3. Cite the source IDs inline like [S1], [S2].
4. Do not invent facts outside the retrieved context.
5. Prefer precise answers over long answers.
"""


class RAGService:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def answer(
        self,
        question: str,
        history: list[dict],
        chat_client,
        model: str,
        temperature: float,
        max_tokens: int,
        top_k: int,
    ) -> dict:
        hits = self.vector_store.query(question, n_results=max(20, top_k * 4))

        if not hits:
            return {
                "answer": "I couldn't find relevant indexed content for that question in the current PDF.",
                "sources": [],
                "chat_result": ChatResult(
                    text="",
                    provider="",
                    model="",
                    latency_s=0.0,
                ),
            }

        reranked_hits = self._rerank_hits(hits, question, top_k=top_k)
        context_text, source_payload = self._build_context(reranked_hits)

        recent_history = history[-6:] if history else []
        history_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history
        ).strip()

        user_prompt = f"""
Conversation history:
{history_text if history_text else "(none)"}

Question:
{question}

Source context:
{context_text}

Answer the question using only the source context above.
""".strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        chat_result = chat_client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer_text = strip_think_tags(chat_result.text)

        return {
            "answer": answer_text,
            "sources": source_payload,
            "chat_result": chat_result,
        }

    def _rerank_hits(self, hits: list[RetrievalHit], query: str, top_k: int) -> list[RetrievalHit]:
        if len(hits) <= top_k:
            for hit in hits:
                hit.score = 1.0
            return hits

        tokenized_corpus = [simple_tokenize(hit.text) or ["_"] for hit in hits]
        query_tokens = simple_tokenize(query)

        if query_tokens:
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = np.array(bm25.get_scores(query_tokens), dtype=float)
        else:
            bm25_scores = np.zeros(len(hits), dtype=float)

        semantic_scores = []
        for hit in hits:
            distance = hit.distance if hit.distance is not None else 1.0
            semantic_scores.append(1.0 / (1.0 + float(distance)))
        semantic_scores = np.array(semantic_scores, dtype=float)

        bm25_norm = self._normalize(bm25_scores)
        sem_norm = self._normalize(semantic_scores)

        combined = (0.35 * bm25_norm) + (0.65 * sem_norm)
        ranked_indices = np.argsort(combined)[::-1][:top_k]

        reranked: list[RetrievalHit] = []
        for idx in ranked_indices:
            hit = hits[int(idx)]
            hit.score = float(combined[int(idx)])
            reranked.append(hit)

        return reranked

    def _build_context(self, hits: list[RetrievalHit]) -> tuple[str, list[dict]]:
        blocks: list[str] = []
        source_payload: list[dict] = []

        for idx, hit in enumerate(hits, start=1):
            source_id = f"S{idx}"
            metadata = dict(hit.metadata or {})

            labels: list[str] = []
            if metadata.get("page") is not None:
                labels.append(f"page {metadata['page']}")
            for key in ("h1", "h2", "h3", "h4"):
                if metadata.get(key):
                    labels.append(f"{key}: {metadata[key]}")

            label_text = ", ".join(labels) if labels else "no extra metadata"

            blocks.append(f"[{source_id}] ({label_text})\n{hit.text}")
            source_payload.append(
                {
                    "id": source_id,
                    "text": hit.text,
                    "metadata": metadata,
                    "score": round(hit.score, 4),
                }
            )

        return "\n\n" + ("\n\n-----\n\n".join(blocks)) + "\n", source_payload

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values

        min_v = float(np.min(values))
        max_v = float(np.max(values))
        if abs(max_v - min_v) < 1e-12:
            return np.ones_like(values)

        return (values - min_v) / (max_v - min_v)