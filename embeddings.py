from functools import lru_cache

import requests


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = _load_sentence_transformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class OllamaEmbedder:
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model_name,
                "input": texts,
                "truncate": True,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embeddings", [])

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def build_embedder(provider_name: str, model_name: str, base_url: str | None = None):
    if provider_name == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name=model_name)

    if provider_name == "ollama":
        if not base_url:
            raise ValueError("Ollama base URL is required for Ollama embeddings.")
        return OllamaEmbedder(model_name=model_name, base_url=base_url)

    raise ValueError(f"Unsupported embedding provider: {provider_name}")