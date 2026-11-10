from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_TITLE", "Chat with PDF")
    app_public_url: str = os.getenv("APP_PUBLIC_URL", "http://localhost:8501")

    data_dir: Path = Path(os.getenv("APP_DATA_DIR", "data"))
    uploads_dir: Path = Path(os.getenv("UPLOADS_DIR", "data/uploads"))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "data/chroma"))
    manifest_dir: Path = Path(os.getenv("MANIFEST_DIR", "data/manifests"))
    markdown_cache_dir: Path = Path(os.getenv("MARKDOWN_CACHE_DIR", "data/markdown_cache"))

    default_chunk_size: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "900"))
    default_chunk_overlap: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "120"))
    default_retrieval_k: int = int(os.getenv("DEFAULT_RETRIEVAL_K", "6"))

    default_embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
    default_embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    sambanova_api_key: str = os.getenv("SAMBANOVA_API_KEY", "")

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.uploads_dir,
            self.chroma_dir,
            self.manifest_dir,
            self.markdown_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()