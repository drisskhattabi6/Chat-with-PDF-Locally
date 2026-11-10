from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExtractedDocument:
    doc_id: str
    source_name: str
    extractor_name: str
    markdown: str
    page_chunks: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    docling_document: Any | None = None


@dataclass
class ChunkRecord:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: Optional[float] = None
    score: float = 0.0


@dataclass
class ChatResult:
    text: str
    provider: str
    model: str
    latency_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    raw: dict[str, Any] = field(default_factory=dict)