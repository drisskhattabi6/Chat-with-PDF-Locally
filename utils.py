import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import requests


def get_logger(name: str = "chatpdf") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def build_collection_name(doc_hash: str, config_payload: dict[str, Any]) -> str:
    cfg_hash = sha256_text(json.dumps(config_payload, sort_keys=True, ensure_ascii=False))
    raw = f"pdf_{doc_hash[:12]}_{cfg_hash[:12]}"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", raw)[:63]


def strip_think_tags(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def human_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)

    if hours:
        return f"{hours}h {mins}m {secs}s"
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def flatten_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = json.dumps(value, ensure_ascii=False)[:1500]
    return clean


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def list_ollama_models(base_url: str, timeout: int = 8) -> list[str]:
    base_url = base_url.rstrip("/")
    response = requests.get(f"{base_url}/api/tags", timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return sorted([item["name"] for item in data.get("models", []) if item.get("name")])


def coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(str(item["text"]))
        return "\n".join(part for part in parts if part)

    if content is None:
        return ""

    return str(content)