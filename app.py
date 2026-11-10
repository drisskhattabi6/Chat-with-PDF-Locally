import io
from pathlib import Path

import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

from chunking import CHUNKING_LABELS, chunk_document
from config import SETTINGS
from embeddings import build_embedder
from pdf_extractors import EXTRACTOR_LABELS, extract_pdf
from providers import PROVIDER_PRESETS, build_chat_client, get_default_api_key
from rag import RAGService
from utils import (
    build_collection_name,
    get_logger,
    human_duration,
    list_ollama_models,
    read_json,
    safe_filename,
    sha256_bytes,
    write_json,
)
from vector_store import ChromaVectorStore, delete_collection

SETTINGS.ensure_directories()
logger = get_logger()

st.set_page_config(page_title=SETTINGS.app_name, page_icon="📄", layout="wide")


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "upload_hash" not in st.session_state:
        st.session_state.upload_hash = ""
    if "active_manifest" not in st.session_state:
        st.session_state.active_manifest = None
    if "fetched_models" not in st.session_state:
        st.session_state.fetched_models = {}


def generate_chat_pdf(messages: list[dict], manifest: dict | None) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 40
    margin_x = 40
    line_height = 16
    max_width = width - (margin_x * 2)

    pdf.setFont("Helvetica-Bold", 16)
    title = "Conversation History"
    title_width = pdf.stringWidth(title, "Helvetica-Bold", 16)
    pdf.drawString((width - title_width) / 2, y, title)

    y -= 25
    pdf.setFont("Helvetica", 10)

    if manifest:
        doc_line = f"PDF: {manifest.get('pdf_name', 'Unknown')} | Extractor: {manifest.get('extractor', '')} | Chunking: {manifest.get('chunk_method', '')}"
        for line in simpleSplit(doc_line, pdf._fontname, pdf._fontsize, max_width):
            pdf.drawString(margin_x, y, line)
            y -= line_height

    y -= 10
    pdf.setFont("Helvetica", 11)

    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        text = f"{role}: {msg['content']}"

        wrapped = simpleSplit(text, pdf._fontname, pdf._fontsize, max_width)
        for line in wrapped:
            if y < 50:
                pdf.showPage()
                pdf.setFont("Helvetica", 11)
                y = height - 40
            pdf.drawString(margin_x, y, line)
            y -= line_height

        if msg.get("meta_caption"):
            pdf.setFont("Helvetica-Oblique", 9)
            for line in simpleSplit(msg["meta_caption"], pdf._fontname, pdf._fontsize, max_width):
                if y < 50:
                    pdf.showPage()
                    pdf.setFont("Helvetica-Oblique", 9)
                    y = height - 40
                pdf.drawString(margin_x, y, line)
                y -= 12
            pdf.setFont("Helvetica", 11)

        y -= 8

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander("Sources used", expanded=False):
        for src in sources:
            metadata = src.get("metadata", {}) or {}
            label_parts = [src["id"]]

            if metadata.get("page") is not None:
                label_parts.append(f"page {metadata['page']}")
            for key in ("h1", "h2", "h3", "h4"):
                if metadata.get(key):
                    label_parts.append(f"{key}: {metadata[key]}")

            st.markdown(f"**{' | '.join(label_parts)}**")
            st.caption(src.get("text", "")[:1200])


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])
            if message.get("meta_caption"):
                st.caption(message["meta_caption"])


def build_meta_caption(provider_name: str, model_name: str, chat_result, source_count: int) -> str:
    parts = [
        f"Provider: {provider_name}",
        f"Model: {model_name}",
        f"Latency: {human_duration(chat_result.latency_s)}",
        f"Retrieved sources: {source_count}",
    ]

    if chat_result.prompt_tokens is not None:
        parts.append(f"Prompt tokens: {chat_result.prompt_tokens}")
    if chat_result.completion_tokens is not None:
        parts.append(f"Completion tokens: {chat_result.completion_tokens}")
    if chat_result.total_tokens is not None:
        parts.append(f"Total tokens: {chat_result.total_tokens}")

    return " | ".join(parts)


def try_get_ollama_models(base_url: str) -> list[str]:
    try:
        return list_ollama_models(base_url)
    except Exception:
        return []


init_state()

st.title("💬 Chat with PDF")
st.caption("Refactored for robust indexing, multi-provider LLM support, and selectable chunking.")

with st.sidebar:
    st.header("Document")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    extractor_key = st.selectbox(
        "PDF extractor",
        options=list(EXTRACTOR_LABELS.keys()),
        format_func=lambda key: EXTRACTOR_LABELS[key],
    )

    chunking_options = ["recursive", "markdown_recursive", "token"]
    if extractor_key == "pymupdf4llm":
        chunking_options.append("page_recursive")

    chunk_method = st.selectbox(
        "Chunking method",
        options=chunking_options,
        format_func=lambda key: CHUNKING_LABELS[key],
    )

    chunk_size = st.number_input(
        "Chunk size",
        min_value=100,
        max_value=4000,
        value=SETTINGS.default_chunk_size,
        step=100,
        help="Chars for recursive splitters, tokens for token splitter.",
    )

    chunk_overlap = st.number_input(
        "Chunk overlap",
        min_value=0,
        max_value=1200,
        value=SETTINGS.default_chunk_overlap,
        step=20,
    )

    st.divider()
    st.header("Embeddings")

    embedding_provider = st.selectbox(
        "Embedding provider",
        options=["sentence_transformers", "ollama"],
        format_func=lambda x: "SentenceTransformers (local)" if x == "sentence_transformers" else "Ollama",
        index=0 if SETTINGS.default_embedding_provider == "sentence_transformers" else 1,
    )

    if embedding_provider == "sentence_transformers":
        embedding_model = st.text_input(
            "Embedding model",
            value=SETTINGS.default_embedding_model,
        )
        embedding_base_url = ""
    else:
        embedding_base_url = st.text_input(
            "Ollama base URL",
            value=SETTINGS.ollama_base_url,
        )
        embedding_ollama_models = try_get_ollama_models(embedding_base_url)
        if embedding_ollama_models:
            default_embed_index = 0
            if "nomic-embed-text:latest" in embedding_ollama_models:
                default_embed_index = embedding_ollama_models.index("nomic-embed-text:latest")
            embedding_model = st.selectbox(
                "Embedding model",
                options=embedding_ollama_models,
                index=default_embed_index,
            )
        else:
            embedding_model = st.text_input(
                "Embedding model",
                value="nomic-embed-text:latest",
            )

    retrieval_k = st.slider(
        "Retrieved chunks for answer",
        min_value=1,
        max_value=12,
        value=SETTINGS.default_retrieval_k,
    )

    st.divider()
    st.header("Answer model")

    provider_name = st.selectbox(
        "LLM provider",
        options=list(PROVIDER_PRESETS.keys()),
    )
    provider_preset = PROVIDER_PRESETS[provider_name]

    llm_base_url = st.text_input("Base URL", value=provider_preset.base_url)

    if provider_preset.needs_api_key:
        llm_api_key = st.text_input(
            "API key",
            type="password",
            value=get_default_api_key(provider_name),
        )
    else:
        llm_api_key = ""

    fetch_key = f"{provider_name}|{llm_base_url}"
    fetch_models_button = st.button("Fetch provider models", use_container_width=True)

    if fetch_models_button:
        try:
            client = build_chat_client(
                provider_name=provider_name,
                base_url=llm_base_url,
                api_key=llm_api_key,
            )
            models = client.list_models()
            st.session_state.fetched_models[fetch_key] = models
            st.success(f"Fetched {len(models)} models.")
        except Exception as exc:
            st.error(f"Could not fetch models: {exc}")

    if provider_name == "Ollama":
        auto_models = try_get_ollama_models(llm_base_url)
        if auto_models:
            llm_model = st.selectbox("Model", options=auto_models)
        else:
            llm_model = st.text_input("Model", value="")
    else:
        fetched_models = st.session_state.fetched_models.get(fetch_key, [])
        if fetched_models:
            llm_model = st.selectbox("Model", options=fetched_models)
        else:
            llm_model = st.text_input("Model", value="")

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.1, step=0.1)
    max_tokens = st.number_input("Max output tokens", min_value=128, max_value=8192, value=1200, step=128)

    st.divider()
    st.header("Actions")

    process_button = st.button("Process / Load index", use_container_width=True)
    clear_chat_button = st.button("Clear chat", use_container_width=True)
    delete_index_button = st.button("Delete current index", use_container_width=True)

    download_payload = generate_chat_pdf(st.session_state.messages, st.session_state.active_manifest)
    st.download_button(
        "Download chat as PDF",
        data=download_payload,
        file_name="chat_history.pdf",
        mime="application/pdf",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )

if clear_chat_button:
    st.session_state.messages = []
    st.rerun()

if delete_index_button:
    manifest = st.session_state.active_manifest
    if manifest:
        delete_collection(SETTINGS.chroma_dir, manifest["collection_name"])
        manifest_path = Path(manifest["manifest_path"])
        markdown_cache_path = Path(manifest["markdown_cache_path"])

        if manifest_path.exists():
            manifest_path.unlink()
        if markdown_cache_path.exists():
            markdown_cache_path.unlink()

        st.session_state.active_manifest = None
        st.session_state.messages = []
        st.success("Current index deleted.")
        st.rerun()
    else:
        st.warning("No active index to delete.")

if uploaded_pdf is not None:
    current_pdf_bytes = uploaded_pdf.getvalue()
    current_pdf_hash = sha256_bytes(current_pdf_bytes)

    if st.session_state.upload_hash and st.session_state.upload_hash != current_pdf_hash:
        st.session_state.messages = []
        st.session_state.active_manifest = None

    st.session_state.upload_hash = current_pdf_hash
else:
    current_pdf_bytes = b""
    current_pdf_hash = ""

if process_button:
    if uploaded_pdf is None:
        st.error("Please upload a PDF first.")
    else:
        try:
            original_name = uploaded_pdf.name
            safe_name = safe_filename(original_name)
            pdf_path = SETTINGS.uploads_dir / f"{current_pdf_hash}_{safe_name}"
            pdf_path.write_bytes(current_pdf_bytes)

            index_config = {
                "extractor": extractor_key,
                "chunk_method": chunk_method,
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "embedding_base_url": embedding_base_url,
            }

            collection_name = build_collection_name(current_pdf_hash, index_config)
            manifest_path = SETTINGS.manifest_dir / f"{collection_name}.json"
            markdown_cache_path = SETTINGS.markdown_cache_dir / f"{collection_name}.md"

            embedder = build_embedder(
                provider_name=embedding_provider,
                model_name=embedding_model,
                base_url=embedding_base_url or None,
            )
            vector_store = ChromaVectorStore(
                persist_directory=SETTINGS.chroma_dir,
                collection_name=collection_name,
                embedder=embedder,
            )

            if manifest_path.exists() and vector_store.count() > 0:
                manifest = read_json(manifest_path)
                st.session_state.active_manifest = manifest
                st.session_state.messages = []
                st.success("Existing index loaded from cache.")
            else:
                with st.spinner("Extracting PDF..."):
                    extracted = extract_pdf(
                        pdf_path=pdf_path,
                        extractor_key=extractor_key,
                        doc_id=current_pdf_hash,
                    )

                if not extracted.markdown.strip():
                    raise ValueError("Extraction returned empty text.")

                with st.spinner("Chunking document..."):
                    chunks = chunk_document(
                        extracted=extracted,
                        method=chunk_method,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                    )

                if not chunks:
                    raise ValueError("Chunking returned zero chunks.")

                with st.spinner("Embedding and indexing..."):
                    vector_store.reset()
                    vector_store.index_chunks(chunks)

                markdown_cache_path.write_text(extracted.markdown, encoding="utf-8")

                manifest = {
                    "pdf_name": original_name,
                    "pdf_path": str(pdf_path),
                    "doc_hash": current_pdf_hash,
                    "collection_name": collection_name,
                    "extractor": extractor_key,
                    "chunk_method": chunk_method,
                    "chunk_size": int(chunk_size),
                    "chunk_overlap": int(chunk_overlap),
                    "chunk_count": len(chunks),
                    "embedding_provider": embedding_provider,
                    "embedding_model": embedding_model,
                    "embedding_base_url": embedding_base_url,
                    "manifest_path": str(manifest_path),
                    "markdown_cache_path": str(markdown_cache_path),
                }
                write_json(manifest_path, manifest)

                st.session_state.active_manifest = manifest
                st.session_state.messages = []
                st.success(f"Processing complete. Indexed {len(chunks)} chunks.")

        except Exception as exc:
            logger.exception("Processing failed")
            st.error(f"Processing failed: {exc}")

active_manifest = st.session_state.active_manifest

if active_manifest:
    st.success(
        f"Ready: {active_manifest['pdf_name']} | extractor={active_manifest['extractor']} | "
        f"chunking={active_manifest['chunk_method']} | chunks={active_manifest['chunk_count']}"
    )
    with st.expander("Active index details", expanded=False):
        st.json(
            {
                "pdf_name": active_manifest["pdf_name"],
                "extractor": active_manifest["extractor"],
                "chunk_method": active_manifest["chunk_method"],
                "chunk_size": active_manifest["chunk_size"],
                "chunk_overlap": active_manifest["chunk_overlap"],
                "chunk_count": active_manifest["chunk_count"],
                "embedding_provider": active_manifest["embedding_provider"],
                "embedding_model": active_manifest["embedding_model"],
                "collection_name": active_manifest["collection_name"],
            }
        )
else:
    st.info("Upload a PDF and click 'Process / Load index'.")

st.markdown("---")
render_chat_history()

question = st.chat_input(
    "Ask something about the indexed PDF...",
    disabled=active_manifest is None,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            manifest = st.session_state.active_manifest
            if not manifest:
                raise ValueError("No active index found.")

            embedder = build_embedder(
                provider_name=manifest["embedding_provider"],
                model_name=manifest["embedding_model"],
                base_url=manifest.get("embedding_base_url") or None,
            )

            vector_store = ChromaVectorStore(
                persist_directory=SETTINGS.chroma_dir,
                collection_name=manifest["collection_name"],
                embedder=embedder,
            )

            rag = RAGService(vector_store=vector_store)
            chat_client = build_chat_client(
                provider_name=provider_name,
                base_url=llm_base_url,
                api_key=llm_api_key,
            )

            result = rag.answer(
                question=question,
                history=st.session_state.messages[:-1],
                chat_client=chat_client,
                model=llm_model,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                top_k=int(retrieval_k),
            )

            answer = result["answer"]
            sources = result["sources"]
            chat_result = result["chat_result"]

            meta_caption = build_meta_caption(
                provider_name=provider_name,
                model_name=llm_model,
                chat_result=chat_result,
                source_count=len(sources),
            )

            st.markdown(answer)
            render_sources(sources)
            st.caption(meta_caption)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "meta_caption": meta_caption,
                }
            )

        except Exception as exc:
            logger.exception("Answer generation failed")
            error_text = f"Error: {exc}"
            st.error(error_text)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_text,
                    "sources": [],
                    "meta_caption": "",
                }
            )