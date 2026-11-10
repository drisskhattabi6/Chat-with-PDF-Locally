from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from schemas import ChunkRecord, ExtractedDocument


CHUNKING_LABELS = {
    "recursive": "Recursive character",
    "markdown_recursive": "Markdown headers + recursive",
    "token": "Token",
    "page_recursive": "Page-aware recursive",
}


def _bounded_overlap(chunk_size: int, overlap: int) -> int:
    overlap = max(0, overlap)
    return min(overlap, max(0, chunk_size - 1))


def chunk_document(
    extracted: ExtractedDocument,
    method: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    chunk_size = max(100, int(chunk_size))
    chunk_overlap = _bounded_overlap(chunk_size, int(chunk_overlap))

    if method == "page_recursive":
        return _chunk_page_recursive(extracted, chunk_size, chunk_overlap)

    if method == "markdown_recursive":
        return _chunk_markdown_recursive(extracted, chunk_size, chunk_overlap)

    if method == "token":
        return _chunk_token(extracted, chunk_size, chunk_overlap)

    if method == "recursive":
        return _chunk_recursive(extracted, chunk_size, chunk_overlap)

    raise ValueError(f"Unsupported chunking method: {method}")


def _chunk_recursive(extracted: ExtractedDocument, chunk_size: int, chunk_overlap: int) -> list[ChunkRecord]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    texts = splitter.split_text(extracted.markdown)

    chunks: list[ChunkRecord] = []
    for idx, text in enumerate(texts):
        text = text.strip()
        if not text:
            continue
        chunks.append(
            ChunkRecord(
                id=f"{extracted.doc_id}_recursive_{idx}",
                text=text,
                metadata={
                    "source_name": extracted.source_name,
                    "extractor": extracted.extractor_name,
                    "chunking_method": "recursive",
                    "chunk_index": idx,
                },
            )
        )
    return chunks


def _chunk_token(extracted: ExtractedDocument, chunk_size: int, chunk_overlap: int) -> list[ChunkRecord]:
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = splitter.split_text(extracted.markdown)

    chunks: list[ChunkRecord] = []
    for idx, text in enumerate(texts):
        text = text.strip()
        if not text:
            continue
        chunks.append(
            ChunkRecord(
                id=f"{extracted.doc_id}_token_{idx}",
                text=text,
                metadata={
                    "source_name": extracted.source_name,
                    "extractor": extracted.extractor_name,
                    "chunking_method": "token",
                    "chunk_index": idx,
                },
            )
        )
    return chunks


def _chunk_markdown_recursive(
    extracted: ExtractedDocument,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ],
        strip_headers=False,
        return_each_line=False,
    )
    sections = header_splitter.split_text(extracted.markdown)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[ChunkRecord] = []
    chunk_counter = 0

    if not sections:
        return _chunk_recursive(extracted, chunk_size, chunk_overlap)

    for section_idx, section in enumerate(sections):
        subtexts = recursive_splitter.split_text(section.page_content)
        for sub_idx, text in enumerate(subtexts):
            text = text.strip()
            if not text:
                continue

            metadata = {
                "source_name": extracted.source_name,
                "extractor": extracted.extractor_name,
                "chunking_method": "markdown_recursive",
                "section_index": section_idx,
                "subchunk_index": sub_idx,
                "chunk_index": chunk_counter,
                **section.metadata,
            }

            chunks.append(
                ChunkRecord(
                    id=f"{extracted.doc_id}_md_{chunk_counter}",
                    text=text,
                    metadata=metadata,
                )
            )
            chunk_counter += 1

    return chunks


def _chunk_page_recursive(
    extracted: ExtractedDocument,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    if not extracted.page_chunks:
        return _chunk_recursive(extracted, chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[ChunkRecord] = []
    chunk_counter = 0

    for page_idx, page in enumerate(extracted.page_chunks):
        page_text = (page.get("text") or "").strip()
        if not page_text:
            continue

        page_number = page.get("page", page_idx + 1)
        subtexts = splitter.split_text(page_text)

        for sub_idx, text in enumerate(subtexts):
            text = text.strip()
            if not text:
                continue

            chunks.append(
                ChunkRecord(
                    id=f"{extracted.doc_id}_page_{page_number}_{sub_idx}",
                    text=text,
                    metadata={
                        "source_name": extracted.source_name,
                        "extractor": extracted.extractor_name,
                        "chunking_method": "page_recursive",
                        "page": page_number,
                        "page_subchunk_index": sub_idx,
                        "chunk_index": chunk_counter,
                    },
                )
            )
            chunk_counter += 1

    return chunks