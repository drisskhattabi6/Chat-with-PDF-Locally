from pathlib import Path

from schemas import ExtractedDocument


EXTRACTOR_LABELS = {
    "pymupdf4llm": "Fast / Small - PyMuPDF4LLM",
    "docling": "Balanced / Medium - Docling",
}


def extract_with_pymupdf4llm(pdf_path: Path, doc_id: str) -> ExtractedDocument:
    import pymupdf4llm

    page_dicts = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        header=False,
        footer=False,
    )

    page_chunks: list[dict] = []
    markdown_parts: list[str] = []

    for idx, page in enumerate(page_dicts or []):
        text = (page.get("text") or "").strip()
        metadata = page.get("metadata", {}) or {}
        page_number = metadata.get("page") or metadata.get("page_number") or (idx + 1)

        if text:
            markdown_parts.append(text)
            page_chunks.append(
                {
                    "page": page_number,
                    "text": text,
                    "metadata": metadata,
                }
            )

    markdown = "\n\n".join(markdown_parts).strip()

    return ExtractedDocument(
        doc_id=doc_id,
        source_name=pdf_path.name,
        extractor_name="pymupdf4llm",
        markdown=markdown,
        page_chunks=page_chunks,
        metadata={"page_count": len(page_chunks)},
    )


def extract_with_docling(pdf_path: Path, doc_id: str) -> ExtractedDocument:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document
    markdown = doc.export_to_markdown().strip()

    return ExtractedDocument(
        doc_id=doc_id,
        source_name=pdf_path.name,
        extractor_name="docling",
        markdown=markdown,
        page_chunks=[],
        metadata={},
        docling_document=doc,
    )


def extract_pdf(pdf_path: Path, extractor_key: str, doc_id: str) -> ExtractedDocument:
    if extractor_key == "pymupdf4llm":
        return extract_with_pymupdf4llm(pdf_path=pdf_path, doc_id=doc_id)

    if extractor_key == "docling":
        return extract_with_docling(pdf_path=pdf_path, doc_id=doc_id)

    raise ValueError(f"Unsupported extractor: {extractor_key}")