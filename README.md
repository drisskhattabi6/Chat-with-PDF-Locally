
# ChatPDF: Chat with PDF Locally 🤖

## Overview

**ChatPDF** is a local RAG (Retrieval-Augmented Generation) application that lets you upload a PDF, index its content, and chat with it through a simple Streamlit interface.

This updated version is more modular and more robust than the original one:

- supports **multiple PDF extraction engines**
- supports **multiple LLM providers**
- lets the user **choose the chunking strategy**
- uses **persistent local vector search with ChromaDB**
- shows **retrieved sources** used to answer questions
- caches indexed PDFs to avoid unnecessary reprocessing

The app is designed for local-first usage, but it can also connect to hosted providers through API keys.

![](./imgs/ChatPDF3.png)

---

## Demo

![Tutorial GIF](./imgs/tutorial.gif)

<video src="./imgs/video.mp4" autoplay controls loop controlsList="nodownload" style="max-width: 100%; height: auto;">
  Your browser does not support the video tag.
</video>

---

## What's New in This Version

Compared to the old version, this refactor includes:

- **Marker removed**
  - replaced with **Docling** as the balanced / medium PDF extractor
- **Fast extractor kept**
  - **PyMuPDF4LLM** is used as the fast / lightweight extractor
- **More LLM providers**
  - Ollama
  - OpenAI
  - Groq
  - OpenRouter
  - SambaNova
- **Selectable chunking methods**
  - Recursive character
  - Markdown headers + recursive
  - Token-based
  - Page-aware recursive
- **Better indexing**
  - PDF hash-based caching
  - collection naming based on document + configuration
- **Cleaner architecture**
  - extraction, chunking, embeddings, providers, vector store, and RAG logic are split into separate files
- **Source-aware answers**
  - the app shows retrieved chunks used to build the answer

---

## Technology Stack

- [Streamlit](https://streamlit.io/)
- [ChromaDB](https://www.trychroma.com/)
- [PyMuPDF4LLM](https://pymupdf.readthedocs.io/)
- [Docling](https://docling-project.github.io/docling/)
- [SentenceTransformers](https://www.sbert.net/)
- [Ollama](https://ollama.com/)
- Python

---

## Features

- **PDF Extraction**
  - Fast extraction with **PyMuPDF4LLM**
  - Balanced extraction with **Docling**
- **Flexible Chunking**
  - choose the chunking method that fits your document
- **Multiple LLM Providers**
  - local or hosted models
- **Local Vector Search**
  - embeddings stored in a persistent ChromaDB database
- **Hybrid Retrieval**
  - vector retrieval + BM25 reranking
- **Source Display**
  - answers include the retrieved source chunks
- **Chat Export**
  - download the conversation as a PDF
- **Caching**
  - already indexed PDFs can be loaded again without reprocessing

---

## Supported Providers

### LLM Providers
- **Ollama** (local)
- **OpenAI**
- **Groq**
- **OpenRouter**
- **SambaNova**

### Embedding Providers
- **SentenceTransformers** (local)
- **Ollama embeddings**

---

## Project Structure

```bash
.
├── app.py                  # Streamlit app
├── config.py               # App settings and environment loading
├── schemas.py              # Shared data structures
├── utils.py                # Helpers and utility functions
├── pdf_extractors.py       # PDF extraction engines
├── chunking.py             # Chunking methods
├── embeddings.py           # Embedding providers
├── vector_store.py         # ChromaDB wrapper
├── providers.py            # LLM provider clients
├── rag.py                  # RAG pipeline and reranking
├── requirements.txt        # Dependencies
├── .env.example            # Example environment variables
├── imgs/                   # Screenshots, logo, demo media
├── data/                   # Auto-generated app data
└── README.md
```

---

## How It Works

1. **Upload a PDF**
   - the app saves the file locally
   - computes a file hash
   - checks whether the same document + same config was already indexed

2. **Choose a PDF extractor**
   - **PyMuPDF4LLM**: faster and lighter
   - **Docling**: more balanced and often better for structured PDFs

3. **Choose a chunking method**
   - Recursive character
   - Markdown headers + recursive
   - Token
   - Page-aware recursive

4. **Choose an embedding provider**
   - SentenceTransformers
   - Ollama embeddings

5. **Index the document**
   - extracted text is chunked
   - embeddings are generated
   - chunks are stored in ChromaDB

6. **Ask questions**
   - relevant chunks are retrieved
   - reranked using vector similarity + BM25
   - the selected LLM answers using only the retrieved context

7. **Inspect sources**
   - the app displays the source chunks used to generate the answer

---

## Architecture

### High-level pipeline

```text
PDF Upload
   ↓
PDF Extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
ChromaDB Index
   ↓
Retrieval + Reranking
   ↓
LLM Answer Generation
   ↓
Answer + Sources
```

You can still keep your existing screenshots and diagrams:

![](./imgs/Pre-processing.png)

![](./imgs/img6.png)

---

## Requirements

- **Python 3.10+**
- `pip`
- for local LLM usage: **Ollama**
- for cloud providers: the relevant **API key**
- enough RAM/CPU depending on model size and PDF complexity

> Note: Python 3.10+ is recommended for compatibility with the current refactored stack.

---

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/drisskhattabi6/Chat-with-PDF-Locally.git
cd Chat-with-PDF-Locally
```

### 2) Create and activate a virtual environment

#### Linux / macOS
```bash
python -m venv .venv
source .venv/bin/activate
```

#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Create your environment file

Copy `.env.example` to `.env` and update it if needed.

Example:

```bash
APP_TITLE="Chat with PDF"
APP_PUBLIC_URL="http://localhost:8501"

OLLAMA_BASE_URL="http://localhost:11434"

OPENAI_API_KEY=""
GROQ_API_KEY=""
OPENROUTER_API_KEY=""
SAMBANOVA_API_KEY=""

DEFAULT_CHUNK_SIZE="900"
DEFAULT_CHUNK_OVERLAP="120"
DEFAULT_RETRIEVAL_K="6"

EMBEDDING_PROVIDER="sentence_transformers"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

---

## Optional: Install Ollama

If you want to use **local LLMs** or **Ollama embeddings**, install Ollama first.

Then pull the models you want, for example:

```bash
ollama pull qwen2.5:latest
ollama pull nomic-embed-text:latest
```

You can use any Ollama chat model you prefer.

---

## Run the App

```bash
streamlit run app.py
```

Then open:

```bash
http://localhost:8501
```

![](./imgs/App.png)

---

## How to Use

1. **Upload a PDF**
2. **Select a PDF extractor**
   - Fast / Small: PyMuPDF4LLM
   - Balanced / Medium: Docling
3. **Select a chunking method**
4. **Select the embedding provider**
5. **Click `Process / Load index`**
6. **Select the LLM provider**
7. **Choose or enter the model name**
8. **Ask questions about the PDF**
9. **Inspect the retrieved sources**
10. **Download the chat as PDF if needed**

---

## Chunking Methods

The app lets you choose how the extracted text is split before indexing.

### 1. Recursive character
A general-purpose splitter based on text separators.

Best for:
- most standard PDFs
- general usage

### 2. Markdown headers + recursive
First splits by markdown headers, then recursively splits large sections.

Best for:
- structured reports
- manuals
- research documents

### 3. Token
Splits text by token count instead of character count.

Best for:
- model-aware chunk sizing
- tighter chunk budget control

### 4. Page-aware recursive
Splits text page by page, then recursively chunks each page.

Best for:
- page-based citation workflows
- preserving page locality

> Page-aware chunking works best when the selected extractor returns page-level content.

---

## PDF Extractors

### PyMuPDF4LLM
- faster
- lightweight
- useful for searchable PDFs
- supports page-aware extraction

### Docling
- more balanced than a very heavy OCR-style pipeline
- good for structured and moderately complex PDFs
- better suited than the old large extraction approach for this app version

---

## Screenshots

### Sidebar

![](./imgs/sidebar.png)

### PDF Processing

![](./imgs/pdf_processing.png)

### Providers

#### Ollama
![](./imgs/provider1.png)
![](./imgs/provider1-1.png)

#### SambaNova
![](./imgs/provider2.png)
![](./imgs/provider2-2.png)

### Chat Interface

![](./imgs/img3.png)

![](./imgs/chat1.png)

---

## Notes About the Refactor

This version removes the old files:

- `llm_inference.py`
- `md_convertor.py`

Their responsibilities are now handled by:

- `providers.py`
- `pdf_extractors.py`

This makes the app easier to extend and maintain.

---

## Troubleshooting

### 1) No Ollama models found
Make sure Ollama is installed and running:

```bash
ollama list
```

If needed, pull a model:

```bash
ollama pull qwen2.5:latest
```

### 2) Embedding model fails
If you use Ollama embeddings, make sure the embedding model is installed:

```bash
ollama pull nomic-embed-text:latest
```

If you use SentenceTransformers, verify the model name is valid.

### 3) API provider does not respond
Check:
- API key is correct
- base URL is correct
- selected model exists for that provider

### 4) PDF processing returns empty text
This can happen if:
- the PDF is image-only
- the file is corrupted
- the selected extractor is not suitable for that document

Try:
- switching extractor
- using Docling instead of the fast extractor
- testing with another PDF

### 5) Index loads but answers are weak
Try:
- increasing retrieval `k`
- changing the chunking strategy
- using smaller or larger chunk size
- switching extractor for better structure

### 6) Streamlit app is slow
Possible reasons:
- very large PDF
- large embedding model
- slow local machine
- remote provider latency

### 7) Provider model list fails
Some hosted providers may require a valid API key before model listing works.

---

## Known Limitations

- performance depends on your hardware and chosen models
- very large PDFs may take time to process
- extraction quality depends on PDF structure
- hosted provider quality depends on selected model
- some chunking methods are better suited for specific document types than others

---

## Future Improvements

Possible next improvements:

- multi-PDF collections
- citations with exact page rendering
- chat history persistence
- Docker support
- model presets
- OCR fallback pipeline for image-only PDFs
- evaluation dashboard for chunking and retrieval quality

---

## Author

If you want, you can keep your original contact section here:

[LinkedIn](https://www.linkedin.com/in/idriss-khattabi-b3a266235/)
