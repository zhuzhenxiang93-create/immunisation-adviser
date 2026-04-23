# Immunisation Guidelines Adviser

A RAG-based AI agent that answers clinical questions about New Zealand immunisation guidelines, powered by Azure OpenAI and hybrid BM25 + vector retrieval.

Built for the University of Auckland COMPSCI 714 Hackathon in partnership with Microsoft and the UoA Research and Innovation Office.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           FastAPI Backend               │
│  POST /query  ·  POST /query/stream     │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         LangGraph RAG Pipeline          │
│                                         │
│  retrieve_node → generate_node          │
│       │               │                 │
│  BM25 + Vector     gpt-4o               │
│  RRF fusion        answer +             │
│  (3 129 chunks)    citations            │
└─────────────────────────────────────────┘
```

**Retrieval** — Hybrid BM25 (keyword) + cosine vector search fused with Reciprocal Rank Fusion (RRF). Each chunk preserves its handbook section boundary as overlap context.

**Generation** — `gpt-4o` with a clinical-safety system prompt; returns structured JSON with answer, citations, and confidence level (`high / medium / low / not_found`).

---

## Project Structure

```
immunisation-adviser/
├── agent/
│   ├── query_handler.py      # LangGraph pipeline (retrieve → generate → format)
│   ├── retriever.py          # BM25 + vector + RRF hybrid retrieval
│   ├── generator.py          # GPT-4o answer generation
│   └── output_formatter.py   # Structured output + markdown formatting
├── api/
│   ├── main.py               # FastAPI app (REST + SSE streaming)
│   ├── auth_manager.py       # SQLite user auth
│   └── jwt_utils.py          # JWT token helpers
├── config/
│   └── azure_config.py       # Provider config (Azure / OpenAI / Qwen)
├── ingestion/
│   ├── csv_to_chunks.py      # Convert CSV → JSON chunks with overlap
│   ├── embed_and_index.py    # Generate embeddings → local JSON or Azure Search
│   ├── chunk_documents.py    # Text splitter for raw documents
│   └── scrape_handbook.py    # Optional: scrape handbook from web
├── ui/
│   ├── advisor_interface.py  # Gradio UI
│   └── static/index.html     # Bootstrap UI (served by FastAPI)
├── evaluation/
│   ├── question_set.json     # 6 curated test questions
│   └── run_eval.py           # Batch evaluation script
├── data/
│   ├── immunisation_rag_chunks(1).csv   # Source data (3 129 chunks)
│   └── chunks_raw.json                  # Processed chunks (no embeddings)
├── start.py                  # One-command launcher
├── test_agent.py             # Quick smoke test
└── .env.example              # Environment variable template
```

---

## Quick Start

### 1. Clone & create environment

```bash
git clone https://github.com/zhuzhenxiang93-create/immunisation-adviser.git
cd immunisation-adviser

conda create -n immunisation-adviser python=3.11
conda activate immunisation-adviser
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials. Choose one provider:

**Option A — Azure OpenAI (recommended)**
```env
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT_GPT4O=gpt-4o
AZURE_OPENAI_DEPLOYMENT_EMBEDDING=text-embedding-3-large
```

**Option B — Standard OpenAI**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

**Option C — Qwen / DashScope**
```env
LLM_PROVIDER=qwen
DASHSCOPE_API_KEY=sk-...
```

### 3. Build the knowledge base (first time only)

Convert the source CSV to chunks and generate embeddings:

```bash
# Step 1: Convert CSV → JSON with section-aligned overlap
python -m ingestion.csv_to_chunks

# Step 2: Generate embeddings (calls Azure OpenAI, takes ~5 min)
python -m ingestion.embed_and_index data/chunks_raw.json --local
```

This creates `data/chunks_with_embeddings.json` (3 129 chunks, ~500 MB).

### 4. Start the application

```bash
python start.py
```

Opens the Bootstrap web UI at **http://127.0.0.1:8000** automatically.

Or run the Gradio UI instead:

```bash
python -m ui.advisor_interface
# → http://127.0.0.1:7860
```

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/auth/register` | — | Register and get JWT |
| POST | `/auth/login` | — | Login and get JWT |
| GET | `/auth/me` | JWT | Current user info |
| POST | `/query` | JWT | Full RAG query |
| POST | `/query/stream` | JWT | Streaming SSE response |
| GET | `/history` | JWT | Last N queries |
| DELETE | `/history` | JWT | Clear session history |
| GET | `/health` | — | Health check |

Interactive API docs: **http://127.0.0.1:8000/docs**

---

## Chunking Strategy

The source CSV is pre-chunked by handbook section (~814 chars average, ~200 tokens). To preserve context at section boundaries, the last **200 characters** of each chunk are prepended to the next chunk within the same section (sliding overlap).

- 3 129 total chunks
- 1 544 chunks with overlap applied
- Chunk boundaries aligned to handbook section structure

---

## Evaluation

Run the built-in question set (6 clinical queries):

```bash
python -m evaluation.run_eval
```

Results are saved to `evaluation/results.json` with confidence scores and citations for manual review.

---

## Data Source

New Zealand Immunisation Handbook (Te Whatu Ora / Health New Zealand)  
https://www.tewhatuora.govt.nz/for-health-professionals/clinical-guidance/immunisation-handbook

> This tool is for clinical advisor support only. Final clinical decisions remain with qualified healthcare staff.
