# Immunisation Guidelines Adviser

A RAG-based AI agent that answers clinical questions about New Zealand immunisation guidelines, powered by Azure OpenAI and hybrid BM25 + FAISS retrieval.

Built for the University of Auckland COMPSCI 714 Hackathon in partnership with Microsoft and the UoA Research and Innovation Office.

---

## Architecture

```
User Query
    │
    ▼ PII scan (block if NHI / phone / email / DOB detected)
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
│  classify → retrieve → generate → format│
│               │            │            │
│         BM25 + FAISS    gpt-4o-mini     │
│         RRF fusion      answer +        │
│         (3,633 chunks)  citations +     │
│                         confidence      │
└─────────────────────────────────────────┘
                    │
                    ▼ Output PII redaction
                    │
                    ▼ Audit log (query · sources · citations · answer)
```

**Retrieval** — Hybrid BM25 (keyword) + FAISS semantic search fused with Reciprocal Rank Fusion (RRF).
- FAISS `IndexFlatIP` on L2-normalised `text-embedding-3-small` vectors (1,536 dim) = exact cosine similarity with C++/SIMD speed
- BM25 handles verbatim medical terms (MMR, BCG, HBsAg, Flucelvax, MMRV)
- RRF combines ranked lists by reciprocal rank, avoiding BM25/cosine score-scale mismatch

**Generation** — `gpt-4o-mini` with a clinical-safety system prompt; returns structured JSON with answer, citations, and confidence level (`high / medium / low / not_found`).

**Responsible AI** — Two-layer PII filter (input block + output redaction), `not_found` escalation enforced at API layer, full audit trail per query.

---

## Project Structure

```
immunisation-adviser/
├── agent/
│   ├── query_handler.py      # LangGraph pipeline (classify → retrieve → generate → format)
│   ├── classifier.py         # Rule-based query classification across 6 dimensions
│   ├── retriever.py          # BM25 + FAISS IndexFlatIP + RRF hybrid retrieval
│   ├── generator.py          # gpt-4o-mini answer generation (JSON output)
│   └── output_formatter.py   # Structured output + markdown formatting
├── api/
│   ├── main.py               # FastAPI app (REST + SSE streaming)
│   ├── auth_manager.py       # SQLite user auth
│   ├── jwt_utils.py          # JWT token helpers
│   ├── audit_logger.py       # SQLite audit log (query · sources · citations · answer)
│   └── pii_filter.py         # PII detection (input block) + output redaction
├── config/
│   └── azure_config.py       # Provider config (Azure OpenAI / standard OpenAI)
├── ingestion/
│   ├── csv_to_chunks.py      # Convert CSV → JSON chunks with 200-char section overlap
│   ├── embed_and_index.py    # Generate text-embedding-3-small vectors → local JSON or Azure Search
│   ├── analyze_transcripts.py # Process 1,797 Contact Lens transcripts → topic stats + eval questions
│   └── audit_pii_redaction.py # Verify PII redaction across all transcript files (Milestone 1)
├── ui/
│   ├── advisor_interface.py  # Gradio UI (local demo)
│   └── static/index.html     # Bootstrap 5 UI (served by FastAPI at /)
├── evaluation/
│   ├── question_set.json              # 30 hand-crafted clinical questions
│   ├── transcript_question_set.json   # 30 questions derived from real IMAC call transcripts
│   └── run_eval.py                    # Batch evaluation script
├── data/
│   └── immunisation_rag_chunks(1).csv # Source CSV (3,633 chunks, 5 sources)
│   # chunks_raw.json and chunks_with_embeddings.json are gitignored (generated locally)
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

Edit `.env` and fill in your credentials.

**Option A — Azure OpenAI (recommended for this project)**
```env
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT_GPT4O=gpt-4o-mini
AZURE_OPENAI_DEPLOYMENT_EMBEDDING=text-embedding-3-small
```

**Option B — Standard OpenAI**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

### 3. Build the knowledge base (first time only)

```bash
# Step 1: Convert CSV → JSON chunks with section-aligned 200-char overlap
python -m ingestion.csv_to_chunks

# Step 2: Generate embeddings (text-embedding-3-small, calls Azure OpenAI, ~5 min)
python -m ingestion.embed_and_index data/chunks_raw.json --local
```

This creates `data/chunks_with_embeddings.json` (~117 MB, 3,633 chunks with 1,536-dim vectors).

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
| POST | `/query` | JWT | Full RAG query (returns answer + citations + confidence + classification) |
| POST | `/query/stream` | JWT | Streaming SSE response |
| GET | `/history` | JWT | Last N queries for current user |
| DELETE | `/history` | JWT | Clear query history |
| GET | `/reports/summary` | JWT | Aggregate statistics (confidence, vaccine type, query type, daily volume) |
| GET | `/health` | — | Health check |

Interactive API docs: **http://127.0.0.1:8000/docs**

---

## Knowledge Base

| Source | Chunks | Role |
|--------|--------|------|
| NZ Immunisation Handbook (Te Whatu Ora) | 2,275 | Primary approved NZ source |
| IMAC Vaccines & Diseases (immune.org.nz) | 756 | Approved NZ source |
| Immunisation Advisory Centre (IMAC) | 227 | Approved NZ source |
| PHARMAC Schedule Online | 106 | Approved NZ source |
| WHO Immunization guidance | 269 | Supplementary international reference |
| **Total** | **3,633** | |

> Medsafe product information is listed as an approved source in the project brief but is absent from the current dataset — identified as a next development step.

---

## Chunking Strategy

The source CSV is pre-chunked at handbook section level. To preserve context at boundaries, the last **200 characters** of each chunk are prepended to the next chunk within the same section.

- **3,633** total chunks
- **1,193** chunks with 200-char overlap applied
- Boundaries aligned to handbook section structure
- Vectors: `text-embedding-3-small`, 1,536 dimensions, L2-normalised

---

## Call Transcript Analysis

1,797 anonymised Amazon Connect Contact Lens transcripts (April 2026) were processed to understand real IMAC advisor–caller interaction patterns.

- **Milestone 1 verified**: Contact Lens PII redaction active on 100% of files. Zero NHI numbers, names, emails, or addresses found. One residual phone number caught by post-hoc filter.
- Top query types: eligibility/funding (378), dosage (197), catch-up schedules (115)
- Top vaccines: influenza (359), COVID-19 (159), varicella (115)
- Volume: weekday business hours only (08:00–16:00 NZ), peak Thursday

Transcripts are used **for analysis and evaluation only** — they are not included in the retrieval knowledge base.

---

## Evaluation

Run the full 60-question evaluation set:

```bash
python -m evaluation.run_eval
```

The evaluation set comprises:
- **30 hand-crafted questions** (`evaluation/question_set.json`) — covering all major vaccine types and query categories
- **30 real-call questions** (`evaluation/transcript_question_set.json`) — derived from real IMAC call transcripts

Results saved to `evaluation/results.json` with confidence scores and citations for manual review.

---

## Responsible AI

| Requirement | Implementation |
|---|---|
| Clinical safety | System prompt prohibits diagnosis and treatment recommendations; all answers close with advisor disclaimer |
| Source transparency | Every answer includes citations; no-citation answers trigger automatic warning |
| Accuracy over recall | `not_found` confidence → API replaces answer with fixed escalation message; `low` confidence → ⚠ warning appended |
| Privacy by design | Input PII scan blocks NHI/phone/email/DOB before reaching model; output PII redaction applied to all responses |
| Auditability | SQLite audit log records query, confidence, sources retrieved, citations used, and answer text for post-hoc review |
| No live system connection | Local SQLite + local FAISS + Azure OpenAI API only; no connection to IMAC phone system or patient records |

---

## Data Usage Policy

> Transcript content is used solely for topic analysis and evaluation question extraction. It is not included in the RAG retrieval index. All answers are grounded exclusively in approved NZ immunisation guidance.
>
> This tool is for clinical advisor support only. Final clinical decisions remain with qualified healthcare staff.
