# Immunisation Guidelines Adviser Agent — Presentation Outline
## IMAC × UoA AI Hackathon 2026 | 10 Minutes

---

## Overall Structure

| # | Slide Title | Duration | Speaker Focus |
|---|---|---|---|
| 1 | Title | 0:20 | — |
| 2 | The Problem | 1:10 | Business context |
| 3 | Our Solution & Design | 1:30 | Design decisions |
| 4 | Technical Architecture | 1:00 | System components |
| 5 | Live Demo | 2:30 | PoC demonstration |
| 6 | Evaluation Results | 1:00 | Metrics & evidence |
| 7 | AI Governance | 1:00 | Responsible AI |
| 8 | Feasibility & Next Steps | 1:00 | Path to production |
| 9 | Summary | 0:30 | Close |

**Total: 10 minutes**

---

## Slide 1 — Title

**Layout:** Centred, full-bleed dark blue background

**Content:**
- Main title: **Immunisation Guidelines Adviser Agent**
- Subtitle: *AI-powered retrieval support for clinical advisors*
- Team member names (all listed)
- IMAC × University of Auckland AI Hackathon 2026
- Date: June 2026

**Design notes:**
- IMAC logo top-right
- UoA logo bottom-right
- No bullet points — clean and professional

---

## Slide 2 — The Problem


**Title:** Clinical advisors spend too much time finding answers that already exist

**Left — Key points (4 bullets max):**
- IMAC operates NZ's national 0800-IMMUNE immunisation advice line
- Advisors answer high volumes of **repeated queries** on vaccines, schedules, and contraindications
- Each query requires manually searching across NZ Handbook, IMAC website, and Pharmac resources
- This reduces capacity for complex cases and introduces **variability in advice**

**Right — Quote box:**
> *"A large proportion of queries are repeated and the answers already exist in official New Zealand immunisation guidance."*
> — IMAC Project Brief, March 2026

**Speaker talking points:**
- Make clear this is a real operational pain point for IMAC
- Emphasise the clinical risk angle: inconsistent advice framing is a patient safety concern
- Transition: "We were asked to build a proof-of-concept that changes this"

---

## Slide 3 — Our Solution & Design

**Layout:** Three-column card layout

**Title:** A RAG-powered adviser agent — retrieve, summarise, cite

**Column 1 — What it does:**
- Takes free-text query from clinical advisor
- Retrieves relevant sections from approved NZ guidance
- Returns a concise, cited answer
- Flags clearly when no answer is found

**Column 2 — How we designed it:**
- **Aggregator approach** with Handbook as the overarching source of truth
- LLM synthesises a cited answer grounded in Handbook guidance
- IMAC website surfaced as supplementary context alongside the answer
- Confidence levels: high / medium / low / not found
- Design confirmed with Theo Brandt, IMAC (May 2026) ← *show screenshot*

**Column 3 — Core principles:**
- Accuracy over recall
- Source transparency at all times
- Clinical safety first
- Privacy by design


---

## Slide 4 — Technical Architecture



**Title:** From advisor query to cited answer in seconds

**Pipeline diagram (horizontal flow):**

```
[Advisor Query]
      ↓
[PII Input Filter]          → blocks NHI numbers, phone, DOB
      ↓
[Query Normaliser]          → 145 medical synonym expansion rules
      ↓
[Query Classifier]          → 6 dimensions: vaccine type, clinical scenario, urgency...
      ↓
┌──────────────────────────────────────────┐
│  Hybrid Retrieval Engine                 │
│  BM25 (keyword exact match)  ──┐         │
│                                ├─ Weighted RRF  →  Top 10 chunks  │
│  FAISS (semantic vector)     ──┘         │
└──────────────────┬───────────────────────┘
                   ↓
[GPT-4o-mini]               → citation-only system prompt (10 safety rules)
                   ↓
[PII Output Redaction]      → removes any PII in generated text
                   ↓
  {answer, citations[], confidence, sources}
```

**Right — Knowledge Base summary box:**
| Source 
|---|---|
| NZ Immunisation Handbook |
| IMAC Vaccines & Diseases | 
| PHARMAC Schedule Online | 
| WHO Immunization |


- Embedding model: `text-embedding-3-small` (1,536 dims)
- LLM: `GPT-4o-mini` via Azure AI Foundry
- Retrieval: Weighted RRF (FAISS 70% / BM25 30%, k=20)

**Speaker talking points:**
- Don't go deep on each component — one sentence per step
- Emphasise: every design choice serves the clinical safety goal
- "The system was built so that if you change any one component, the others remain auditable"

---

## Slide 5 — Live Demo

**Layout:** Three demo scenario boxes (pre-loaded, ready to run)

**Title:** Proof of Concept — live demonstration

**Demo Scenario 1 — Standard clinical query (45 seconds):**
> *"When should the MMR vaccine be given to a 12-month-old child in New Zealand?"*

Expected output to highlight:
- Concise answer based on NZ Handbook
- Citation: chapter name + section + URL
- Confidence: **high**

**Demo Scenario 2 — PII protection (30 seconds):**
> *"My patient's phone number is 021-xxx-xxxx and they have an egg allergy — can they receive the influenza vaccine?"*

Expected output to highlight:
- System blocks the query: HTTP 400
- Returns: `nz_phone` PII warning
- Advisor is prompted to remove personal details before re-submitting

**Demo Scenario 3 — Not found / escalation (45 seconds):**
> *"What documentation must be completed after administering a vaccine in New Zealand?"*

Expected output to highlight:
- System responds: *"I could not find a clear answer in the approved guidance"*
- Confidence: **not_found**
- Does NOT fabricate an answer
- Escalation instruction provided

**Fallback:** If live demo fails, show pre-recorded screenshots of each scenario

**Speaker talking points:**
- Scenario 1: "This is the typical use case — common questions answered in under 3 seconds"
- Scenario 2: "IMAC handles real patient data — PII protection is non-negotiable"
- Scenario 3: "The system knows what it doesn't know — accuracy over recall"

---

## Slide 6 — Evaluation Results

**Layout:** Central table + three metric highlight boxes below

**Title:** Evaluated against 70 clinical questions — including real advisor transcripts

**Ablation table:**

| Retrieval Method | Section Hit@10 | MRR |
|---|---:|---:|
| BM25-only (keyword) | 82.9% | 0.623 |
| Vector-only FAISS | 88.6% | 0.656 |
| **Hybrid BM25 + FAISS (ours)** | **88.6%** | **0.715** |

**Three highlight boxes:**

| 📊 70 Questions Evaluated | ✅ 100% Citation Coverage | 🚫 80% Escalation Accuracy |
|---|---|---|
| 40 hand-crafted + 30 from real IMAC transcripts | Every answered question has at least one cited source | System correctly refuses to answer 4 out of 5 "not found" queries |

**One-line interpretation:**
> Hybrid retrieval matches pure vector search on recall (88.6%) while ranking correct content significantly higher (MRR +9%), improving LLM generation quality.

**Speaker talking points:**
- MRR explains *why* hybrid matters: the correct content appears earlier in the top-10, so the LLM is more likely to use it
- The 70-question set includes questions derived from real IMAC call transcripts — this is not a synthetic benchmark
- Be honest: 4 questions remain unanswerable (knowledge base gaps, not model failures)

---

## Slide 7 — AI Governance

**Layout:** 5-row table (principle → implementation)

**Title:** Responsible AI: built in from day one, not added at the end

| Principle | What it means | Our implementation |
|---|---|---|
| **Clinical safety first** | Agent supports advisors, never replaces them | 10-rule system prompt; every answer closes: *"Final clinical decisions remain with the qualified advisor"* |
| **Source transparency** | Every claim must be traceable to approved guidance | Citations include: document name + chapter + section + URL + verbatim excerpt |
| **Accuracy over recall** | Better to say "not found" than to confabulate | `not_found` confidence level + escalation response when evidence is insufficient |
| **Privacy by design** | No PII in inputs, outputs, or logs | Input filter (NHI, phone, DOB, email) + output redaction + no PII stored in audit log |
| **Auditability** | All interactions must be reviewable | SQLite audit log: query, retrieved chunks, citations, generated answer — per session |

**Speaker talking points:**
- These five principles come directly from the IMAC brief — show you read and acted on it
- Emphasise that privacy and auditability are *system features*, not policies on paper
- "If something goes wrong with an answer, we can trace exactly which chunks were retrieved and why"

---

## Slide 8 — Feasibility & Next Steps

**Layout:** Left column (what we built) + Right column (path to production) + bottom timeline

**Title:** From proof of concept to production — a realistic pathway

**Left — PoC achievements:**
- ✅ Working RAG agent with FastAPI + LangGraph
- ✅ 5 approved sources
- ✅ PII protection (input + output)
- ✅ Audit logging for every interaction
- ✅ Query classification across 6 clinical dimensions
- ✅ Evaluated on 70 questions including real transcripts

**Right — What production requires:**
- ☁️ Deploy to Azure (already using Azure AI Foundry — natural next step)
- 👩‍⚕️ Manual clinical review: domain expert sign-off on answer quality sample
- 🔁 Feedback loop: flag incorrect answers, update knowledge base on guidance changes  #还可以再加一些



**Data requirements for production:**
- Continuously updated NZ Handbook (versioned, not static)
- Full Medsafe CMI/datasheet corpus
- Ongoing transcript data for evaluation refresh
- IMAC-reviewed ground-truth Q&A for model monitoring

**Speaker talking points:**
- The Azure infrastructure is already in use (LLM and embedding are on Azure) — deployment is incremental, not a rewrite
- The biggest remaining gap is Medsafe data and human clinical review
- Transcript gap analysis (stretch goal) would directly inform which knowledge base sections to prioritise

---

## Slide 9 — Summary

**Layout:** Three-column highlight + closing quote

**Title:** Faster, consistent, cited guidance — keeping the advisor in control

**Three columns:**

| The Problem | Our Solution | The Results |
|---|---|---|
| Clinical advisors spend significant time searching for answers that already exist in approved guidance | RAG agent: hybrid retrieval + citation-only LLM + safety guardrails across 3,633 chunks from 5 sources | 88.6% retrieval accuracy · 100% citation coverage · 80% escalation accuracy · MRR 0.715 |

**Closing quote (large, centred):**
> *"The final clinical decision always remains with the qualified advisor."*

**Bottom line:**
- Thank IMAC and Theo Brandt for the project brief and guidance
- Open for questions

---



## Speaker Assignment Template

| Slide | Suggested Speaker | Key message to land |
|---|---|---|
| 1 | All (brief intro) | Who we are |
| 2 | Member A | This is a real, unsolved operational problem |
| 3 | Member B | We made deliberate design choices with the client |
| 4 | Member C | The system is well-engineered and auditable |
| 5 | Member D (+ support) | The PoC works — watch it live |
| 6 | Member C | We measured our performance rigorously |
| 7 | Member B | Governance is built in, not an afterthought |
| 8 | Member A | This can realistically reach production |
| 9 | All | Thank you |

*Adjust assignments based on team size. Every member must speak.*
