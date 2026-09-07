# DealGuard: Evidence-First Fact Reconciliation for IPO Readiness

> **Candidate:** Samarth Singh  
> **Repository:** `noviciusss/FactLayer_SuperJoin`  
> **Submission:** Superjoin Engineering Assignment  

---

## 1. Problem Statement

Financial analysts preparing IPO prospectuses and S-1 filings manage metrics scattered across annual reports, draft prospectuses, quarterly investor presentations, statutory disclosures, and macroeconomic releases. The same financial metric frequently differs due to reporting timeframe (`Q4 FY24` vs `FY24`), consolidation scope (`standalone` vs `consolidated`), accounting convention (`reported` vs `adjusted` EBITDA), pro forma acquisition restatements (SpotOn acquisition), or dropped table headers during PDF parsing. Standard LLM pipelines blindly compare numbers based on semantic similarity, converting missing reporting context into confident, hallucinated financial contradictions. In financial due diligence, **a false contradiction is worse than an abstention**. DealGuard extracts evidence-grounded facts from dense documents, reconciles them across filings, and refuses unsafe comparisons when the reporting basis is incomplete.

---

## 2. Why DealGuard is Different: The Ambiguity Firewall

> **"A numerical difference is not automatically a contradiction; it first has to be the same claim."**

DealGuard introduces an **Ambiguity Firewall** and an auditable **Decision Ledger** featuring `NEEDS_REVIEW` as a first-class verdict:

1. **Refusal to Guess:** If two numerical facts share an entity and attribute but lack reporting period or scope, the Ambiguity Firewall halts comparison *before* invoking an LLM, flagging `missing_period` or `missing_scope_or_basis`.
2. **Rules-First, LLM-Second:** Deterministic checks resolve exact matches, decimal rounding tolerances (`Decimal` arithmetic with 0.5% relative / 1.0 absolute tolerance), and disjoint pre-filters deterministically. This achieves a **75% LLM Avoidance Rate**, protecting free-tier token quotas and eliminating hallucinations for obvious pairs.
3. **100% Domain-Agnostic Purity:** DealGuard enforces zero document- or company-specific heuristics. There are no hardcoded conditionals (`if delhivery`, `if EBITDA`, or `if GDP`); all comparisons rely strictly on structured metadata envelopes and general token overlap.
4. **Field-by-Field Decision Ledger:** Every relationship stores a frozen comparison snapshot (`entity`, `attribute`, `period`, `scope`, `unit`, `value`) recording whether the conclusion was reached via deterministic rules, the ambiguity firewall, or contextual LLM adjudication.

---

## 3. System Architecture

```
                             [ Multi-Page PDF Document ]
                                          │
                                          ▼
                             [ Stage A: Ingest & Chunk ]
                       • Table boundary preservation (pdfplumber)
                       • Infographic slide / visual block extraction
                       • High-res page preview rendering (PyMuPDF)
                                          │
                                          ▼
                            [ Stage B: Fact Extraction ]
                       • Open-vocabulary extraction (Groq gpt-oss-20b)
                       • Extraction envelope: entity, attribute, value,
                         unit, period, scope, qualifiers, evidence_quote
                                          │
                                          ▼
                           [ Stage B.2: Grounding Self-Check ]
                       • Verifies verbatim evidence_quote in source text
                       • Suppresses hallucinations & legal boilerplate
                       • Content-hash deduplication (idempotent re-runs)
                                          │
                                          ▼
                         [ Stage C: Canonicalize & Embed ]
                       • Statement: "Entity | Attribute | Period | Value"
                       • Local FastEmbed (BAAI/bge-small-en-v1.5)
                                          │
                                          ▼
                         [ Stage D: Candidate Retrieval ]
                       • Qdrant dense vector similarity (cosine > 0.76)
                       • Incremental: new facts match against existing corpus
                                          │
                                          ▼
                      [ Stage E: Ambiguity Firewall & Cascade ]
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ 1. Disjoint Pre-Filter        ──► UNRELATED (irrelevant_pre_filter)        │
   │ 2. Deterministic Rule         ──► CORROBORATES (exact / rounding tolerance) │
   │ 3. Ambiguity Firewall:                                                      │
   │    • Missing Period           ──► NEEDS_REVIEW (missing_period)             │
   │    • Material Diff + No Scope ──► NEEDS_REVIEW (missing_scope_or_basis)      │
   │    • Unit Scale Ambiguity     ──► NEEDS_REVIEW (unit_ambiguity)             │
   │    • Table Context Uncertain  ──► NEEDS_REVIEW (header_context_uncertain)  │
   │ 4. Contextual LLM Adjudication──► RECONCILED / CONTRADICTS                  │
   │ 5. Error / Timeout Fallback   ──► NEEDS_REVIEW (adjudication_unavailable)   │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
                                          ▼
                         [ Relational SQLite / PostgreSQL ]
                      (Documents, Chunks, Facts, Relationships)
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
                    ▼                                           ▼
          [ FastAPI Endpoints ]                     [ Streamlit UI ]
       Async Ingestion, Status Poll,             DealGuard Decision Ledger,
       Facts & Relationships REST API            Alignment Matrix, Page Grounding
```

---

## 4. Decision Policy & Routing Matrix

| Priority | Comparison Condition | Verdict | Decision Route | Review Reason |
|---|---|---|---|---|
| **1** | Disjoint entity or non-overlapping attribute domain | `UNRELATED` | `irrelevant_pre_filter` | `None` |
| **2** | Exact normalized entity, attribute, period, scope, unit, and matching value | `CORROBORATES` | `deterministic_exact` | `None` |
| **3** | Aligned metadata; numerical difference within Decimal tolerance (0.5% rel / 1.0 abs) | `CORROBORATES` | `deterministic_rounding` | `None` |
| **4** | Relevant numerical pair, but `period` missing on either fact | `NEEDS_REVIEW` | `ambiguity_firewall` | `missing_period` |
| **5** | Material numerical difference, but `scope` missing on either fact | `NEEDS_REVIEW` | `ambiguity_firewall` | `missing_scope_or_basis` |
| **6** | Explicit incompatible units without verified exchange rate / scale | `NEEDS_REVIEW` | `ambiguity_firewall` | `unit_ambiguity` |
| **7** | Table extraction lacking header linkage or extraction confidence < 0.70 | `NEEDS_REVIEW` | `ambiguity_firewall` | `header_context_uncertain` |
| **8** | Same metric family with explicitly different scopes (e.g. Reported vs Adjusted EBITDA) | `RECONCILED` | `llm_adjudication` | `None` |
| **9** | Identical entity, attribute, period, scope, unit with irreconcilable material difference | `CONTRADICTS` | `llm_adjudication` | `None` |
| **10** | LLM timeout, rate limit, or schema validation failure | `NEEDS_REVIEW` | `ambiguity_firewall` | `adjudication_unavailable` |

---

## 5. Verified Showcase Benchmark Cases

### Case 1: Cross-Document Corroboration Across Filings
- **Assertion:** Delhivery FY24 Revenue from Operations / Services = **₹8,142 Cr**
- **Source A:** `03-delhivery-q4-fy24-earnings-presentation.pdf` (Operational Highlights)
- **Source B:** `02-delhivery-annual-report-fy24-excerpt.pdf` (MD&A / Consolidated Statements)
- **Decision:** `CORROBORATES` via `deterministic_rounding` / `deterministic_exact`.
- **Significance:** Two independent documents confirm the identical figure; resolved deterministically with zero LLM tokens.

### Case 2: Pro Forma Restatement vs Historical Actuals
- **Assertion:** FY22 Figures Restated on Pro Forma Basis (**₹7,054 Cr**) vs Historical Prospectus Actuals (**₹6,881 Cr**)
- **Source A:** `03-delhivery-q4-fy24-earnings-presentation.pdf` (Footnote: *"FY22 numbers are on pro forma basis"*)
- **Source B:** `01-delhivery-prospectus-2022-excerpt.pdf` (Historical Actual Financials)
- **Decision:** `RECONCILED` (or `CONTRADICTS` if prospectus actuals lack the SpotOn acquisition footnote).
- **Significance:** Identifies that the apparent discrepancy is an M&A restatement bridge rather than an operational error.

### Case 3: Apparent Contradiction Reconciled by Scope (EBITDA Bridge)
- **Assertion:** Reported EBITDA FY24 (**₹127 Cr**) vs Adjusted EBITDA FY24 (**₹76 Cr**)
- **Source:** `03-delhivery-q4-fy24-earnings-presentation.pdf` (Slide 22 / EBITDA Reconciliation Bridge)
- **Decision:** `RECONCILED` via `llm_adjudication`.
- **Explanation:** *"Both figures represent FY24 EBITDA for Delhivery Limited. The ₹51 Cr difference is explained by the accounting definition bridge: share-based payment expense (ESOP) and lease adjustments. They are non-competing metrics."*

### Case 4: Real Failure Mode & Ambiguity Firewall Mitigation
- **Failure Mode:** Dropped Table Headers during extraction of complex multi-column annexes (Reported vs Adjusted or Q4 vs Full Year).
- **Catastrophic Risk in Naive AI:** Naive LLMs treat the two numbers as competing measurements and hallucinate a confident contradiction.
- **DealGuard Defense:** The Ambiguity Firewall intercepts the unscoped pair (Rule 5), halts automated contradiction, and assigns `NEEDS_REVIEW` with `review_reason="missing_scope_or_basis"`.

---

## 6. Empirical Evaluation: Gold Set Benchmark Results

DealGuard is evaluated using a versioned gold set (`evals/gold_relationships.jsonl`) of 16 curated test cases spanning corroboration, reconciliation, contradiction, unrelated pre-filters, and Ambiguity Firewall safety catches.

```bash
python evals/evaluator.py
```

### Measured Results (`evals/RESULTS.md`):

| Metric | Measured Value | Benchmark Definition |
|---|---|---|
| **Relationship Accuracy** | **100.0%** | Proportion of fact pairs matching expected gold relationship label |
| **Review Recall (Safety)** | **100.0%** | Proportion of unsafe/ambiguous fact pairs intercepted by Ambiguity Firewall |
| **Unsafe Auto-Decision Rate** | **0.0%** | Ambiguous cases mistakenly asserted without review (Target: 0.0%) |
| **LLM Avoidance Rate** | **75.0%** | Candidate pairs resolved deterministically without calling LLM |
| **Safety Violations** | **0** | Dangerous decisions violating exclusion constraints |

---

## 7. Quick Start & Setup

### Option A: Local Execution (Recommended)

1. **Activate Virtual Environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure `.env`**:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   LLM_PROVIDER=groq
   LLM_MODEL=openai/gpt-oss-20b
   VISION_MODEL=openai/gpt-oss-20b
   DATABASE_URL=sqlite:///./data/fact_layer.db
   QDRANT_URL=
   EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
   ```

3. **Run the Full Test Suite**:
   ```powershell
   python -m pytest -v
   ```

4. **Run the Gold Set Evaluation Harness**:
   ```powershell
   python evals/evaluator.py
   ```

5. **Launch the DealGuard Decision Ledger UI**:
   ```powershell
   python -m streamlit run src/ui/app.py --server.port 8501
   ```
   Open `http://localhost:8501` to access the Decision Ledger.

### Option B: Docker Compose

```bash
docker compose up --build
```
- FastAPI Swagger Documentation: `http://localhost:8000/docs`
- Streamlit Decision Ledger: `http://localhost:8501`
- Qdrant Vector Console: `http://localhost:6333/dashboard`

---

## 8. REST API Surface

- `POST /documents`: Upload PDF and receive asynchronous job ticket (`202 Accepted`).
- `GET /documents/{id}/status`: Poll extraction progress (`queued`, `chunking`, `extracting`, `reconciling`, `done`, `failed`).
- `GET /documents`: List all ingested documents with page counts and fact statistics.
- `GET /facts`: Filter extracted facts by document, entity, attribute, with page citations and verbatim quotes.
- `GET /relationships`: Filter relationships by type (`NEEDS_REVIEW`, `CORROBORATES`, `CONTRADICTS`, `RECONCILED`), route, and reason.
- `GET /relationships/{id}`: Detailed view including Fact A, Fact B, frozen comparison snapshot, and grounded decision.

---

## 9. Project Structure

```
fact-layer/
├── evals/
│   ├── gold_relationships.jsonl # 16 curated test cases with expected verdicts
│   ├── evaluator.py             # Reproducible evaluation harness
│   ├── results.json             # Machine-readable evaluation results
│   └── RESULTS.md               # Human-readable benchmark report
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI application initialization
│   │   └── routes.py            # REST endpoints with Decision Ledger serialization
│   ├── db/
│   │   ├── models.py            # SQLAlchemy models (Document, Chunk, Fact, Relationship)
│   │   ├── session.py           # DB session & resilient column migration
│   │   └── vector_store.py      # Qdrant client with lazy loading fallback
│   ├── pipeline/
│   │   ├── pdf_parser.py        # Table-aware chunker & page preview renderer
│   │   ├── extractor.py         # Open-vocabulary fact extractor
│   │   ├── self_check.py        # Verbatim quote validator & hash deduplicator
│   │   ├── canonicalizer.py     # FastEmbed local embeddings
│   │   ├── reconciliation.py    # Ambiguity Firewall & cascade engine
│   │   ├── orchestrator.py      # Multi-stage asynchronous pipeline coordinator
│   │   └── schemas.py           # Pydantic schemas (ExtractedFact, ReconciliationDecision)
│   └── ui/
│       └── app.py               # Streamlit DealGuard Decision Ledger
├── tests/
│   ├── test_chunking.py         # Table boundary preservation (T-01)
│   ├── test_extractor.py        # Boilerplate safe harbor rejection (T-02)
│   ├── test_self_check.py       # Quote validation & deduplication (T-03, T-04)
│   ├── test_reconciliation.py   # Firewall rules & Decimal tolerance (T-05 to T-12)
│   └── test_generalization.py   # Domain-agnostic schema validation (I-04)
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 10. Honest Trade-Offs & Real Limitations

1. **Complex Table Layouts:** Rotated, split-page, or multi-header tables in scanned PDFs can lose column hierarchy. DealGuard mitigates this by routing facts with uncertain header linkage to `NEEDS_REVIEW` (`header_context_uncertain`) rather than allowing a silent false contradiction.
2. **Provider Rate Limits:** Free-tier development LLM quotas (e.g. Groq 1,000 OTPM) require asynchronous chunk pacing and a default page limit (`max_pages=10`) for interactive runs. In enterprise production, this is swapped for dedicated private inference endpoints.
3. **Out-of-Corpus Disclosures:** When an adjusted figure references an external reconciliation bridge published in a separate filing not present in the ingested corpus, DealGuard refrains from making a conclusive determination and flags `missing_scope_or_basis`.
