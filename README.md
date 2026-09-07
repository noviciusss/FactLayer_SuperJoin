# Fact Knowledge Layer

> A domain-agnostic fact extraction, evidence grounding, and cross-document reconciliation engine built for the Superjoin VIT 2026 Engineering Assignment.

---

## 1. System Overview & Architecture

The Fact Knowledge Layer ingests multi-page documents (corporate filings, prospectuses, earnings decks, macroeconomic reports), extracts grounded factual assertions without hardcoded field schemas or closed vocabularies, clusters candidate pairs using incremental dense vector search, and evaluates relationships (Corroboration, Contradiction, Reconciled Context, Unrelated) using a **rules-first, LLM-second** reasoning cascade.

```
                              [ PDF Document Ingestion ]
                                          │
                                          ▼
                             [ Stage A: Ingest & Chunk ]
                       • Table boundary preservation (pdfplumber)
                       • Image/Infographic slide detection
                       • High-res page preview rendering (PyMuPDF)
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
            [ Standard Chunks ]                         [ Visual Slides ]
                    │                                           │
                    ▼                                           ▼
          [ Stage B: Extraction ]                     [ Vision Fallback ]
         Groq LLaMA / GPT-OSS 20B                    Groq Vision / Multimodal
        (Open-vocabulary extraction)                 (Infographic parsing)
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          │
                                          ▼
                             [ Stage B.2: Self-Check Node ]
                        • Verifies verbatim evidence_quote in text
                        • Suppresses hallucinations & boilerplate
                        • Idempotency content-hash deduplication
                                          │
                                          ▼
                         [ Stage C: Canonicalize & Embed ]
                         "Entity | Attribute | Period | Value"
                         Local FastEmbed (BAAI/bge-small-en-v1.5)
                                          │
                                          ▼
                         [ Stage D: Candidate Clustering ]
                         Qdrant vector similarity k-NN search
                         (Incremental: only searches against existing)
                                          │
                                          ▼
                        [ Stage E: Reconciliation Cascade ]
                    ┌──────────────────────────────────────────┐
                    │ 1. Exact Match Rule   ──► CORROBORATES   │ (Zero LLM cost)
                    │ 2. Heuristic Filter   ──► UNRELATED      │ (Zero LLM cost)
                    │ 3. LLM Adjudication   ──► RECONCILED /   │ (Detailed Bridge
                    │                           CONTRADICTS    │  Explanations)
                    └─────────────────────┬────────────────────┘
                                          │
                                          ▼
                          [ Relational PostgreSQL / SQLite ]
                       (Documents, Chunks, Facts, Relationships)
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
                    ▼                                           ▼
           [ FastAPI Endpoints ]                     [ Streamlit UI ]
        Async 202 Ingestion, Status               "Signal Block" Design,
        Facts & Relationships CRUD                 Evidence Inspector &
                                                   4 Benchmark Cases
```

---

## 2. Setup & Run Instructions

### Option A: Local Standalone Execution (Fastest)

1. **Activate Virtual Environment & Install Dependencies**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables** in `.env`:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   LLM_PROVIDER=groq
   LLM_MODEL=openai/gpt-oss-20b
   VISION_MODEL=openai/gpt-oss-20b
   DATABASE_URL=sqlite:///./data/fact_layer.db
   QDRANT_URL=
   EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
   ```

3. **Launch the Streamlit UI** (Signal Block theme):
   ```bash
   python -m streamlit run src/ui/app.py --server.port 8501
   ```
   Open `http://localhost:8501` in your browser.

4. **Launch the FastAPI Backend** (Optional):
   ```bash
   python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
   ```
   Interactive OpenAPI documentation available at `http://127.0.0.1:8000/docs`.

5. **Run Automated Benchmark Script**:
   ```bash
   python run_benchmark.py --max-pages 5
   # Or evaluate active database records:
   python run_benchmark.py --eval-only
   ```

### Option B: Full Containerized Stack (Docker Compose)

```bash
docker compose up --build
```
- **Streamlit UI**: `http://localhost:8501`
- **FastAPI API**: `http://localhost:8000/docs`
- **Qdrant Dashboard**: `http://localhost:6333/dashboard`
- **PostgreSQL**: Port `5432`

---

## 3. Automated Test Suite

Run the full unit and integration test suite:
```bash
python -m pytest tests/test_reconciliation.py tests/test_self_check.py tests/test_chunking.py tests/test_extractor.py tests/test_generalization.py -v
```

| Test ID | Test Name | Focus | Status |
|---|---|---|---|
| **T-01** | `test_t01_table_chunking_preserves_table` | Table bounding box detection and markdown preservation without shredding | **PASSED** |
| **T-02** | `test_t02_boilerplate_safe_harbor_returns_empty` | Suppression of legal boilerplate & safe harbor disclaimers | **PASSED** |
| **T-03** | `test_t03_self_check_verifies_exact_quote` | Verbatim quote verification in source text | **PASSED** |
| **T-03b**| `test_t03_self_check_rejects_hallucinated_quote` | Rejection & confidence penalty for ungrounded claims | **PASSED** |
| **T-04** | `test_t04_idempotent_duplicate_deduplication` | Idempotent content-hash deduplication across uploads | **PASSED** |
| **T-05** | `test_t05_numeric_tolerance_corroborates` | Numeric tolerance rule handling rounding (e.g. 8,142 vs 8,142.3) | **PASSED** |
| **I-05** | `test_i05_domain_agnostic_fact_representation` | Zero-shot generalization to aerospace and macro policy domains | **PASSED** |

---

## 4. The 4 Benchmark Evaluation Cases

### Case 1: Corroborated Fact Across Filings
- **Assertion**: FY24 Revenue from operations / services (`₹8,142 Cr`)
- **Source A**: Delhivery Q4 FY24 Earnings Presentation (Slide 7 / Operational Highlights)
- **Source B**: Delhivery FY24 Annual Report (MD&A / Consolidated Financial Statements)
- **Verdict**: `CORROBORATES` (Rule match & LLM confirmation)
- **Explanation**: Both documents report consistent figures for Delhivery revenue from operations (`₹8,142 Cr`) for the fiscal year ended March 31, 2024 under consolidated scope.

### Case 2: Genuine / Likely Contradiction
- **Assertion**: TL Year-over-Year Revenue Growth: `40%` vs `13%`
- **Source A**: Delhivery Q4 FY24 Earnings Presentation (Slide 14)
- **Source B**: Delhivery Q4 FY24 Earnings Presentation (Table 2)
- **Verdict**: `CONTRADICTS`
- **Explanation**: Both claims describe TL's year-over-year revenue expansion for the same period. Fact A asserts a 40% YoY expansion, whereas Fact B reports a 13% YoY growth rate. Because no segment qualifiers or alternate time horizons are stated, the figures represent conflicting factual statements.

### Case 3: Apparent Contradiction Reconciled by Context (EBITDA Bridge)
- **Assertion**: Reported EBITDA (`₹127 Cr`) vs Adjusted EBITDA (`₹76 Cr`)
- **Source**: Delhivery Q4 FY24 Earnings Presentation (Slide 22: EBITDA Reconciliation Bridge)
- **Verdict**: `RECONCILED`
- **Reconciliation Basis**: Accounting definition bridge (share-based payment / ESOP add-back, IPO expense deductions, and actual lease rent paid).
- **Explanation**: The discrepancy between ₹127 Cr and ₹76 Cr is reconciled by the management bridge accounting for non-cash share-based compensation and contractual lease adjustments.

### Case 4: Real Stress-Tested Failure Mode
- **Failure Mode Discovered**: Scope / Sibling Dropping in Complex Multi-Column Financial Annexes.
- **Stress-Test Scenario**: In complex financial tables with rotated headers or multi-level column groupings (e.g. Q4 quarterly figures juxtaposed against full-year numbers without repeated inline labels), LLM extraction occasionally captured the numeric value while dropping the `scope` or `period` qualifier.
- **Impact**: Unscoped metrics risk spurious direct comparisons (e.g., matching a quarterly figure against an annual figure as a false contradiction).
- **Engineering Mitigation Built**:
  1. **Self-Check Node**: Verifies that table header qualifiers are captured in the fact metadata before committing to the relational store.
  2. **Strict Scope Rule**: If scope is null and variance is significant, the rule cascade forbids automated corroboration and flags the pair for contextual review.
  3. **Table-Aware Markdown Chunking**: Formatting table columns with markdown headers during Stage A chunking preserves column-to-cell associations.

---

## 5. Architectural Decisions & Brownie Points

1. **Schema Generality**:
   No fields like `revenue` or `ebitda` are hardcoded. Facts are stored as an open envelope (`entity`, `attribute`, `value`, `unit`, `period`, `scope`, `qualifiers JSON`). The `qualifiers` JSON blob allows the schema to evolve dynamically without database migrations.
2. **Rules-First, LLM-Second Cascade**:
   Rule checks resolve exact and rounding matches with zero LLM API calls, preventing token exhaustion and keeping costs negligible.
3. **Incremental Ingestion**:
   When Document N is ingested, only its newly extracted facts are vector-searched against the existing Qdrant corpus. Historical facts are never recomputed from scratch.
4. **Large PDF Scalability**:
   Multi-threaded chunk processing with concurrency controls and page preview image rendering for visual evidence inspection.

---

## 6. Limitations & Future Work

1. **Cross-Page Table Splitting**: Tables spanning across multiple consecutive PDF pages are currently extracted as separate chunk fragments rather than merged into a single logical table entity.
2. **Currency & Scale Normalization**: Automated unit conversion (e.g., converting ₹ Lakhs to ₹ Crores or USD to INR based on historical exchange rates) could be introduced as a pre-normalization step before numeric tolerance checks.

---

## 7. Additional Notes

- **Build Agent**: Built and verified with Google Antigravity.
- **Time Box**: Completed in ~40 hours.
