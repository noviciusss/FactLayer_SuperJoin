"""Streamlit Frontend for Fact Knowledge Layer with Signal Block CSS aesthetic."""
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import List, Optional
import streamlit as st
from src.config import settings
from src.db.models import Document, Fact, JobStatus, Relationship, RelationType
from src.db.session import SessionLocal, init_db
from src.pipeline.orchestrator import orchestrator

# Ensure DB initialized
init_db()

st.set_page_config(
    page_title="Fact Knowledge Layer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Signal Block Design System CSS ────────────────────────────────────
SIGNAL_BLOCK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Signal Block Cards */
.signal-card {
    background: #ffffff;
    border: 2px solid #0f172a;
    border-radius: 8px;
    box-shadow: 4px 4px 0px #0f172a;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
    transition: transform 0.1s ease-in-out;
}

.signal-card:hover {
    transform: translate(-1px, -1px);
    box-shadow: 5px 5px 0px #0f172a;
}

/* Fact Mini-Card */
.fact-box {
    background: #f8fafc;
    border: 1.5px solid #334155;
    border-radius: 6px;
    padding: 1rem;
    height: 100%;
}

/* Badge Tags */
.badge {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-radius: 4px;
    border: 1.5px solid #0f172a;
}

.badge-corroborates {
    background-color: #dcfce7;
    color: #166534;
}

.badge-contradicts {
    background-color: #fee2e2;
    color: #991b1b;
}

.badge-reconciled {
    background-color: #ede9fe;
    color: #5b21b6;
}

.badge-unrelated {
    background-color: #f1f5f9;
    color: #475569;
}

/* Evidence Quote Block */
.evidence-quote {
    background-color: #fef9c3;
    border-left: 3px solid #ca8a04;
    padding: 0.5rem 0.75rem;
    font-size: 0.85rem;
    font-style: italic;
    margin-top: 0.5rem;
    border-radius: 0 4px 4px 0;
}

/* Metric Display */
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #0f172a;
}

/* Case 4 Alert Box */
.case4-box {
    background: #fff1f2;
    border: 2px dashed #e11d48;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
}
</style>
"""

st.markdown(SIGNAL_BLOCK_CSS, unsafe_allow_html=True)


# ── Database Helpers ──────────────────────────────────────────────────
def get_db():
    return SessionLocal()


def fetch_documents():
    db = get_db()
    try:
        return db.query(Document).order_by(Document.upload_ts.desc()).all()
    finally:
        db.close()


def fetch_facts(doc_id=None, entity=None, attribute=None):
    db = get_db()
    try:
        q = db.query(Fact)
        if doc_id:
            q = q.filter(Fact.document_id == doc_id)
        if entity:
            q = q.filter(Fact.entity.ilike(f"%{entity}%"))
        if attribute:
            q = q.filter(Fact.attribute.ilike(f"%{attribute}%"))
        return q.all()
    finally:
        db.close()


def fetch_relationships(rel_type=None):
    db = get_db()
    try:
        q = db.query(Relationship)
        if rel_type:
            q = q.filter(Relationship.relation_type == rel_type)
        return q.order_by(Relationship.created_at.desc()).all()
    finally:
        db.close()


# ── Sidebar Navigation ────────────────────────────────────────────────
st.sidebar.title("⚡ FACT LAYER")
st.sidebar.caption("Domain-Agnostic Fact Extraction & Cross-Document Reconciliation")

nav = st.sidebar.radio(
    "Navigation",
    ["1. Ingestion & Documents", "2. Facts Browser & Evidence", "3. Cross-Doc Reconciliation", "4. The 4 Benchmark Cases"]
)

# ── Screen 1: Ingestion & Documents ───────────────────────────────────
if nav == "1. Ingestion & Documents":
    st.header("📄 Document Ingestion & Status")
    st.write("Upload multi-page PDFs to trigger Stage A (Chunking), Stage B (Fact Extraction), Stage C (Canonicalization), Stage D (Clustering), and Stage E (Reconciliation).")

    col_up, col_starter = st.columns([1, 1])

    with col_up:
        st.subheader("Upload PDF Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        page_limit = st.number_input("Max pages to process (0 for all)", min_value=0, max_value=200, value=15, step=5)
        pages_arg = None if page_limit == 0 else int(page_limit)

        if uploaded_file and st.button("Ingest Document", type="primary"):
            save_path = settings.UPLOAD_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            db = get_db()
            doc = Document(
                filename=uploaded_file.name,
                file_path=str(save_path),
                status=JobStatus.QUEUED
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            doc_id = doc.id
            db.close()

            st.success(f"Uploaded {uploaded_file.name}. Starting pipeline...")
            
            # Progress tracker
            progress_bar = st.progress(10)
            status_text = st.empty()

            try:
                status_text.info("Running Stage A (Table-aware chunking & preview generation)...")
                progress_bar.progress(30)
                res = orchestrator.process_document(doc_id, max_pages=pages_arg)
                progress_bar.progress(100)
                status_text.success(f"Processing Complete! Facts: {res['facts_extracted']}, Relationships: {res['relationships_found']}")
                st.rerun()
            except Exception as e:
                status_text.error(f"Processing failed: {e}")

    with col_starter:
        st.subheader("Quick-Load Starter Datasets")
        st.caption("Ingest curated excerpts from the assignment datasets:")
        
        delhivery_files = list(Path("delhivery").glob("*.pdf")) if Path("delhivery").exists() else []
        macro_files = list(Path("india-macroeconomy").glob("*.pdf")) if Path("india-macroeconomy").exists() else []

        selected_starter = st.selectbox(
            "Select Starter PDF to Ingest",
            options=[str(p) for p in (delhivery_files + macro_files)]
        )

        starter_page_limit = st.number_input("Starter Pages to Process", min_value=1, max_value=100, value=10, step=5)

        if selected_starter and st.button("Ingest Starter Dataset PDF"):
            p = Path(selected_starter)
            target_path = settings.UPLOAD_DIR / p.name
            if not target_path.exists():
                import shutil
                shutil.copy(p, target_path)

            db = get_db()
            doc = Document(
                filename=p.name,
                file_path=str(target_path),
                status=JobStatus.QUEUED
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            doc_id = doc.id
            db.close()

            progress_bar = st.progress(20)
            with st.spinner(f"Ingesting {p.name}..."):
                res = orchestrator.process_document(doc_id, max_pages=int(starter_page_limit))
                progress_bar.progress(100)
                st.success(f"Ingested {p.name}: {res['facts_extracted']} facts, {res['relationships_found']} relationships.")
                st.rerun()

    st.divider()
    st.subheader("Ingested Documents Library")
    docs = fetch_documents()
    if not docs:
        st.info("No documents ingested yet. Upload a PDF or ingest a starter file above.")
    else:
        for d in docs:
            st.markdown(f"""
            <div class="signal-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.1rem; font-weight: 700;">{d.filename}</span>
                        <span style="margin-left: 0.5rem;" class="badge badge-reconciled">{d.doc_type_guess or 'general'}</span>
                    </div>
                    <div>
                        <span class="badge badge-corroborates">{d.status.value.upper()}</span>
                    </div>
                </div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">
                    Pages: <b>{d.page_count}</b> | Uploaded: {d.upload_ts.strftime('%Y-%m-%d %H:%M:%S')} | ID: <code>{d.id[:8]}...</code>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Screen 2: Facts Browser & Evidence ────────────────────────────────
elif nav == "2. Facts Browser & Evidence":
    st.header("🔍 Extracted Facts & Grounded Evidence")
    st.write("Browse structured facts with verbatim evidence quotes and source page image previews.")

    docs = fetch_documents()
    doc_options = {d.id: d.filename for d in docs}
    
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        filter_doc = st.selectbox("Filter by Document", options=["All"] + list(doc_options.keys()), format_func=lambda x: "All Documents" if x == "All" else doc_options.get(x, x))
    with col_f2:
        filter_entity = st.text_input("Filter by Entity", placeholder="e.g. Delhivery, India")
    with col_f3:
        filter_attr = st.text_input("Filter by Attribute", placeholder="e.g. Revenue, EBITDA, GDP")

    selected_doc_id = None if filter_doc == "All" else filter_doc
    facts = fetch_facts(doc_id=selected_doc_id, entity=filter_entity or None, attribute=filter_attr or None)

    st.caption(f"Showing {len(facts)} extracted facts")

    col_list, col_preview = st.columns([1.2, 1])

    with col_list:
        for f in facts:
            with st.expander(f"{f.entity} — {f.attribute}: {f.value} {f.unit or ''} ({f.period or 'N/A'})"):
                st.markdown(f"**Canonical Statement:** `{f.canonical_statement}`")
                st.markdown(f"**Scope:** {f.scope or 'None'} | **Confidence:** `{round(f.confidence, 2)}`")
                if f.qualifiers:
                    st.json(f.qualifiers)
                st.markdown(f"""
                <div class="evidence-quote">
                    <b>Evidence Quote:</b> "{f.evidence_quote}"
                </div>
                """, unsafe_allow_html=True)
                
                if f.chunk and f.chunk.image_ref and os.path.exists(f.chunk.image_ref):
                    if st.button(f"Inspect Page {f.chunk.page_number} Image", key=f"btn_{f.id}"):
                        st.session_state["selected_image_ref"] = f.chunk.image_ref
                        st.session_state["selected_fact_quote"] = f.evidence_quote
                        st.session_state["selected_page_num"] = f.chunk.page_number

    with col_preview:
        st.subheader("Source Page Grounding")
        if "selected_image_ref" in st.session_state and os.path.exists(st.session_state["selected_image_ref"]):
            st.image(st.session_state["selected_image_ref"], caption=f"Page {st.session_state.get('selected_page_num')} Preview")
            st.markdown(f"""
            <div class="evidence-quote">
                <b>Verified Evidence Quote:</b><br/>"{st.session_state.get('selected_fact_quote')}"
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Click 'Inspect Page Image' on any fact in the left panel to preview the grounded source page.")


# ── Screen 3: Cross-Doc Reconciliation (The Main Stage) ───────────────
elif nav == "3. Cross-Doc Reconciliation":
    st.header("⚖️ Cross-Document Reconciliation Engine")
    st.write("Compare candidate pairs across documents evaluated by the rules-first, LLM-second cascade.")

    rel_filter = st.selectbox("Filter Relationship Type", ["ALL", "CORROBORATES", "CONTRADICTS", "RECONCILED"])
    filter_type = None if rel_filter == "ALL" else RelationType[rel_filter]
    
    relationships = fetch_relationships(rel_type=filter_type)
    st.caption(f"Showing {len(relationships)} relationships")

    if not relationships:
        st.info("No cross-document relationships found for this filter. Ingest multiple overlapping documents to trigger reconciliation.")
    else:
        for r in relationships:
            badge_class = f"badge-{r.relation_type.value.lower()}"
            
            st.markdown(f"""
            <div class="signal-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <span class="badge {badge_class}">{r.relation_type.value}</span>
                    <span style="font-size: 0.85rem; color: #64748b;">Confidence: <b>{round(r.confidence, 2)}</b></span>
                </div>
            """, unsafe_allow_html=True)

            # Fact A and Fact B Side-by-Side
            c1, c2 = st.columns(2)
            fa = r.fact_a
            fb = r.fact_b

            with c1:
                st.markdown(f"""
                <div class="fact-box">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 700;">FACT A ({fa.document.filename if fa and fa.document else 'Doc A'})</div>
                    <div style="font-size: 1.05rem; font-weight: 700; margin-top: 0.25rem;">{fa.entity} — {fa.attribute}</div>
                    <div class="metric-value">{fa.value} <span style="font-size: 0.9rem;">{fa.unit or ''}</span></div>
                    <div style="font-size: 0.85rem; margin-top: 0.25rem;"><b>Period:</b> {fa.period or 'N/A'} | <b>Scope:</b> {fa.scope or 'N/A'}</div>
                    <div class="evidence-quote">"{fa.evidence_quote}"</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="fact-box">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 700;">FACT B ({fb.document.filename if fb and fb.document else 'Doc B'})</div>
                    <div style="font-size: 1.05rem; font-weight: 700; margin-top: 0.25rem;">{fb.entity} — {fb.attribute}</div>
                    <div class="metric-value">{fb.value} <span style="font-size: 0.9rem;">{fb.unit or ''}</span></div>
                    <div style="font-size: 0.85rem; margin-top: 0.25rem;"><b>Period:</b> {fb.period or 'N/A'} | <b>Scope:</b> {fb.scope or 'N/A'}</div>
                    <div class="evidence-quote">"{fb.evidence_quote}"</div>
                </div>
                """, unsafe_allow_html=True)

            # Explanation & Reconciliation Basis
            st.markdown(f"""
                <div style="margin-top: 1rem; padding: 0.75rem; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px;">
                    <div style="font-weight: 700; font-size: 0.9rem; color: #0f172a;">Reasoning & Grounded Explanation:</div>
                    <div style="font-size: 0.9rem; color: #334155; margin-top: 0.25rem;">{r.explanation}</div>
                    {f'<div style="font-size: 0.85rem; color: #6366f1; margin-top: 0.5rem;"><b>Reconciliation Basis:</b> {r.reconciliation_basis}</div>' if r.reconciliation_basis else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Screen 4: The 4 Benchmark Cases ───────────────────────────────────
elif nav == "4. The 4 Benchmark Cases":
    st.header("🎯 The 4 Required Evaluation Cases")
    st.write("Direct showcase for assignment evaluation: Corroboration, Contradiction, Reconciled Ambiguity, and the Real Failure Mode (Case 4).")

    tabs = st.tabs([
        "Case 1: Corroborated Fact",
        "Case 2: Genuine Contradiction",
        "Case 3: Reconciled Ambiguity",
        "Case 4: Extraction/Reasoning Failure"
    ])

    with tabs[0]:
        st.subheader("Case 1: Corroborated Fact Across Filings")
        st.markdown("""
        **Target Fact:** FY24 Revenue from Operations / Services (**₹8,142 Cr**)
        - **Source A:** Delhivery Q4 FY24 Earnings Presentation (Slide 7 / Operational Highlights)
        - **Source B:** Delhivery FY24 Annual Report (MD&A / Consolidated Financial Statements)
        - **Expected Verdict:** `CORROBORATES` (Rule match or high LLM corroboration)
        """)
        corroborated_rels = fetch_relationships(rel_type=RelationType.CORROBORATES)
        if corroborated_rels:
            st.success(f"Found {len(corroborated_rels)} corroborated relationship(s) in active database.")
            r = corroborated_rels[0]
            st.markdown(f"**Explanation:** {r.explanation}")
        else:
            st.info("Ingest both `03-delhivery-q4-fy24-earnings-presentation.pdf` and `02-delhivery-annual-report-fy24-excerpt.pdf` to see live live pair.")

    with tabs[1]:
        st.subheader("Case 2: Genuine Contradiction (Pro Forma vs Historical Actuals)")
        st.markdown("""
        **Target Fact:** FY22 Figures Restated on Pro Forma Basis (**₹7,054 Cr**) vs Historical Prospectus Actuals
        - **Source A:** Delhivery Q4 FY24 Earnings Presentation (Footnote: *'FY22 numbers are on pro forma basis'*)
        - **Source B:** Delhivery 2022 Prospectus Summary Financials
        - **Why it conflicts:** The Prospectus was filed prior to the SpotOn acquisition pro forma adjustment.
        - **Expected Verdict:** `CONTRADICTS` (or `RECONCILED` citing the pro forma restatement footnote)
        """)
        contradict_rels = fetch_relationships(rel_type=RelationType.CONTRADICTS)
        if contradict_rels:
            st.warning(f"Found {len(contradict_rels)} contradiction(s) in active database.")
        else:
            st.info("Ingest `01-delhivery-prospectus-2022-excerpt.pdf` and `03-delhivery-q4-fy24-earnings-presentation.pdf` to inspect.")

    with tabs[2]:
        st.subheader("Case 3: Apparent Contradiction Reconciled by Context (EBITDA Bridge)")
        st.markdown("""
        **Target Fact:** Reported EBITDA FY24 (**₹127 Cr**) vs Adjusted EBITDA FY24 (**₹76 Cr**)
        - **Source:** Delhivery Q4 FY24 Earnings Presentation (Slide 22 / EBITDA Reconciliation Bridge)
        - **Reconciliation Basis:** Accounting definition bridge: ESOP / share-based payments add-back, IPO expenses, and actual lease rent paid.
        - **Expected Verdict:** `RECONCILED`
        """)
        reconciled_rels = fetch_relationships(rel_type=RelationType.RECONCILED)
        if reconciled_rels:
            st.info(f"Found {len(reconciled_rels)} reconciled relationship(s) in active database.")
        else:
            st.info("Ingest `03-delhivery-q4-fy24-earnings-presentation.pdf` to trigger EBITDA bridge reconciliation.")

    with tabs[3]:
        st.subheader("Case 4: Real Extraction & Reasoning Failure (Stress-Tested)")
        st.markdown("""
        <div class="case4-box">
            <h4 style="color: #e11d48; margin-top: 0;">Documented Failure Mode: Scope / Sibling Dropping in Complex Annexes</h4>
            <p>
                <b>Stress-Test Scenario:</b> In complex tables with merged headers or quarterly columns (e.g., Q4 vs Full Year FY24 or Reported vs Adjusted figures without explicit inline labels),
                LLMs can occasionally extract the numeric value while missing the 'scope' or 'quarter' qualifier.
            </p>
            <p>
                <b>Resulting Failure:</b> When scope is dropped, the system risks incorrectly matching a Reported EBITDA figure directly against an Adjusted EBITDA figure as an unexplained contradiction or spurious corroboration instead of recognizing the need for the bridge.
            </p>
            <p>
                <b>Engineering Defense & Mitigation:</b>
                <ol>
                    <li><b>Grounding Self-Check Node:</b> Penalizes confidence when qualifiers appear in table headers but not in the fact cell.</li>
                    <li><b>Strict Scope Requirement:</b> If scope is null and variance is significant, the rule cascade forbids auto-corroboration and forces LLM reconciliation review.</li>
                    <li><b>Table-Aware Chunking:</b> Formatting table columns with markdown headers preserves header-to-cell associations.</li>
                </ol>
            </p>
        </div>
        """, unsafe_allow_html=True)
