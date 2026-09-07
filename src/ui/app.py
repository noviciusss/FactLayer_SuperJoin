"""Streamlit Frontend for DealGuard: Evidence-First Fact Reconciliation for IPO Readiness."""
import os
import sys
import textwrap
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
    page_title="DealGuard — IPO Fact Reconciliation",
    page_icon="🛡️",
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
    border-color: #166534;
}

.badge-contradicts {
    background-color: #fee2e2;
    color: #991b1b;
    border-color: #991b1b;
}

.badge-reconciled {
    background-color: #ede9fe;
    color: #5b21b6;
    border-color: #5b21b6;
}

.badge-needs_review {
    background-color: #fef3c7;
    color: #92400e;
    border-color: #b45309;
}

.badge-unrelated {
    background-color: #f1f5f9;
    color: #475569;
    border-color: #475569;
}

.badge-route {
    background-color: #eff6ff;
    color: #1d4ed8;
    border-color: #3b82f6;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    text-transform: none;
}

.badge-reason {
    background-color: #fdf2f8;
    color: #be185d;
    border-color: #db2777;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    text-transform: none;
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
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f172a;
}

/* KPI Stat Cards */
.kpi-stat-card {
    background: #ffffff;
    border: 1.5px solid #0f172a;
    border-radius: 6px;
    box-shadow: 3px 3px 0px #0f172a;
    padding: 0.75rem 1rem;
    text-align: center;
}

.kpi-stat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f172a;
}

.kpi-stat-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 700;
    margin-top: 0.2rem;
}

/* Alignment Matrix Table */
.matrix-table {
    width: 100%;
    font-size: 0.82rem;
    border-collapse: collapse;
    margin-top: 0.5rem;
}

.matrix-table td, .matrix-table th {
    padding: 4px 8px;
    border: 1px solid #e2e8f0;
}

.pill-pass {
    background: #dcfce7;
    color: #166534;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.72rem;
}

.pill-fail {
    background: #fee2e2;
    color: #991b1b;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.72rem;
}

.pill-blocked {
    background: #fef3c7;
    color: #92400e;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.72rem;
}

/* Case 4 Box */
.case4-box {
    background: #fff1f2;
    border: 2px dashed #e11d48;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
</style>
"""

st.markdown(SIGNAL_BLOCK_CSS, unsafe_allow_html=True)


def md_html(raw_html: str):
    """Render HTML safely without Markdown treating indented lines as code blocks."""
    st.markdown(textwrap.dedent(raw_html).strip(), unsafe_allow_html=True)


# ── Database Helpers ──────────────────────────────────────────────────
def get_db():
    return SessionLocal()


def fetch_system_metrics():
    db = get_db()
    try:
        total_docs = db.query(Document).count()
        total_facts = db.query(Fact).count()
        total_rels = db.query(Relationship).count()
        llm_avoided = 0
        if hasattr(Relationship, "decision_route"):
            llm_avoided = db.query(Relationship).filter(
                Relationship.decision_route.in_(["deterministic_exact", "deterministic_rounding", "irrelevant_pre_filter"])
            ).count()
        blocked_reviews = 0
        if hasattr(RelationType, "NEEDS_REVIEW"):
            blocked_reviews = db.query(Relationship).filter(
                Relationship.relation_type == RelationType.NEEDS_REVIEW
            ).count()
        return {
            "total_docs": total_docs,
            "total_facts": total_facts,
            "total_rels": total_rels,
            "llm_avoided": llm_avoided,
            "blocked_reviews": blocked_reviews,
        }
    finally:
        db.close()


def fetch_documents():
    db = get_db()
    try:
        return db.query(Document).order_by(Document.upload_ts.desc()).all()
    finally:
        db.close()


from sqlalchemy.orm import joinedload


def fetch_facts(doc_id=None, entity=None, attribute=None):
    db = get_db()
    try:
        q = db.query(Fact).options(joinedload(Fact.chunk), joinedload(Fact.document))
        if doc_id:
            q = q.filter(Fact.document_id == doc_id)
        if entity:
            q = q.filter(Fact.entity.ilike(f"%{entity}%"))
        if attribute:
            q = q.filter(Fact.attribute.ilike(f"%{attribute}%"))
        return q.all()
    finally:
        db.close()


def fetch_relationships(rel_type=None, decision_route=None, review_reason=None, search=None):
    db = get_db()
    try:
        q = db.query(Relationship).options(
            joinedload(Relationship.fact_a).joinedload(Fact.document),
            joinedload(Relationship.fact_a).joinedload(Fact.chunk),
            joinedload(Relationship.fact_b).joinedload(Fact.document),
            joinedload(Relationship.fact_b).joinedload(Fact.chunk),
        )
        if rel_type and rel_type != "ALL":
            if hasattr(RelationType, rel_type):
                q = q.filter(Relationship.relation_type == RelationType[rel_type])
        if decision_route and decision_route != "ALL":
            if hasattr(Relationship, "decision_route"):
                q = q.filter(Relationship.decision_route == decision_route)
        if review_reason and review_reason != "ALL":
            if hasattr(Relationship, "review_reason"):
                q = q.filter(Relationship.review_reason == review_reason)
        if search:
            q = q.join(Fact, (Relationship.fact_a_id == Fact.id) | (Relationship.fact_b_id == Fact.id)).filter(
                Fact.entity.ilike(f"%{search}%") | Fact.attribute.ilike(f"%{search}%")
            ).distinct()
        return q.order_by(Relationship.created_at.desc()).all()
    finally:
        db.close()


# ── Sidebar Navigation ────────────────────────────────────────────────
st.sidebar.title("🛡️ DEALGUARD")
st.sidebar.caption("Evidence-First Fact Reconciliation for IPO Readiness")

nav = st.sidebar.radio(
    "Navigation",
    [
        "1. Ingestion & Documents",
        "2. Facts Browser & Grounding",
        "3. Decision Ledger",
        "4. The 4 Benchmark Cases"
    ]
)

# ── Screen 1: Ingestion & Documents ───────────────────────────────────
if nav == "1. Ingestion & Documents":
    st.header("📄 Document Ingestion & Status")
    st.write("Upload financial filings, prospectuses, or macroeconomic publications. The system extracts open-vocabulary facts, verifies page grounding, and builds the canonical fact ledger.")

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

            st.success(f"Uploaded {uploaded_file.name}. Starting DealGuard pipeline...")

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
            md_html(f"""
            <div class="signal-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.1rem; font-weight: 700;">{d.filename}</span>
                        <span style="margin-left: 0.5rem;" class="badge badge-reconciled">{d.doc_type_guess or 'filing'}</span>
                    </div>
                    <div>
                        <span class="badge badge-corroborates">{d.status.value.upper()}</span>
                    </div>
                </div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">
                    Pages: <b>{d.page_count}</b> | Uploaded: {d.upload_ts.strftime('%Y-%m-%d %H:%M:%S')} | ID: <code>{d.id[:8]}...</code>
                </div>
            </div>
            """)


# ── Screen 2: Facts Browser & Grounding ───────────────────────────────
elif nav == "2. Facts Browser & Grounding":
    st.header("🔍 Extracted Facts & Grounded Evidence")
    st.write("Browse open-vocabulary structured facts with verbatim evidence quotes and high-resolution page image inspection.")

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
                st.markdown(f"**Scope:** `{f.scope or 'None'}` | **Extraction Confidence:** `{round(f.confidence, 2)}`")
                if f.qualifiers:
                    st.json(f.qualifiers)
                md_html(f"""
                <div class="evidence-quote">
                    <b>Verbatim Evidence:</b> "{f.evidence_quote}"
                </div>
                """)

                if f.chunk and f.chunk.image_ref and os.path.exists(f.chunk.image_ref):
                    if st.button(f"Inspect Source Page {f.chunk.page_number}", key=f"btn_{f.id}"):
                        st.session_state["selected_image_ref"] = f.chunk.image_ref
                        st.session_state["selected_fact_quote"] = f.evidence_quote
                        st.session_state["selected_page_num"] = f.chunk.page_number

    with col_preview:
        st.subheader("Source Page Grounding")
        if "selected_image_ref" in st.session_state and os.path.exists(st.session_state["selected_image_ref"]):
            st.image(st.session_state["selected_image_ref"], caption=f"Page {st.session_state.get('selected_page_num')} Preview")
            md_html(f"""
            <div class="evidence-quote">
                <b>Verified Evidence Quote:</b><br/>"{st.session_state.get('selected_fact_quote')}"
            </div>
            """)
        else:
            st.info("Click 'Inspect Source Page' on any fact in the left panel to inspect the source page preview.")


# ── Screen 3: Decision Ledger (The Main Stage) ────────────────────────
elif nav == "3. Decision Ledger":
    st.header("⚖️ DealGuard Decision Ledger")
    st.write("Auditable fact-pair relationships evaluated by the Ambiguity Firewall and rules-first reconciliation engine. Every decision records exact field alignment, decision routes, and abstention reasons.")

    # ── Top KPI Metrics Bar ───────────────────────────────────────────
    metrics = fetch_system_metrics()
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        md_html(f"""
        <div class="kpi-stat-card">
            <div class="kpi-stat-num">{metrics['total_docs']}</div>
            <div class="kpi-stat-label">Documents Ingested</div>
        </div>
        """)
    with c2:
        md_html(f"""
        <div class="kpi-stat-card">
            <div class="kpi-stat-num">{metrics['total_facts']}</div>
            <div class="kpi-stat-label">Facts Extracted</div>
        </div>
        """)
    with c3:
        md_html(f"""
        <div class="kpi-stat-card">
            <div class="kpi-stat-num">{metrics['total_rels']}</div>
            <div class="kpi-stat-label">Pairs Evaluated</div>
        </div>
        """)
    with c4:
        md_html(f"""
        <div class="kpi-stat-card">
            <div class="kpi-stat-num" style="color: #166534;">{metrics['llm_avoided']}</div>
            <div class="kpi-stat-label">LLM Calls Avoided</div>
        </div>
        """)
    with c5:
        md_html(f"""
        <div class="kpi-stat-card">
            <div class="kpi-stat-num" style="color: #b45309;">{metrics['blocked_reviews']}</div>
            <div class="kpi-stat-label">Reviews Blocked</div>
        </div>
        """)

    st.divider()

    # ── Multi-Dimensional Filters ─────────────────────────────────────
    col_r1, col_r2, col_r3, col_r4 = st.columns([1.5, 1.5, 1.5, 2])
    with col_r1:
        filter_type = st.selectbox("Relationship Type", ["ALL", "NEEDS_REVIEW", "CORROBORATES", "CONTRADICTS", "RECONCILED"])
    with col_r2:
        filter_route = st.selectbox("Decision Route", ["ALL", "deterministic_exact", "deterministic_rounding", "ambiguity_firewall", "llm_adjudication"])
    with col_r3:
        filter_reason = st.selectbox("Review Reason", ["ALL", "missing_period", "missing_scope_or_basis", "unit_ambiguity", "header_context_uncertain", "adjudication_unavailable"])
    with col_r4:
        filter_search = st.text_input("Search Entity / Attribute", placeholder="e.g. EBITDA, Revenue, Delhivery")

    relationships = fetch_relationships(
        rel_type=filter_type,
        decision_route=filter_route,
        review_reason=filter_reason,
        search=filter_search or None
    )

    st.caption(f"Showing **{len(relationships)}** relationships matching criteria")

    if not relationships:
        st.info("No cross-document relationships found for this filter combination. Try clearing filters or ingesting overlapping filings.")
    else:
        for r in relationships:
            with st.container(border=True):
                badge_class = f"badge-{r.relation_type.value.lower()}"
                route_str = r.decision_route or "deterministic_rule"
                reason_str = r.review_reason

                col_head1, col_head2 = st.columns([3, 1])
                with col_head1:
                    badge_html = f'<span class="badge {badge_class}">[{r.relation_type.value}]</span> <span class="badge badge-route">Route: {route_str}</span>'
                    if reason_str:
                        badge_html += f' <span class="badge badge-reason">Reason: {reason_str}</span>'
                    st.markdown(badge_html, unsafe_allow_html=True)
                with col_head2:
                    st.markdown(f'<div style="text-align: right; font-size: 0.8rem; color: #64748b; font-family: monospace;">Rel ID: <b>{r.id[:8]}</b> | Conf: <b>{round(r.confidence, 2)}</b></div>', unsafe_allow_html=True)

                # Fact A and Fact B Side-by-Side
                c1, c2 = st.columns(2)
                fa = r.fact_a
                fb = r.fact_b

                with c1:
                    md_html(f"""
                    <div class="fact-box">
                        <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 700;">
                            FACT A: {fa.document.filename if fa and fa.document else 'Doc A'} (p.{fa.chunk.page_number if fa and fa.chunk else '?'})
                        </div>
                        <div style="font-size: 1.05rem; font-weight: 700; margin-top: 0.25rem;">{fa.entity if fa else 'N/A'} — {fa.attribute if fa else 'N/A'}</div>
                        <div class="metric-value">{fa.value if fa else ''} <span style="font-size: 0.9rem;">{fa.unit or '' if fa else ''}</span></div>
                        <div style="font-size: 0.82rem; margin-top: 0.25rem; color: #475569;">
                            <b>Period:</b> <code>{fa.period or 'null'}</code> | <b>Scope:</b> <code>{fa.scope or 'null'}</code>
                        </div>
                        <div class="evidence-quote">"{fa.evidence_quote if fa else ''}"</div>
                    </div>
                    """)

                with c2:
                    md_html(f"""
                    <div class="fact-box">
                        <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 700;">
                            FACT B: {fb.document.filename if fb and fb.document else 'Doc B'} (p.{fb.chunk.page_number if fb and fb.chunk else '?'})
                        </div>
                        <div style="font-size: 1.05rem; font-weight: 700; margin-top: 0.25rem;">{fb.entity if fb else 'N/A'} — {fb.attribute if fb else 'N/A'}</div>
                        <div class="metric-value">{fb.value if fb else ''} <span style="font-size: 0.9rem;">{fb.unit or '' if fb else ''}</span></div>
                        <div style="font-size: 0.82rem; margin-top: 0.25rem; color: #475569;">
                            <b>Period:</b> <code>{fb.period or 'null'}</code> | <b>Scope:</b> <code>{fb.scope or 'null'}</code>
                        </div>
                        <div class="evidence-quote">"{fb.evidence_quote if fb else ''}"</div>
                    </div>
                    """)

                # Alignment Check & Comparison Snapshot
                snapshot = r.comparison_snapshot or {}
                alignment = snapshot.get("alignment", {})

                with st.expander("🔍 Field Alignment Check & Snapshot Details", expanded=(r.relation_type == RelationType.NEEDS_REVIEW)):
                    align_ent = '<span class="pill-pass">PASS</span>' if alignment.get("entity_match") else '<span class="pill-fail">MISMATCH</span>'
                    align_attr = '<span class="pill-pass">PASS</span>' if alignment.get("attribute_match") else '<span class="pill-fail">MISMATCH</span>'

                    if not alignment.get("has_period_a") or not alignment.get("has_period_b"):
                        align_period = '<span class="pill-blocked">BLOCKED — Period Missing</span>'
                    elif alignment.get("period_match"):
                        align_period = '<span class="pill-pass">PASS</span>'
                    else:
                        align_period = '<span class="pill-fail">DIFFERENT PERIODS</span>'

                    if not alignment.get("has_scope_a") or not alignment.get("has_scope_b"):
                        align_scope = '<span class="pill-blocked">BLOCKED — Scope Unstated</span>'
                    elif alignment.get("scope_match"):
                        align_scope = '<span class="pill-pass">PASS</span>'
                    else:
                        align_scope = '<span class="pill-fail">DIFFERENT SCOPES</span>'

                    if not alignment.get("unit_match"):
                        align_unit = '<span class="pill-blocked">BLOCKED — Unit Incompatible</span>'
                    else:
                        align_unit = '<span class="pill-pass">PASS</span>'

                    val_diff = alignment.get("value_diff")
                    if alignment.get("value_close"):
                        align_val = f'<span class="pill-pass">PASS (Within Tolerance: Δ={val_diff or "0"})</span>'
                    else:
                        align_val = f'<span class="pill-fail">MATERIAL DIFFERENCE (Δ={val_diff or "N/A"})</span>'

                    md_html(f"""
                    <table class="matrix-table">
                        <tr><th>Dimension</th><th>Status</th><th>Notes</th></tr>
                        <tr><td><b>Entity</b></td><td>{align_ent}</td><td>Normalized lexical token overlap</td></tr>
                        <tr><td><b>Attribute</b></td><td>{align_attr}</td><td>Metric family / synonym overlap</td></tr>
                        <tr><td><b>Period</b></td><td>{align_period}</td><td>Fact A: <code>{fa.period if fa else 'null'}</code> vs Fact B: <code>{fb.period if fb else 'null'}</code></td></tr>
                        <tr><td><b>Scope</b></td><td>{align_scope}</td><td>Fact A: <code>{fa.scope if fa else 'null'}</code> vs Fact B: <code>{fb.scope if fb else 'null'}</code></td></tr>
                        <tr><td><b>Unit</b></td><td>{align_unit}</td><td>Fact A: <code>{fa.unit if fa else 'null'}</code> vs Fact B: <code>{fb.unit if fb else 'null'}</code></td></tr>
                        <tr><td><b>Value</b></td><td>{align_val}</td><td>Compared using Decimal precision</td></tr>
                    </table>
                    """)

                # Grounded System Decision
                md_html(f"""
                <div style="margin-top: 0.75rem; padding: 0.75rem; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px;">
                    <div style="font-weight: 700; font-size: 0.88rem; color: #0f172a;">Grounded System Decision:</div>
                    <div style="font-size: 0.88rem; color: #334155; margin-top: 0.25rem;">{r.explanation}</div>
                    {f'<div style="font-size: 0.82rem; color: #4338ca; margin-top: 0.5rem; font-weight: 600;"><b>Reconciliation Basis:</b> {r.reconciliation_basis}</div>' if r.reconciliation_basis else ''}
                </div>
                """)


# ── Screen 4: The 4 Benchmark Cases ───────────────────────────────────
elif nav == "4. The 4 Benchmark Cases":
    st.header("🎯 The 4 Required Evaluation Cases")
    st.write("Direct showcase for assignment evaluation: Corroboration, Contradiction, Reconciled Ambiguity, and the Real Failure Mode (Case 4).")

    tabs = st.tabs([
        "Case 1: Real Cross-Doc Corroboration",
        "Case 2: Pro Forma vs Historical Actuals",
        "Case 3: Contextual Reconciliation (EBITDA Bridge)",
        "Case 4: Real Failure Mode & Ambiguity Firewall"
    ])

    with tabs[0]:
        st.subheader("Case 1: Cross-Document Corroboration Across Filings")
        st.markdown("""
        **Target Metric:** Delhivery FY24 Revenue from Operations / Services (**₹8,142 Cr**)
        - **Source A:** `03-delhivery-q4-fy24-earnings-presentation.pdf` (Slide 7)
        - **Source B:** `02-delhivery-annual-report-fy24-excerpt.pdf` (Consolidated Financial Statements)
        - **Why it matters:** Same entity, same attribute, same period (`FY24`), same scope (`consolidated`), and matching value.
        - **System Behavior:** Resolved deterministically with `route=deterministic_rounding` or `deterministic_exact` without calling the LLM.
        """)
        corroborated_rels = fetch_relationships(rel_type=RelationType.CORROBORATES.value)
        if corroborated_rels:
            st.success(f"Found {len(corroborated_rels)} corroborated relationship(s) in active database.")
            r = corroborated_rels[0]
            st.markdown(f"**Explanation:** {r.explanation}")
        else:
            st.info("Ingest both `03-delhivery-q4-fy24-earnings-presentation.pdf` and `02-delhivery-annual-report-fy24-excerpt.pdf` to see live pair.")

    with tabs[1]:
        st.subheader("Case 2: Pro Forma Restatement vs Historical Actuals")
        st.markdown("""
        **Target Metric:** FY22 Figures Restated on Pro Forma Basis (**₹7,054 Cr**) vs Historical Prospectus Actuals
        - **Source A:** `03-delhivery-q4-fy24-earnings-presentation.pdf` (Footnote: *'FY22 numbers are on pro forma basis'*)
        - **Source B:** `01-delhivery-prospectus-2022-excerpt.pdf` (Historical Actual Financials)
        - **Why it conflicts:** The 2022 Prospectus was filed prior to the SpotOn acquisition pro forma adjustment.
        - **System Verdict:** `RECONCILED` citing the pro forma restatement footnote, or flagged for review if footnote was ungrounded.
        """)
        contradict_rels = fetch_relationships(rel_type=RelationType.CONTRADICTS.value)
        if contradict_rels:
            st.warning(f"Found {len(contradict_rels)} contradiction(s) in active database.")
        else:
            st.info("Ingest `01-delhivery-prospectus-2022-excerpt.pdf` and `03-delhivery-q4-fy24-earnings-presentation.pdf` to inspect.")

    with tabs[2]:
        st.subheader("Case 3: Apparent Contradiction Reconciled by Accounting Scope (EBITDA Bridge)")
        st.markdown("""
        **Target Metric:** Reported EBITDA FY24 (**₹127 Cr**) vs Adjusted EBITDA FY24 (**₹76 Cr**)
        - **Source:** `03-delhivery-q4-fy24-earnings-presentation.pdf` (Slide 22 / EBITDA Reconciliation Bridge)
        - **Reconciliation Basis:** Accounting definition bridge: ESOP / share-based payments add-back, IPO expenses, and lease accounting add-backs.
        - **System Verdict:** `RECONCILED` — Never incorrectly labeled as a contradiction.
        """)
        reconciled_rels = fetch_relationships(rel_type=RelationType.RECONCILED.value)
        if reconciled_rels:
            st.info(f"Found {len(reconciled_rels)} reconciled relationship(s) in active database.")
        else:
            st.info("Ingest `03-delhivery-q4-fy24-earnings-presentation.pdf` to trigger EBITDA bridge reconciliation.")

    with tabs[3]:
        st.subheader("Case 4: Real Failure Mode & Ambiguity Firewall Defense")
        md_html("""
        <div class="case4-box">
            <h4 style="color: #e11d48; margin-top: 0;">Stress-Tested Failure Mode: Dropped Scope in Multi-Column Annex Tables</h4>
            <p>
                <b>Real-World Failure:</b> In complex tables with merged headers (e.g. Reported vs Adjusted figures or Q4 vs Full Year columns),
                OCR or LLM chunk extractors occasionally extract the numeric figure while dropping the 'scope' or 'quarter' qualifier.
            </p>
            <p>
                <b>Catastrophic Risk in Naive Systems:</b> A normal LLM will compare ₹127 Cr against ₹76 Cr, observe the material difference, and hallucinate a confident <code>CONTRADICTS</code> assertion.
            </p>
            <p>
                <b>DealGuard Ambiguity Firewall Defense:</b>
                <ol>
                    <li><b>Rule 5 Triggered:</b> Values materially differ, but <code>scope</code> is null on one fact.</li>
                    <li><b>Firewall Blocks Comparison:</b> Halts execution <i>before</i> LLM adjudication.</li>
                    <li><b>Verdict:</b> Outputs <code>NEEDS_REVIEW</code> with reason <code>missing_scope_or_basis</code>.</li>
                    <li><b>Analyst Audit:</b> The Decision Ledger flags the exact missing field so an analyst can verify in 15 seconds.</li>
                </ol>
            </p>
        </div>
        """)
        review_rels = fetch_relationships(rel_type=RelationType.NEEDS_REVIEW.value)
        if review_rels:
            st.warning(f"Found {len(review_rels)} Ambiguity Firewall blocked relationship(s) in active database.")
            r = review_rels[0]
            st.markdown(f"**Live Example:** {r.explanation}")
        else:
            st.info("Run tests or benchmark to view live Ambiguity Firewall blocked relationships.")
