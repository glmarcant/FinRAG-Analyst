"""
FinRAG Analyst — Streamlit UI
==============================
Run from project root:
    streamlit run app.py
"""

import sys
import base64
from pathlib import Path
import streamlit as st

_LOGOS_DIR = Path(__file__).resolve().parent / "assets" / "logos"

def _load_logo(filename: str) -> str:
    """Return a base64 data URI for a logo file, or empty string if not found."""
    p = _LOGOS_DIR / filename
    if not p.exists():
        return ""
    ext = p.suffix.lower()
    mime = "image/svg+xml" if ext == ".svg" else "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent / "src" / "generation"))
sys.path.append(str(Path(__file__).resolve().parent / "src" / "ingestion_retrieval"))

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="FinRAG Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@300;400;500;600;700;800&family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #0a0c0f;
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem !important;
    max-width: 100% !important;
}

[data-testid="stColumn"] {
    padding: 0 1rem !important;
}

/* ── Grid layout ── */
.finrag-layout {
    display: grid;
    grid-template-columns: 340px 1fr;
    min-height: 100vh;
}

/* ── Left panel ── */
.left-panel {
    background: #0d1017;
    border-right: 1px solid #1e2530;
    padding: 2.5rem 2rem;
    display: flex;
    flex-direction: column;
    gap: 2rem;
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    width: 340px;
    overflow-y: auto;
}

.logo-area {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.logo-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, #1a2332, #0f1923);
    border: 1px solid #2a3a50;
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    width: fit-content;
}

.logo-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #3b82f6;
    box-shadow: 0 0 8px #3b82f6;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px #3b82f6; }
    50% { opacity: 0.6; box-shadow: 0 0 3px #3b82f6; }
}

.logo-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #3b82f6;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.logo-title {
    font-family: 'Figtree', sans-serif;
    font-size: 2rem;
    font-weight: 600;
    color: #f0f4f8;
    line-height: 1.1;
    letter-spacing: -0.01em;
}

.logo-title em {
    color: #3b82f6;
    font-style: italic;
}

.logo-sub {
    font-size: 0.8rem;
    color: #4a5568;
    line-height: 1.5;
    margin-top: 0.25rem;
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.stat-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0.8rem;
    background: #111720;
    border: 1px solid #1a2230;
    border-radius: 8px;
}

.stat-label {
    font-size: 0.72rem;
    color: #4a5568;
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stat-value {
    font-size: 0.8rem;
    color: #a0aec0;
    font-weight: 500;
}

/* ── Corpus section ── */
.corpus-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.75rem;
}

.company-pill {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.6rem 0.8rem;
    background: #111720;
    border: 1px solid #1a2230;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}

.company-pill:hover { border-color: #2a3a50; }

.company-badge {
    width: 26px;
    height: 26px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    flex-shrink: 0;
    font-family: 'DM Sans', sans-serif;
}

.company-name {
    font-size: 0.8rem;
    color: #a0aec0;
    font-weight: 500;
}

.company-docs {
    font-size: 0.7rem;
    color: #4a5568;
    margin-left: auto;
    font-family: 'DM Mono', monospace;
}

/* ── Suggested questions ── */
.suggest-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.75rem;
}

/* ── Main content ── */
.main-content {
    margin-left: 340px;
    padding: 3rem 4rem;
    min-height: 100vh;
}

/* ── Hero section ── */
.hero {
    margin-bottom: 3rem;
}

.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.75rem;
}

.hero-heading {
    font-family: 'Figtree', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #f0f4f8;
    line-height: 1.15;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
}

.hero-heading span {
    color: #3b82f6;
}

.hero-sub {
    font-size: 1rem;
    color: #4a5568;
    max-width: 520px;
    line-height: 1.6;
}

/* ── Input area ── */
.stTextArea {
    margin-bottom: 1rem;
}

/* Style the BaseWeb wrapper (the actual visible box) */
.stTextArea [data-baseweb="textarea"] {
    background: #0d1017 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 14px !important;
}

.stTextArea [data-baseweb="textarea"]:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.08) !important;
}

/* Override Streamlit text area */
.stTextArea textarea {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    resize: none !important;
    box-shadow: none !important;
    padding: 0.75rem 1rem !important;
}

.stTextArea textarea::placeholder { color: #4a5568 !important; }
.stTextArea textarea:focus { outline: none !important; box-shadow: none !important; border: none !important; }

/* ── Button ── */
.stButton > button {
    background: #3b82f6 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: #2563eb !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3) !important;
}

.stButton > button:active { transform: translateY(0) !important; }

/* ── Suggested question buttons ── */
.suggest-btn > button {
    background: #111720 !important;
    color: #6b7280 !important;
    border: 1px solid #1a2230 !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.9rem !important;
    font-size: 0.75rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    width: 100% !important;
    margin-bottom: 0.4rem !important;
    transition: all 0.15s !important;
}

.suggest-btn > button:hover {
    background: #151e2b !important;
    color: #a0aec0 !important;
    border-color: #2a3a50 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Answer card ── */
.answer-card {
    background: #0d1017;
    border: 1px solid #1e2530;
    border-radius: 14px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.answer-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #60a5fa, transparent);
}

.answer-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
}

.answer-text {
    font-size: 0.95rem;
    color: #cbd5e0;
    line-height: 1.75;
}

/* ── Sources section ── */
.sources-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.sources-count {
    background: #1a2230;
    color: #4a5568;
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    font-size: 0.6rem;
}

.source-card {
    background: #080c11;
    border: 1px solid #1a2230;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 0.75rem;
    align-items: start;
    transition: border-color 0.2s;
}

.source-card:hover { border-color: #2a3a50; }

.source-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #3b82f6;
    background: rgba(59, 130, 246, 0.1);
    border-radius: 5px;
    padding: 0.2rem 0.5rem;
    margin-top: 0.1rem;
}

.source-meta {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}

.source-title {
    font-size: 0.82rem;
    color: #a0aec0;
    font-weight: 500;
}

.source-preview {
    font-size: 0.75rem;
    color: #4a5568;
    font-family: 'DM Mono', monospace;
    line-height: 1.5;
}

.source-score {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #4a5568;
    white-space: nowrap;
}

.score-high { color: #22c55e; }
.score-mid  { color: #eab308; }
.score-low  { color: #ef4444; }

.type-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.type-table {
    background: rgba(234, 179, 8, 0.1);
    color: #eab308;
    border: 1px solid rgba(234, 179, 8, 0.2);
}

.type-text {
    background: rgba(59, 130, 246, 0.1);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.2);
}

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 2rem;
    text-align: center;
    color: #2d3748;
}

.empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.3;
}

.empty-text {
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── Loading ── */
.stSpinner > div {
    border-color: #3b82f6 transparent transparent transparent !important;
}

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #1a2230;
    margin: 1.5rem 0;
}

/* ── Question display ── */
.question-display {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #e2e8f0;
    margin-bottom: 1.5rem;
    line-height: 1.4;
    padding-left: 1rem;
    border-left: 3px solid #3b82f6;
}
</style>
""", unsafe_allow_html=True)


# ── Load model once ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    from generator import _model, answer as _answer
    return _model, _answer


# ── Suggested questions ────────────────────────────────────────────────────────
SUGGESTED = [
    "What was Apple's total revenue in 2024?",
    "What are Tesla's main risk factors?",
    "What is JPMorgan's net income?",
    "How does Apple describe its competition?",
    "What is Tesla's capital expenditure strategy?",
    "How does JPMorgan manage credit risk?",
]

COMPANIES = [
    {"name": "Apple",    "docs": "10-K · 10-Q", "badge": "A",  "bg": "#1d1d1f", "fg": "#f5f5f7", "logo": _load_logo("apple.png")},
    {"name": "Tesla",    "docs": "10-K · 10-Q", "badge": "T",  "bg": "#cc0000", "fg": "#ffffff",  "logo": _load_logo("tesla.png")},
    {"name": "JPMorgan", "docs": "10-K · 10-Q", "badge": "JP", "bg": "#003087", "fg": "#ffffff",  "logo": _load_logo("jpmorgan.png")},
]


# ── Session state ──────────────────────────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "loading" not in st.session_state:
    st.session_state.loading = False


# ── LEFT PANEL ─────────────────────────────────────────────────────────────────
with st.sidebar:
    pass  # we build the sidebar via HTML below

# We use columns to simulate the fixed left panel
left, right = st.columns([1.05, 3], gap="large")

with left:
    # Logo
    st.markdown("""
    <div class="logo-area" style="margin-bottom:1.5rem">
        <div class="logo-badge">
            <div class="logo-dot"></div>
            <span class="logo-label">RAG System · Live</span>
        </div>
        <div class="logo-title">Fin<em>RAG</em><br>Analyst</div>
        <div class="logo-sub">Financial document intelligence powered by retrieval-augmented generation.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Corpus
    st.markdown('<div class="corpus-title">Corpus</div>', unsafe_allow_html=True)
    for c in COMPANIES:
        icon = (f'<img src="{c["logo"]}" style="width:24px;height:24px;border-radius:5px;object-fit:contain;flex-shrink:0">'
                if c["logo"] else
                f'<div class="company-badge" style="background:{c["bg"]};color:{c["fg"]}">{c["badge"]}</div>')
        st.markdown(f"""
        <div class="company-pill">
            {icon}
            <span class="company-name">{c['name']}</span>
            <span class="company-docs">{c['docs']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Suggested questions
    st.markdown('<div class="suggest-title">Suggested questions</div>', unsafe_allow_html=True)
    for q in SUGGESTED:
        with st.container():
            st.markdown('<div class="suggest-btn">', unsafe_allow_html=True)
            if st.button(q, key=f"suggest_{q[:20]}"):
                st.session_state.query = q
                with st.spinner("Retrieving and generating answer..."):
                    try:
                        _, answer_fn = load_resources()
                        st.session_state.result = answer_fn(q, top_k=5)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.session_state.result = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Model info
    st.markdown("""
    <div class="stats-bar">
        <div class="stat-item">
            <span class="stat-label">Embedding</span>
            <span class="stat-value">all-mpnet-v2</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Generator</span>
            <span class="stat-value">GPT-4o mini</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Vector store</span>
            <span class="stat-value">FAISS</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Chunking</span>
            <span class="stat-value">Structural</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── RIGHT PANEL — Main content ─────────────────────────────────────────────────
with right:
    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">DATA522 · FinRAG Analyst</div>
        <div class="hero-heading">Ask anything about<br><span>financial filings</span></div>
        <div class="hero-sub">Query SEC filings from Apple, Tesla, and JPMorgan. Every answer is grounded in retrieved document passages with full source attribution.</div>
    </div>
    """, unsafe_allow_html=True)

    # Input
    query_input = st.text_area(
        label="Query",
        value=st.session_state.query,
        placeholder="e.g. What was Apple's gross margin in FY2024?",
        height=80,
        key="query_input",
        label_visibility="collapsed",
    )
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn2:
        ask_clicked = st.button("Ask FinRAG →", type="primary", use_container_width=True)

    # Run query
    if ask_clicked and query_input.strip():
        st.session_state.query = query_input.strip()
        with st.spinner("Retrieving and generating answer..."):
            try:
                _, answer_fn = load_resources()
                result = answer_fn(query_input.strip(), top_k=5)
                st.session_state.result = result
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.result = None

    # Display result
    if st.session_state.result:
        result = st.session_state.result

        # Question
        st.markdown(
            f'<div class="question-display">{result["question"]}</div>',
            unsafe_allow_html=True
        )

        # Answer card
        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-label">▎ Answer</div>
            <div class="answer-text">{result["answer"].replace(chr(10), "<br>")}</div>
        </div>
        """, unsafe_allow_html=True)

        # Sources
        sources = result["sources"]
        st.markdown(f"""
        <div class="sources-header">
            Retrieved sources
            <span class="sources-count">{len(sources)}</span>
        </div>
        """, unsafe_allow_html=True)

        for s in sources:
            score = s["score"]
            score_class = "score-high" if score > 0.72 else "score-mid" if score > 0.55 else "score-low"
            type_class = "type-table" if s["element_type"] == "table" else "type-text"
            section = s["section_title"] if s["section_title"] != "N/A" else "—"

            st.markdown(f"""
            <div class="source-card">
                <div class="source-num">{s['source_num']}</div>
                <div class="source-meta">
                    <div class="source-title">
                        {s['company']} &nbsp;·&nbsp; {s['doc_type']} {s['period']}
                        &nbsp; <span class="type-badge {type_class}">{s['element_type']}</span>
                    </div>
                    <div class="source-preview">{s['text_preview']}</div>
                    <div style="font-size:0.7rem;color:#2d3748;margin-top:0.2rem">
                        Section: {section}
                    </div>
                </div>
                <div class="source-score {score_class}">{score}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📄</div>
            <div class="empty-text">
                Ask a question above or pick one<br>from the suggested list on the left.
            </div>
        </div>
        """, unsafe_allow_html=True)
