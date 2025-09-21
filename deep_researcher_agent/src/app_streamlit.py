# src/app_streamlit.py
"""
Streamlit UI for Deep Researcher Agent (local-only)
Run:
    pip install streamlit
    streamlit run src/app_streamlit.py
"""

import streamlit as st
import os, sys, time, io
from pathlib import Path
from typing import List
import re
import base64

# Ensure local src is importable (robust)
try:
    ROOT = Path(__file__).resolve().parents[1]
except Exception:
    ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Local modules (must exist in src/)
from ingest_and_chunk import ingest_folder
from embed_index import build_or_load_index
from retrieve import query_index
from reasoning import decompose_query_with_trace
from synthesize import synthesize_answer, export_answer_markdown

DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORTS_DIR = ROOT / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Deep Researcher (Local)", layout="wide")

# ---- Helper utilities ----
def highlight_text(text: str, query: str):
    """Simple highlighting of query tokens in text (HTML)."""
    if not query:
        return text
    tokens = set(re.findall(r'\w+', query.lower()))
    def repl(m):
        w = m.group(0)
        if w.lower() in tokens:
            return f"<mark>{w}</mark>"
        return w
    return re.sub(r'\w+', repl, text)

def render_retrieved_chunks(chunks: List[dict], page: int = 0, per_page: int = 5, query: str = ""):
    start = page * per_page
    end = start + per_page
    for i, c in enumerate(chunks[start:end], start+1):
        with st.expander(f"{i}. {c['source']} — {c['chunk_id']} — score: {c['score']:.3f}", expanded=False):
            st.write("Chunk text:")
            st.markdown(highlight_text(c['text'], query), unsafe_allow_html=True)
            st.write("")
            st.write(f"Source: **{c['source']}** | Chunk ID: `{c['chunk_id']}` | Score: {c['score']:.3f}")

def save_uploaded_file(uploaded):
    """Save uploaded files into data/uploads and return saved path."""
    fname = uploaded.name
    save_path = UPLOAD_DIR / fname
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())
    return save_path

def build_index_and_attach_state(docs, use_semantic=False):
    """Wrapper to build index and store in session_state."""
    st.session_state["status"] = "Building index..."
    st.experimental_rerun_token = time.time()
    idx_obj = build_or_load_index(docs)
    # attach into session
    st.session_state["index_obj"] = idx_obj
    st.session_state["last_build_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["status"] = "Index ready"
    return idx_obj

# ---- Sidebar controls ----
st.sidebar.title("Controls")
with st.sidebar.form(key="settings"):
    top_k = st.number_input("top_k (neighbors)", min_value=1, max_value=20, value=5, step=1)
    per_page = st.number_input("results per page", min_value=1, max_value=10, value=5, step=1)
    chunk_size = st.number_input("chunk_size (words) — requires rebuild", min_value=50, max_value=2000, value=300, step=50)
    use_semantic = st.checkbox("Attempt semantic model (sentence-transformers)", value=False)
    rebuild = st.form_submit_button("Rebuild index (re-ingest + index)")

st.sidebar.markdown("---")
st.sidebar.markdown("Upload documents (.txt, .md, .pdf):")
uploaded = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
if uploaded:
    saved = []
    for u in uploaded:
        sp = save_uploaded_file(u)
        saved.append(sp.name)
    st.sidebar.success(f"Saved: {', '.join(saved)}")
    # flag to rebuild
    st.sidebar.info("Now click 'Rebuild index' to include uploaded files in the index.")

st.sidebar.markdown("---")
st.sidebar.write("Index status:")
st.sidebar.write(st.session_state.get("status", "Not built"))
st.sidebar.write("Last build:", st.session_state.get("last_build_time", "N/A"))

# ---- Main UI ----
st.title("Deep Researcher — Local (Streamlit)")
st.markdown("Search your local documents using local embeddings or TF-IDF. Everything runs locally; no external APIs required.")

col1, col2 = st.columns([3,1])

with col1:
    query_text = st.text_input("Enter research query", value="", key="query_input")
    do_search = st.button("Search")
    if st.button("Quick example: climate effects"):
        query_text = "What are the effects of climate change?"
        st.session_state["query_input"] = query_text
        do_search = True

with col2:
    st.write("Index stats")
    idx_obj = st.session_state.get("index_obj", None)
    if idx_obj is None:
        st.info("Index not built yet. Click Rebuild index in sidebar to build from data/")
    else:
        embs = getattr(idx_obj, "embeddings", None)
        if embs is None:
            st.write("embeddings: None")
        else:
            st.write(f"chunks: {embs.shape[0]}")
            st.write(f"embedding dim: {embs.shape[1]}")
            st.write(f"vectorizer present: {bool(getattr(idx_obj, 'vectorizer', None))}")

# Build index on first run if not present
if "index_obj" not in st.session_state:
    # Try to load or build automatically from data/ if files exist
    try:
        docs = ingest_folder(str(DATA_DIR))
        if docs:
            idx = build_or_load_index(docs)
            st.session_state["index_obj"] = idx
            st.session_state["status"] = "Index loaded"
            st.session_state["last_build_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        st.error("Initial index build failed: " + str(e))

# Rebuild index button action
if rebuild:
    try:
        # ingest fresh from data/ (includes uploads)
        docs = ingest_folder(str(DATA_DIR))
        # If chunk_size changed, update ingest function? Here we cannot change chunk size without editing code,
        # but we show a message. (To actually change, update ingest_and_chunk.chunk_text signature or file.)
        st.session_state["status"] = "Building index..."
        idx = build_or_load_index(docs)
        st.session_state["index_obj"] = idx
        st.success("Index rebuilt.")
        st.session_state["last_build_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        st.error("Rebuild failed: " + str(e))

# --- Perform search ---
results = []
trace = []
if do_search and query_text:
    if "index_obj" not in st.session_state:
        st.error("Index not ready. Rebuild index first.")
    else:
        idx = st.session_state["index_obj"]
        sub_qs, trace = decompose_query_with_trace(query_text)
        st.markdown("**Reasoning Trace:**")
        for t in trace:
            st.write("-", t)

        all_hits = []
        for sub in sub_qs:
            hits = query_index(sub, idx, k=top_k)
            all_hits.extend(hits)

        # sort and dedupe by chunk_id (keep best score)
        seen = {}
        for h in all_hits:
            cid = (h["source"], h["chunk_id"])
            if cid not in seen or h["score"] > seen[cid]["score"]:
                seen[cid] = h
        results = sorted(list(seen.values()), key=lambda x: x["score"], reverse=True)

        # put into session for follow-ups and paging
        st.session_state["last_results"] = results
        st.session_state["last_query"] = query_text
        st.session_state["last_trace"] = trace

# If results present, display and let user synthesize/export/refine
if st.session_state.get("last_results"):
    results = st.session_state["last_results"]
    query_display = st.session_state.get("last_query", "")
    trace = st.session_state.get("last_trace", [])
    st.subheader("Retrieved results")
    # Pagination
    page = st.number_input("Page", min_value=0, value=0, step=1)
    render_retrieved_chunks(results, page=page, per_page=per_page, query=query_display)

    # Synthesize section
    if st.button("Synthesize Answer"):
        ans = synthesize_answer(query_display, results)
        st.subheader("Synthesis")
        st.write(ans["summary"])
        st.write(f"Confidence: {ans['confidence']:.3f}")
        st.write("Citations:")
        for c in ans["citations"]:
            st.write("-", c)
        # export UI
        md = export_answer_markdown(query_display, trace, ans)
        with open(md, "r", encoding="utf8") as f:
            md_text = f.read()
        b64 = base64.b64encode(md_text.encode()).decode()
        st.download_button("Download report (.md)", data=md_text, file_name=Path(md).name, mime="text/markdown")
        st.success(f"Report saved to {md}")

    # Follow-up restricted to retrieved context
    if st.checkbox("Ask follow-up on retrieved context"):
        follow = st.text_input("Follow-up question (restricted search)", key="follow_input")
        if st.button("Run follow-up"):
            # Build temporary index from results (small)
            temp_docs = [{"source": r["source"], "chunk_id": r["chunk_id"], "text": r["text"]} for r in results]
            temp_idx = build_or_load_index(temp_docs)
            f_hits = query_index(follow, temp_idx, k=top_k)
            st.write("Follow-up retrieval:")
            for fh in f_hits:
                st.write(f"- {fh['source']} | {fh['chunk_id']} | score {fh['score']:.3f}")
            f_ans = synthesize_answer(follow, f_hits)
            st.subheader("Follow-up Answer")
            st.write(f_ans["summary"])
            st.write(f"Confidence: {f_ans['confidence']:.3f}")

# Footer / help
st.markdown("---")
st.markdown("**Notes:** This UI runs locally. For semantic embeddings (better results) install `sentence-transformers`.")
st.markdown("To include uploaded files, upload and then click 'Rebuild index' in the sidebar.")
