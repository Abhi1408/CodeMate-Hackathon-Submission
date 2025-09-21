# main.py
"""
Deep Researcher Agent — MVP with advanced features:
- PDF+TXT ingestion
- Multi-step reasoning with trace
- Interactive query refinement (follow-ups)
- Export results to Markdown
Run: python main.py
"""

import sys, os
from pathlib import Path
import time

# Ensure src/ on path (robust)
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
    print(f"[INFO] Added src to sys.path: {SRC_PATH}")

# imports from src
from ingest_and_chunk import ingest_folder
from embed_index import build_or_load_index
from reasoning import decompose_query_with_trace
from retrieve import query_index
from synthesize import synthesize_answer, export_answer_markdown

def interactive_loop(index_obj):
    """
    Interactive query loop with refinement option.
    Keeps a short local context of retrieved chunks for refinement queries.
    """
    local_context = []  # aggregated retrieved chunks (list of dicts)
    while True:
        q = input("\nEnter research query (or 'exit'): ").strip()
        if not q:
            print("Type a query or 'exit'.")
            continue
        if q.lower() == "exit":
            break

        # Decompose with trace (returns list + trace steps)
        sub_qs, trace = decompose_query_with_trace(q)
        print("\n[Reasoning Trace]")
        for t in trace:
            print("-", t)

        # Retrieve for each sub-query
        all_hits = []
        for sub in sub_qs:
            hits = query_index(sub, index_obj, k=5)
            print(f"[Retrieve] {len(hits)} hits for sub-query: {sub}")
            all_hits.extend(hits)

        # Add to local context (for refinement)
        local_context.extend(all_hits)

        # Synthesize answer (uses all_hits)
        answer = synthesize_answer(q, all_hits)
        print("\n=== Final Answer ===\n")
        print(answer["summary"])
        print(f"\n[Confidence ≈ {answer['confidence']:.2f}]")
        if answer.get("citations"):
            print("\nCitations:")
            for c in answer["citations"]:
                print("-", c)

        # Export option
        do_export = input("\nExport this result to Markdown? (y/N): ").strip().lower()
        if do_export == "y":
            fname = export_answer_markdown(q, trace, answer)
            print(f"[EXPORT] Saved report to: {fname}")

        # Refinement option: allow follow-up restricted to local_context
        ref = input("\nAsk a follow-up on the retrieved context? (y/N): ").strip().lower()
        if ref == "y":
            follow = input("Enter follow-up question (will search within previously retrieved chunks): ").strip()
            if follow:
                # Build a temporary index-like object from local_context
                # (We reuse embed_index.IndexObject via build_or_load_index by passing docs)
                tmp_index = build_or_load_index([{"source":h["source"], "chunk_id":h["chunk_id"], "text":h["text"]} for h in local_context])
                hits = query_index(follow, tmp_index, k=5)
                print(f"[Refine] {len(hits)} hits (from context):")
                ans = synthesize_answer(follow, hits)
                print("\nRefined Answer:\n", ans["summary"])
        # Continue loop

def main():
    print("=== Deep Researcher Agent (Advanced MVP) ===")
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        print("[ERROR] Please create a 'data' folder with .txt/.md/.pdf files and re-run.")
        return

    # Ingest
    start = time.time()
    docs = ingest_folder(str(data_dir))
    print(f"[INFO] Ingested {len(docs)} chunks from data/ (took {time.time()-start:.2f}s)")

    # Build or load index
    index_obj = build_or_load_index(docs)
    if index_obj is None:
        print("[ERROR] Index build failed.")
        return

    # enter interactive loop
    interactive_loop(index_obj)

    print("\nGoodbye — remember to submit the entire folder with main.py at the root.")

if __name__ == "__main__":
    main()
