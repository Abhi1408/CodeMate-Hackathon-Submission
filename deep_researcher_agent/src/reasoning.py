# src/reasoning.py
"""
Query decomposition utilities.

The Streamlit UI (and main.py) imports `decompose_query_with_trace`.
We provide that function here and also keep a backward-compatible
alias `decompose_query` in case older code imports that name.
"""

import re
from typing import Tuple, List

def decompose_query_with_trace(query: str) -> Tuple[List[str], List[str]]:
    """
    Returns (sub_queries_list, trace_list).
    Trace contains human-readable steps describing decomposition.
    """
    q = (query or "").strip()
    if not q:
        return [], ["Empty query"]

    trace = []
    # Step 1: basic split on semicolons
    parts = [p.strip() for p in re.split(r'\s*;\s*', q) if p.strip()]
    if len(parts) > 1:
        trace.append(f"Split query on ';' into {len(parts)} parts.")
        for i, p in enumerate(parts, 1):
            trace.append(f"Sub-query {i}: {p}")
        return parts, trace

    low = q.lower()
    # Step 2: split on ' and ' where it looks meaningful
    if ' and ' in low:
        parts = [p.strip() for p in re.split(r'\band\b', q, flags=re.IGNORECASE) if p.strip()]
        # only treat as split if each part is at least a short phrase
        if len(parts) > 1 and all(len(p.split()) > 2 for p in parts):
            trace.append(f"Detected 'and' splitting into {len(parts)} sub-queries.")
            for i, p in enumerate(parts, 1):
                trace.append(f"Sub-query {i}: {p}")
            return parts, trace

    # Step 3: handle 'vs' or 'versus' -> comparison decomposition
    if ' vs ' in low or ' versus ' in low:
        parts = [p.strip() for p in re.split(r'\bvs\b|\bversus\b', q, flags=re.IGNORECASE) if p.strip()]
        trace.append("Detected comparison; decomposed into comparison parts.")
        for i, p in enumerate(parts, 1):
            trace.append(f"Compare part {i}: {p}")
        return parts, trace

    # Default: single query, no decomposition
    trace.append("No decomposition applied; treating as single query.")
    return [q], trace

# Backwards compatibility: some code expected decompose_query
def decompose_query(query: str):
    """Compatibility wrapper — returns just the list of sub-queries (no trace)."""
    subs, _ = decompose_query_with_trace(query)
    return subs

# Expose names explicitly for 'from reasoning import ...' style imports
__all__ = ["decompose_query_with_trace", "decompose_query"]
