# src/retrieve.py
"""
Simple retrieval wrapper used by main.py / app_streamlit.py
"""

from typing import List, Dict

def query_index(query: str, index_obj, k: int = 5) -> List[Dict]:
    """
    Query the provided IndexObject and return a list of hits.
    Each hit dict contains: score, source, chunk_id, text
    """
    if index_obj is None:
        raise ValueError("Index object is None")
    # index_obj.search handles text and vectors and checks k <= n_samples
    hits = index_obj.search(query, top_k=k)
    return hits
