# src/synthesize.py
import heapq
import re
import os
from datetime import datetime
from typing import List, Dict

def split_sentences(text: str) -> List[str]:
    s = re.split(r'(?<=[\.\?\!])\s+', text)
    return [seg.strip() for seg in s if seg.strip()]

def synthesize_answer(original_query: str, retrieved_chunks: List[Dict]) -> Dict:
    """
    Produce a short extractive summary from retrieved chunks.
    Returns a dict: {"summary", "citations", "confidence"}.
    """
    if not retrieved_chunks:
        return {"summary": "No relevant information found.", "citations": [], "confidence": 0.0}

    # Sort chunks by score desc and take top N
    sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.get("score", 0.0), reverse=True)[:8]
    combined_text = " ".join([c["text"] for c in sorted_chunks])

    # simple scoring of sentences by token overlap with query
    q_tokens = set(re.findall(r'\w+', (original_query or "").lower()))
    sentences = split_sentences(combined_text)
    scored = []
    for s in sentences:
        tokens = set(re.findall(r'\w+', s.lower()))
        # score = number of shared tokens; add tiny bias for longer sentences
        score = len(tokens & q_tokens) + min(1, len(s)) / 100.0
        scored.append((score, s))

    top = heapq.nlargest(5, scored, key=lambda x: x[0]) if scored else []
    summary = " ".join([t[1] for t in top]).strip()
    if not summary:
        summary = (combined_text[:800] + ("..." if len(combined_text) > 800 else ""))

    # build citations (unique)
    sources = []
    for c in sorted_chunks:
        tag = f"{c.get('source','unknown')} ({c.get('chunk_id','?')})"
        if tag not in sources:
            sources.append(tag)

    # confidence = mean of scores
    scores = [float(c.get("score", 0.0)) for c in sorted_chunks]
    confidence = float(sum(scores) / len(scores)) if scores else 0.0

    return {"summary": summary, "citations": sources, "confidence": confidence}

def export_answer_markdown(query: str, trace: List[str], answer: Dict, exports_dir: str = "exports") -> str:
    """
    Save the answer + trace + citations to a Markdown file under exports/.
    Returns the file path.
    """
    os.makedirs(exports_dir, exist_ok=True)
    t = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_name = f"report_{t}.md"
    fname = os.path.join(exports_dir, safe_name)
    with open(fname, "w", encoding="utf8") as f:
        f.write(f"# Research Report\n\n")
        f.write(f"**Query:** {query}\n\n")
        f.write("## Reasoning Trace\n\n")
        for step in (trace or []):
            f.write(f"- {step}\n")
        f.write("\n## Answer\n\n")
        f.write(answer.get("summary", "") + "\n\n")
        f.write(f"**Confidence:** {answer.get('confidence', 0.0):.3f}\n\n")
        f.write("## Citations\n\n")
        for c in answer.get("citations", []):
            f.write(f"- {c}\n")
    return fname

# Explicit exports
__all__ = ["synthesize_answer", "export_answer_markdown"]
