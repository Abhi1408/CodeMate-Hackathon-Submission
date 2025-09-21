# src/ingest_and_chunk.py
"""
Recursive ingestion of .txt, .md, .pdf, and .docx files under data/.
Splits into overlapping word chunks.
"""

import re
from pathlib import Path
from typing import List, Dict

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def chunk_text(text: str, chunk_size_words: int = 300, overlap: int = 60) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size_words - overlap)
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size_words])
        chunks.append(chunk)
        i += step
    return chunks


def extract_text_from_pdf(path: Path) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed")
    all_text = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                text = p.extract_text() or ""
                all_text.append(text)
    except Exception as e:
        print(f"[WARN] Failed to read PDF {path}: {e}")
    return "\n".join(all_text)


def extract_text_from_docx(path: Path) -> str:
    if docx is None:
        raise RuntimeError("python-docx not installed")
    try:
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"[WARN] Failed to read DOCX {path}: {e}")
        return ""


def ingest_folder(folder_path: str) -> List[Dict]:
    """
    Recursively read all .txt, .md, .pdf, and .docx files inside folder_path.
    Returns a list of chunk dicts with keys: source, chunk_id, text.
    """
    docs = []
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Data folder not found: {folder}")

    file_paths = sorted([p for p in folder.rglob("*") if p.is_file()])
    for p in file_paths:
        fname = p.name
        lower = fname.lower()

        try:
            if lower.endswith(".pdf"):
                if pdfplumber is None:
                    print(f"[INFO] Skipping PDF (pdfplumber not installed): {fname}")
                    continue
                raw = extract_text_from_pdf(p)
            elif lower.endswith(".docx"):
                if docx is None:
                    print(f"[INFO] Skipping DOCX (python-docx not installed): {fname}")
                    continue
                raw = extract_text_from_docx(p)
            elif lower.endswith((".txt", ".md")):
                raw = p.read_text(encoding="utf8", errors="ignore")
            else:
                continue
        except Exception as e:
            print(f"[WARN] Could not read {fname}: {e}")
            continue

        text = normalize_whitespace(raw)
        if not text:
            continue

        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            docs.append(
                {
                    "source": str(p.relative_to(folder)),
                    "chunk_id": f"{fname}::c{idx}",
                    "text": chunk,
                }
            )
    return docs
