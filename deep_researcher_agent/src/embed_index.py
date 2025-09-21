# src/embed_index.py
"""
Index builder / loader that supports:
 - sentence-transformers (semantic embeddings) if installed
 - fallback to sklearn TfidfVectorizer
Stores:
 - exports/embeddings.npy
 - exports/metadata.pkl
 - exports/vectorizer.pkl  (for TF-IDF)
 - exports/index_meta.pkl  (mode, created_at)
Provides IndexObject with .search(query_or_vector, top_k)
"""

import os
import time
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# sklearn utilities
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# optional semantic encoder
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


EXPORTS = Path("exports")
EXPORTS.mkdir(exist_ok=True)

EMB_PATH = EXPORTS / "embeddings.npy"
META_PATH = EXPORTS / "metadata.pkl"
VECT_PATH = EXPORTS / "vectorizer.pkl"
INDEX_META_PATH = EXPORTS / "index_meta.pkl"


class IndexObject:
    def __init__(self,
                 embeddings: np.ndarray,
                 metadata: List[Dict],
                 mode: str,
                 vectorizer=None,
                 encoder_name: Optional[str]=None):
        """
        embeddings: np.array shape (N, D)
        metadata: list of dicts (must include 'source','chunk_id','text')
        mode: 'semantic' or 'tfidf'
        vectorizer: TF-IDF vectorizer (if mode == 'tfidf'), else None
        encoder_name: model name if semantic
        """
        self.embeddings = embeddings.astype(np.float32)
        self.metadata = metadata
        self.mode = mode
        self.vectorizer = vectorizer
        self.encoder_name = encoder_name

        # Build NearestNeighbors index
        # Use cosine distance metric for similarity searching
        n_samples = max(1, self.embeddings.shape[0])
        # Use algorithm='brute' for cosine with small data; can change if needed
        self._nn = NearestNeighbors(n_neighbors=min(10, n_samples), metric='cosine', algorithm='brute')
        if n_samples > 0:
            self._nn.fit(self.embeddings)

    def encode_query(self, q: str) -> np.ndarray:
        if self.mode == "semantic":
            # SentenceTransformer encode -> shape (1, D)
            model = SentenceTransformer(self.encoder_name) if self.encoder_name and SentenceTransformer else None
            if model is None:
                raise RuntimeError("Semantic encoder not available")
            vec = model.encode([q], show_progress_bar=False)
            return np.array(vec, dtype=np.float32)
        else:
            # TF-IDF
            if self.vectorizer is None:
                raise RuntimeError("TF-IDF vectorizer missing")
            vec = self.vectorizer.transform([q]).toarray().astype(np.float32)
            return vec

    def search(self, query_or_vec, top_k: int = 5) -> List[Dict]:
        """
        Query can be text string or a vector (1,D).
        Returns list of dicts: {'score': float, 'source','chunk_id','text'}
        Score is similarity in [0,1] (1 best).
        """
        # Create query vector
        if isinstance(query_or_vec, str):
            qvec = self.encode_query(query_or_vec)
        else:
            qvec = np.asarray(query_or_vec, dtype=np.float32)
            if qvec.ndim == 1:
                qvec = qvec.reshape(1, -1)

        # Ensure dims match otherwise raise informative error
        if qvec.shape[1] != self.embeddings.shape[1]:
            raise ValueError(f"Query vector dimension {qvec.shape[1]} != index embedding dim {self.embeddings.shape[1]}")

        n_samples = max(1, self.embeddings.shape[0])
        k = max(1, min(top_k, n_samples))
        # Kneighbors expects n_neighbors <= n_samples
        # If index was built with fewer neighbors, sklearn will still accept n_neighbors <= n_samples
        dists, idxs = self._nn.kneighbors(qvec, n_neighbors=k)
        dists = np.asarray(dists)
        idxs = np.asarray(idxs)
        # cosine distance -> similarity
        sims = 1.0 - dists[0]
        results = []
        for sim, idx in zip(sims.tolist(), idxs[0].tolist()):
            m = self.metadata[idx]
            results.append({
                "score": float(sim),
                "source": m.get("source"),
                "chunk_id": m.get("chunk_id"),
                "text": m.get("text")
            })
        return results


def build_embeddings_semantic(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, show_progress_bar=True)
    return np.array(vecs, dtype=np.float32), model_name


def build_embeddings_tfidf(texts: List[str]) -> (np.ndarray, TfidfVectorizer):
    vect = TfidfVectorizer(max_features=4000, stop_words='english')
    X = vect.fit_transform(texts)  # sparse
    # convert to dense (ok for small datasets) to feed into NearestNeighbors
    arr = X.toarray().astype(np.float32)
    return arr, vect


def save_index(embeddings: np.ndarray, metadata: List[Dict], mode: str, vectorizer=None, encoder_name: Optional[str]=None):
    np.save(EMB_PATH, embeddings)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    meta = {"mode": mode, "created_at": time.time(), "encoder_name": encoder_name}
    with open(INDEX_META_PATH, "wb") as f:
        pickle.dump(meta, f)
    if mode == "tfidf" and vectorizer is not None:
        with open(VECT_PATH, "wb") as f:
            pickle.dump(vectorizer, f)


def load_index() -> Optional[IndexObject]:
    if not EMB_PATH.exists() or not META_PATH.exists() or not INDEX_META_PATH.exists():
        return None
    embeddings = np.load(EMB_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    with open(INDEX_META_PATH, "rb") as f:
        meta = pickle.load(f)
    mode = meta.get("mode", "tfidf")
    encoder_name = meta.get("encoder_name")
    vectorizer = None
    if mode == "tfidf":
        if VECT_PATH.exists():
            with open(VECT_PATH, "rb") as f:
                vectorizer = pickle.load(f)
    idxobj = IndexObject(embeddings=embeddings, metadata=metadata, mode=mode, vectorizer=vectorizer, encoder_name=encoder_name)
    return idxobj


def build_or_load_index(docs: List[Dict], prefer_semantic: bool = True, force_rebuild: bool = False) -> IndexObject:
    """
    docs: list of dicts containing keys: 'source','chunk_id','text'
    prefer_semantic: if True and sentence-transformers installed, uses semantic model
    force_rebuild: if True, ignore saved exports and rebuild from docs
    """
    # Validate input
    texts = [d["text"] for d in docs]
    metadata = docs

    # If saved index exists and not forced, try load and verify counts and dims
    if not force_rebuild:
        try:
            idx = load_index()
            if idx is not None:
                # verify same number of samples
                if len(idx.metadata) == len(metadata):
                    # OK return idx
                    return idx
                else:
                    # mismatch -> rebuild
                    pass
        except Exception:
            pass

    # Build fresh
    use_semantic = prefer_semantic and (SentenceTransformer is not None)
    if use_semantic:
        try:
            embeddings, encoder_name = build_embeddings_semantic(texts)
            mode = "semantic"
            vectorizer = None
        except Exception:
            # fallback
            embeddings, vectorizer = build_embeddings_tfidf(texts)
            mode = "tfidf"
            encoder_name = None
    else:
        embeddings, vectorizer = build_embeddings_tfidf(texts)
        mode = "tfidf"
        encoder_name = None

    save_index(embeddings, metadata, mode=mode, vectorizer=vectorizer, encoder_name=encoder_name)
    idxobj = IndexObject(embeddings=embeddings, metadata=metadata, mode=mode, vectorizer=vectorizer, encoder_name=encoder_name)
    return idxobj
