import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

try:
    import streamlit as st
except ImportError:  # Allow non-Streamlit usage (e.g., CLI)
    st = None

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

if st is not None:
    @st.cache_resource
    def load_reranker():
        # Heavy model load; cached across Streamlit reruns.
        return CrossEncoder(RERANKER_NAME)
else:
    def load_reranker():
        return CrossEncoder(RERANKER_NAME)

# Backwards-compatible export: code imports `reranker` directly.
# The underlying loader is cached when running under Streamlit.
reranker = load_reranker()

def rerank_chunks(query: str, hits: list, top_k: int = 3):
    """
    Re-rank retrieved chunks using a cross-encoder.
    """

    if not hits:
        return hits

    # Limit candidates for cross-encoder efficiency
    hits = hits[:20]

    # Prepare query-chunk pairs
    pairs = []
    valid_hits = []

    for h in hits:
        content = h["_source"].get("content") or h["_source"].get("semantic_text")
        if not content:
            continue
        pairs.append((query, content))
        valid_hits.append(h)

    if not valid_hits:
        return []

    if len(valid_hits) <= top_k:
        return valid_hits  # all scoreable hits fit; no need to rerank

    # Predict relevance scores
    scores = reranker.predict(pairs)

    # Attach scores
    for hit, score in zip(valid_hits, scores):
        hit["_rerank_score"] = float(score)
    
    # Sort safely
    ranked = sorted(
        valid_hits,
        key=lambda x: x.get("_rerank_score", 0.0),
        reverse=True
    )

    return ranked[:top_k]