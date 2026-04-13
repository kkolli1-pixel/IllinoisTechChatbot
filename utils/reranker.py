import re
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

# University-name noise words — every document is about IIT, so these
# bias the cross-encoder toward chunks that happen to spell out the name.
_UNIVERSITY_NOISE = re.compile(
    r"\b(?:at\s+)?(?:iit|illinois\s+institute\s+of\s+technology|illinois\s+tech)\b",
    re.IGNORECASE,
)


def rerank_chunks(query: str, hits: list, top_k: int = 3):
    """
    Re-rank retrieved chunks using a cross-encoder.
    """

    if not hits:
        return hits

    # Limit candidates for cross-encoder efficiency
    hits = hits[:20]

    # Strip university name from query — every chunk is about IIT, so "at IIT"
    # just biases the cross-encoder toward chunks that spell out the name.
    clean_q = _UNIVERSITY_NOISE.sub("", query).strip()
    clean_q = re.sub(r"\s{2,}", " ", clean_q)
    if not clean_q:
        clean_q = query  # safety: don't blank out the query entirely

    # Prepare query-chunk pairs
    pairs = []
    valid_hits = []

    for h in hits:
        content = h["_source"].get("content") or h["_source"].get("semantic_text")
        if not content:
            continue
        # Prepend topic/section title so the cross-encoder sees structural context
        topic = h["_source"].get("topic") or ""
        if topic and not content.startswith(topic):
            content = f"{topic}. {content}"
        pairs.append((clean_q, content))
        valid_hits.append(h)

    if not valid_hits:
        return []

    if len(valid_hits) <= top_k:
        return valid_hits  # all scoreable hits fit; no need to rerank

    # Predict relevance scores
    scores = reranker.predict(pairs)

    # Attach scores with topic-match boost.
    # The cross-encoder struggles with long policy documents where the topic
    # field is highly relevant but content leads with generic language.
    # Boost chunks whose topic contains query content words.
    query_words = set(re.findall(r"\b[a-z]{3,}\b", clean_q.lower()))
    # Remove common stop words from matching
    query_words -= {"the", "what", "how", "does", "are", "for", "and", "this", "that", "with", "from", "about"}

    for hit, score in zip(valid_hits, scores):
        topic = (hit["_source"].get("topic") or "").lower()
        if topic and query_words:
            topic_words = set(re.findall(r"\b[a-z]{3,}\b", topic))
            overlap = query_words & topic_words
            # Proportional boost: fraction of query words found in topic
            boost = len(overlap) / len(query_words) * 8.0 if overlap else 0.0
        else:
            boost = 0.0
        hit["_rerank_score"] = float(score) + boost

    # Sort safely
    ranked = sorted(
        valid_hits,
        key=lambda x: x.get("_rerank_score", 0.0),
        reverse=True
    )

    return ranked[:top_k]