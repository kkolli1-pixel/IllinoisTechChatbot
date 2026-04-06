from sentence_transformers import SentenceTransformer

try:
    import streamlit as st
except ImportError:  # Allow non-Streamlit usage (e.g., CLI)
    st = None

MODEL_LARGE_NAME = "intfloat/e5-large-v2"

if st is not None:
    @st.cache_resource
    def load_model_large():
        # Heavy model load; cached across Streamlit reruns.
        return SentenceTransformer(MODEL_LARGE_NAME)
else:
    def load_model_large():
        return SentenceTransformer(MODEL_LARGE_NAME)


# Backwards-compatible export: code imports `model_large` directly.
# The underlying loader is cached when running under Streamlit.
model_large = load_model_large()