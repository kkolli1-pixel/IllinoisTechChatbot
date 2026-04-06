# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps (needed by sentence-transformers / torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps first (better layer caching) ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download models so cold starts are fast ───────────────────────────────
# This bakes the models into the image at build time (~1.5 GB extra)
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("intfloat/e5-large-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
EOF

# ── Copy project source ───────────────────────────────────────────────────────
COPY . .

# ── Expose FastAPI port ───────────────────────────────────────────────────────
EXPOSE 8000

# ── Start uvicorn ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]
