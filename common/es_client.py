import os
from elasticsearch import Elasticsearch

# Prefer ES_URL; fall back to ES_HOST (used in this project's .env)
ES_URL = (
    os.getenv("ES_URL")
    or os.getenv("ES_HOST")
    or "http://localhost:9200"
)

es = Elasticsearch(
    ES_URL,
    request_timeout=120,
    max_retries=5,
    retry_on_timeout=True,
)
