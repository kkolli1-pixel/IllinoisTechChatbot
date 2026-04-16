import logging

from common.es_client import es
from common.embedding_model import model_large
from utils.reranker import rerank_chunks
from common.search_utils import clean_query, rrf_fuse
from common.query_augmentation import expand_query
from common.slot_filling import documents_query_validation

logger = logging.getLogger(__name__)

# Lexical search
def documents_lexical_search(query: str, top_k: int):

    cleaned_query = clean_query(query)

    try:
        results = es.search(
            index="iit_documents",
            body={
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": cleaned_query,
                        "fields": [
                            "content^2",
                            "topic^3",
                            "doc_name^2",
                            "doc_type"
                        ]
                    }
                }
            }
        )
        return results["hits"]["hits"]
    except Exception as e:
        logger.error(f"Documents lexical search failed for query '{query}': {e}")
        return []

# semantic search
def documents_semantic_search(query: str, top_k: int):

    query_vector = model_large.encode(
        f"query: {query}",
        normalize_embeddings=True
    ).tolist()

    try:
        results = es.search(
            index="iit_documents",
            body={
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": """
                                cosineSimilarity(params.query_vector, 'semantic_vector') + 1.0
                            """,
                            "params": {
                                "query_vector": query_vector
                            }
                        }
                    }
                }
            }
        )
        return results["hits"]["hits"]
    except Exception as e:
        logger.error(f"Documents semantic search failed for query '{query}': {e}")
        return []

# rrf search + reranker
def documents_rrf_search(query: str, top_k: int = 10):

    # 1. Soft Slot Validation
    validation = documents_query_validation(query)
    if validation.get("needs_clarification"):
        return validation

    expanded_query = expand_query(query, "DOCUMENTS", max_expansions=4)
    # Deterministic query augmentation for common low-recall phrases.
    # Keep this local and keyword-gated to avoid changing routing across domains.
    ql = (query or "").lower()
    extra_phrases = []
    if any(k in ql for k in ("housing", "residence", "dorm", "dormitory")):
        extra_phrases.extend(["residence life handbook", "residence halls", "residence life"])
    if any(k in ql for k in ("visa", "immigration", "international student", "iss")):
        extra_phrases.extend(["office of global services", "immigration status", "iss"])
    if extra_phrases:
        expanded_query = f"{expanded_query.strip()} {' '.join(extra_phrases)}"
    lexical_hits = documents_lexical_search(expanded_query, top_k)
    semantic_hits = documents_semantic_search(query, top_k)

    fused = rrf_fuse(lexical_hits, semantic_hits)

    # rerank ES hits directly using cross-encoder
    # Use the expanded query for reranking so the cross-encoder can align on
    # the same key entities we boosted in lexical retrieval.
    reranked_hits = rerank_chunks(expanded_query, fused, top_k=5)

    # final top results
    return reranked_hits