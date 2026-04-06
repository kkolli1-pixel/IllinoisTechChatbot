import logging

from common.es_client import es
from common.embedding_model import model_large
from utils.reranker import rerank_chunks
from common.search_utils import clean_query, rrf_fuse
from common.query_augmentation import expand_query
from common.slot_filling import calendar_query_validation

logger = logging.getLogger(__name__)

# Lexical search
def calendar_lexical_search(query: str, top_k: int):

    cleaned_query = clean_query(query)

    inner_query = {
        "bool": {
            "should": [
                {
                    "multi_match": {
                        "query": cleaned_query,
                        "fields": [
                            "event_name^8",
                            "term^3",
                            "semantic_text"
                        ],
                    }
                },
                {
                    "match_phrase": {
                        "event_name": {
                            "query": cleaned_query,
                            "boost": 6
                        }
                    }
                }
            ],
            "minimum_should_match": 1
        }
    }

    try:
        results = es.search(
            index="iit_calendar",
            body={
                "size": top_k,
                "query": inner_query,
            },
        )
        return results["hits"]["hits"]
    except Exception as e:
        logger.error(f"Calendar lexical search failed for query '{query}': {e}")
        return []

# Semantic search
def calendar_semantic_search(query: str, top_k: int):

    query_vector = model_large.encode(
        f"query: {query}",
        normalize_embeddings=True
    ).tolist()

    inner_query = {"match_all": {}}

    try:
        results = es.search(
            index="iit_calendar",
            body={
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": inner_query,
                        "script": {
                            "source": """
                                cosineSimilarity(params.query_vector, 'semantic_vector') + 1.0
                            """,
                            "params": {
                                "query_vector": query_vector
                            }
                        }
                    }
                },
            },
        )
        return results["hits"]["hits"]

    except Exception as e:
        logger.error(f"Calendar semantic search failed for query '{query}': {e}")
        return []

# RRF search + reranker
def calendar_rrf_search(query: str, top_k: int = 10):

    # 1. Soft Slot Validation
    validation = calendar_query_validation(query)
    if validation.get("needs_clarification"):
        return validation

    expanded_query = expand_query(query, "CALENDAR")
    lexical_hits = calendar_lexical_search(expanded_query, top_k)
    semantic_hits = calendar_semantic_search(query, top_k)

    fused = rrf_fuse(lexical_hits, semantic_hits)

    # rerank ES hits directly using cross-encoder
    reranked_hits = rerank_chunks(query, fused, top_k=5)

    return reranked_hits


# search
# def hybrid_search(query: str, top_k: int, index_name: str):
#     cleaned_query = _clean_query(query)
#     query_vector = model.encode(cleaned_query).tolist()

#     lexical_results = es.search(
#         index=index_name,
#         body={
#             "size": top_k,
#             "query": {
#                 "script_score": {
#                     "query": {
#                         "bool": {
#                             "should": [
#                                 {
#                                     "multi_match": {
#                                         "query": cleaned_query,
#                                         "fields": ["event_name^4", "term^1.5", "semantic_text"],
#                                     }
#                                 },
#                                 {
#                                     "match_phrase": {
#                                         "event_name": {
#                                             "query": cleaned_query,
#                                             "boost": 3
#                                         }
#                                     }
#                                 }
#                             ],
#                             "minimum_should_match": 1
#                         }
#                     },
#                     "script": {
#                         "source":"""
#                             double bm25 = _score; // BM25 score
#                             double vector_score = cosineSimilarity(params.query_vector, 'semantic_vector') + 1.0; // Semantic score
#                           return (0.6 * bm25) + (0.4 * vector_score);
#                         """,

#                         "params": {
#                             "query_vector": query_vector
#                         }
#                     }
#                 }
#             },
#         },
#     )

#     lexical_hits = lexical_results["hits"]["hits"]

#     # max_bm25_score = max([hit["_score"] for hit in lexical_hits], default=1.0)

#     # # Normalise scores
#     # for hit in lexical_hits:
#     #     hit["_score"] = hit["_score"] / max_bm25_score

#     return lexical_hits