import logging

from common.es_client import es
from common.embedding_model import model_large
from utils.reranker import rerank_chunks
from common.search_utils import clean_query, rrf_fuse
from common.query_augmentation import expand_query
from common.slot_filling import contacts_query_validation

logger = logging.getLogger(__name__)

# Lexical search
def contacts_lexical_search(query: str, top_k: int):

    cleaned_query = clean_query(query)

    try:
        results = es.search(
            index="iit_contacts",
            body={
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": cleaned_query,
                        "fields": [
                            "name^3",
                            "department^2",
                            "category^2",
                            "description",
                            "building",
                            "address",
                        ]
                    }
                }
            }
        )
        return results["hits"]["hits"]
    except Exception as e:
        logger.error(f"Contacts lexical search failed for query '{query}': {e}")
        return []

# Semantic search
def contacts_semantic_search(query: str, top_k: int):

    query_vector = model_large.encode(
        f"query: {query}",
        normalize_embeddings=True
    ).tolist()

    try:
        results = es.search(
            index="iit_contacts",
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
        logger.error(f"Contacts semantic search failed for query '{query}': {e}")
        return []

# RRF search + reranker
def contacts_rrf_search(query: str, top_k: int = 10):

    # 1. Soft Slot Validation
    validation = contacts_query_validation(query)
    if validation.get("needs_clarification"):
        return validation

    expanded_query = expand_query(query, "CONTACTS")
    lexical_hits = contacts_lexical_search(expanded_query, top_k)
    semantic_hits = contacts_semantic_search(query, top_k)

    fused = rrf_fuse(lexical_hits, semantic_hits)

    # rerank ES hits directly using cross-encoder
    reranked_hits = rerank_chunks(query, fused, top_k=5)

    return reranked_hits
    
# Lexical search
# def contacts_lexical_search(query: str, top_k: int = 1):
#     cleaned_query = _clean_query(query)

#     results = es.search(
#         index="iit_contacts",
#         body={
#             "size": top_k,
#             "query": {
#                 "bool": {
#                     "should": [
#                         {
#                             "multi_match": {
#                                 "query": cleaned_query,
#                                 "type": "best_fields",
#                                 "fields": [
#                                     "name^3",
#                                     "department^2",
#                                     "description",
#                                     "building",
#                                     "address"
#                                 ]
#                             }
#                         },
#                         {
#                             "term": {
#                                 "email": {
#                                     "value": cleaned_query,
#                                     "boost": 5
#                                 }
#                             }
#                         },
#                     ]
#                 }
#             }
#         },
#     )

#     return results["hits"]["hits"]

# # Semantic search
# def contacts_semantic_search(query: str, top_k: int = 1):
#     cleaned_query = _clean_query(query)
#     query_vector = model.encode(cleaned_query).tolist()

#     results = es.search(
#         index="iit_contacts",
#         body={
#             "size": top_k,
#             "query": {
#                 "script_score": {
#                     "query": {"match_all": {}},
#                     "script": {
#                         "source": """
#                             cosineSimilarity(params.query_vector, 'semantic_vector') + 1.0
#                         """,
#                         "params": {
#                             "query_vector": query_vector
#                         }
#                     }
#                 }
#             }
#         },
#     )

#     return results["hits"]["hits"]