import re
import logging
import json

from common.es_client import es
from common.embedding_model import model_large
from common.query_augmentation import expand_query
from common.search_utils import clean_query, rrf_fuse
from common.slot_filling import tuition_query_validation
from common.tuition_fee_kind import FEE_KIND_TUITION, should_filter_to_primary_tuition_fee_kind
from utils.reranker import rerank_chunks

logger = logging.getLogger(__name__)

INDEX_NAME = "iit_tuition"

# ── Slot extraction ───────────────────────────────────────────────────────────

# School name mapping — query aliases → exact ES keyword values
_SCHOOL_MAP = {
    "chicago-kent": "Chicago-Kent",
    "chicago kent": "Chicago-Kent",
    "kent": "Chicago-Kent",
    "kent law": "Chicago-Kent",
    "law school": "Chicago-Kent",
    "institute of design": "Institute of Design",
    "intensive english": "Intensive English Program",
    "english program": "Intensive English Program",
    "iep": "Intensive English Program",
    "mies": "Mies",
    "mies campus": "Mies",
    "stuart": "Stuart School of Business",
    "stuart school": "Stuart School of Business",
    "business school": "Stuart School of Business",
}

_LEVEL_MAP = {
    "graduate": "graduate",
    "grad": "graduate",
    "undergraduate": "undergrad",
    "undergrad": "undergrad",
}

_YEAR_RE = re.compile(r"\b(20\d{2}-20\d{2}|20\d{2})\b")

def _extract_tuition_filters(query: str) -> dict:

    q = query.lower()
    filters = {}

    # School — if multiple schools are mentioned (compare/vs), don't hard-filter to one.
    matched_schools = []
    for alias, school in sorted(_SCHOOL_MAP.items(), key=lambda x: -len(x[0])):
        if alias in q and school not in matched_schools:
            matched_schools.append(school)
    if len(matched_schools) == 1:
        filters["school"] = matched_schools[0]

    # Level
    for alias, level in _LEVEL_MAP.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            filters["level"] = level
            break

    # Academic year
    m = _YEAR_RE.search(q)
    if m:
        year = m.group(1)
        if "-" not in year:
            year = f"{year}-{int(year) + 1}"
        filters["academic_year"] = year

    return filters

def _build_filter_clause(filters: dict) -> list:
    """Convert extracted slot values to ES term filter clauses."""
    clauses = []
    if "school" in filters:
        clauses.append({"term": {"school": filters["school"]}})
    if "level" in filters:
        clauses.append({"term": {"level": filters["level"]}})
    if "academic_year" in filters:
        clauses.append({"term": {"academic_year": filters["academic_year"]}})
    return clauses

def _filters_without_level(filters: dict) -> dict:
    return {k: v for k, v in filters.items() if k != "level"}

# ── Lexical search ────────────────────────────────────────────────────────────

def tuition_lexical_search(query: str, top_k: int, filter_clauses: list = None):

    cleaned_query = clean_query(query)

    query_body = {
        "multi_match": {
            "query": cleaned_query,
            "fields": [
                "chunk_text^3",
                "content^2",
                "fee_name^2",
                "section",
                "school",
                "level",
                "academic_year",
                "term",
                "enrollment",
                "program",
                "unit",
                "billing_period",
            ],
        }
    }

    if filter_clauses:
        query_body = {
            "bool": {
                "must": [query_body],
                "filter": filter_clauses,
            }
        }

    try:
        results = es.search(
            index=INDEX_NAME,
            body={"size": top_k, "query": query_body},
        )
        return results["hits"]["hits"]
    except Exception as e:
        logger.error(f"Tuition lexical search failed for query '%s': %s", query, e)
        return []

# ── Semantic search ───────────────────────────────────────────────────────────

def tuition_semantic_search(query: str, top_k: int, filter_clauses: list = None):

    query_vector = model_large.encode(
        f"query: {query}",
        normalize_embeddings=True,
    ).tolist()

    inner_query = (
        {"bool": {"filter": filter_clauses}}
        if filter_clauses
        else {"match_all": {}}
    )

    try:
        results = es.search(
            index=INDEX_NAME,
            body={
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": inner_query,
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'semantic_vector') + 1.0",
                            "params": {"query_vector": query_vector},
                        },
                    }
                },
            },
        )
        return results["hits"]["hits"]
    except Exception as e:
        logger.error(f"Tuition semantic search failed for query '%s': %s", query, e)
        return []

# ── RRF search + reranker ─────────────────────────────────────────────────────

def tuition_rrf_search(query: str, top_k: int = 10):

    validation = tuition_query_validation(query)
    if validation.get("needs_clarification"):
        return validation

    # Extract structured filters from query
    filters = _extract_tuition_filters(query)
    filter_clauses = _build_filter_clause(filters)

    fee_kind_clauses = []
    if should_filter_to_primary_tuition_fee_kind(query):
        fee_kind_clauses.append({"term": {"fee_kind": FEE_KIND_TUITION}})

    combined_filters = filter_clauses + fee_kind_clauses

    if filter_clauses:
        logger.debug(f"Tuition filters applied: {filters}")
    if fee_kind_clauses:
        logger.debug("Tuition fee_kind filter: primary tuition (fee_kind=tuition)")

    expanded_query = expand_query(query, "TUITION")
    lexical_hits = tuition_lexical_search(
        expanded_query, top_k, combined_filters if combined_filters else None
    )
    semantic_hits = tuition_semantic_search(
        query, top_k, combined_filters if combined_filters else None
    )

    # Retry without fee_kind if the index row set is empty (or legacy index without fee_kind).
    if not lexical_hits and not semantic_hits and fee_kind_clauses:
        logger.debug(
            "Tuition search with fee_kind filter returned no results — retrying without fee_kind"
        )
        lexical_hits = tuition_lexical_search(
            expanded_query, top_k, filter_clauses if filter_clauses else None
        )
        semantic_hits = tuition_semantic_search(
            query, top_k, filter_clauses if filter_clauses else None
        )

    # Some schools store tuition under level=all; if school+level finds nothing,
    # retry with the school retained before dropping all filters.
    if not lexical_hits and not semantic_hits and "school" in filters and "level" in filters:
        school_only_filters = _filters_without_level(filters)
        school_only_clauses = _build_filter_clause(school_only_filters)
        school_combined = school_only_clauses + fee_kind_clauses
        logger.debug(
            "Tuition search with school+level returned no results — retrying with school only"
        )
        lexical_hits = tuition_lexical_search(
            expanded_query, top_k, school_combined if school_combined else None
        )
        semantic_hits = tuition_semantic_search(
            query, top_k, school_combined if school_combined else None
        )

    # Fallback to unfiltered if slot filters return nothing
    if not lexical_hits and not semantic_hits and filter_clauses:
        logger.debug("Tuition filters returned no results — falling back to unfiltered")
        lexical_hits = tuition_lexical_search(expanded_query, top_k)
        semantic_hits = tuition_semantic_search(query, top_k)

    # Level diversification: when school is specified but level is not,
    # search separately per level and merge so the LLM sees all student levels.
    needs_diversification = "school" in filters and "level" not in filters
    if needs_diversification:
        per_level_hits = []
        for lvl in ("graduate", "undergrad", "all"):
            lvl_filters = {**filters, "level": lvl}
            lvl_clauses = _build_filter_clause(lvl_filters) + fee_kind_clauses
            lvl_lex = tuition_lexical_search(expanded_query, 5, lvl_clauses if lvl_clauses else None)
            lvl_sem = tuition_semantic_search(query, 5, lvl_clauses if lvl_clauses else None)
            per_level_hits.extend(lvl_lex)
            per_level_hits.extend(lvl_sem)
        if per_level_hits:
            lexical_hits.extend(per_level_hits)

    fused = rrf_fuse(lexical_hits, semantic_hits)
    rerank_k = 10 if needs_diversification else 5
    return rerank_chunks(query, fused, top_k=rerank_k)