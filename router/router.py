import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from sklearn.metrics.pairwise import cosine_similarity

from common.embedding_model import model_large

from questions.calendar_questions import CALENDAR_PROTOTYPES
from questions.contact_questions import CONTACTS_PROTOTYPES
from questions.documents_questions import DOCUMENTS_PROTOTYPES
from questions.tuition_questions import TUITION_PROTOTYPES
from questions.ood_questions import OOD_PROTOTYPES

logger = logging.getLogger(__name__)

# ── Load routing parameters from config ──────────────────────────────────────

def _load_router_config() -> Dict:
    for config_path in [
        Path(__file__).resolve().parent.parent / "config" / "router_config.json",
        Path(__file__).resolve().parent / "router_config.json",
    ]:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load router_config.json: {e}")
    logger.warning("router_config.json not found — using defaults.")
    return {"confidence_threshold": 0.50, "top_k": 5, "multi_domain_ratio": 0.75}

_ROUTER_CONFIG = _load_router_config()
TOP_K = int(_ROUTER_CONFIG.get("top_k", 5))
CONFIDENCE_THRESHOLD = float(_ROUTER_CONFIG.get("confidence_threshold", 0.50))
MULTI_DOMAIN_RATIO = float(_ROUTER_CONFIG.get("multi_domain_ratio", 0.75))

# Domain constants
DOMAIN_CALENDAR = "CALENDAR"
DOMAIN_CONTACTS = "CONTACTS"
DOMAIN_DOCUMENTS = "DOCUMENTS"
DOMAIN_TUITION = "TUITION"
DOMAIN_OOD = "OOD"

ALLOWED_DOMAINS = [
    DOMAIN_CALENDAR,
    DOMAIN_CONTACTS,
    DOMAIN_DOCUMENTS,
    DOMAIN_TUITION
]


# PROTOTYPE QUESTIONS (OOD included for routing, not for search)
PROTOTYPES = {
    DOMAIN_CALENDAR: CALENDAR_PROTOTYPES,
    DOMAIN_CONTACTS: CONTACTS_PROTOTYPES,
    DOMAIN_DOCUMENTS: DOCUMENTS_PROTOTYPES,
    DOMAIN_TUITION: TUITION_PROTOTYPES,
    DOMAIN_OOD: OOD_PROTOTYPES
}

# Precompute prototype embeddings
logger.info("Encoding prototype queries...")

prototype_embeddings = {}

for domain, questions in PROTOTYPES.items():

    embeddings = model_large.encode(
        [f"query: {q}" for q in questions],
        normalize_embeddings=True
    )

    prototype_embeddings[domain] = embeddings

logger.info("Prototype embeddings ready.")

# Router function
def get_routing_intent(query: str) -> Dict[str, List[str]]:

    if not query or not query.strip():
        return {"domains": []}

    try:

        # Query embedding
        query_embedding = model_large.encode(
            f"query: {query}",
            normalize_embeddings=True
        ).reshape(1, -1)

        similarities = []

        # Compare against prototypes
        for domain, embeddings in prototype_embeddings.items():

            sim_scores = cosine_similarity(
                query_embedding,
                embeddings
            )[0]

            for score in sim_scores:
                similarities.append((domain, float(score)))

        # Sort similarities
        similarities.sort(
            key=lambda x: x[1],
            reverse=True
        )

        best_score = similarities[0][1]

        # Confidence check — raise threshold for very short queries
        word_count = len(query.strip().split())
        effective_threshold = 0.68 if word_count < 3 else CONFIDENCE_THRESHOLD
        if best_score < effective_threshold:
            logger.debug("Router confidence too low.")
            return {
                "domains": [],
                "needs_clarification": False,
                "sub_queries": {},
            }

        # Top-K prototype matches
        top_k = similarities[:TOP_K]

        logger.debug("Top prototype matches:")
        for domain, score in top_k:
            logger.debug(f"{domain}: {score:.3f}")

        # Aggregate domain scores
        domain_scores = {}

        for domain, score in top_k:
            if domain not in domain_scores:
                domain_scores[domain] = []
            domain_scores[domain].append(score)

        domain_scores = {
            domain: max(scores)
            for domain, scores in domain_scores.items()
        }

        ranked_domains = sorted(
            domain_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Multi-domain routing
        best_score = ranked_domains[0][1]

        domains = [
            domain
            for domain, score in ranked_domains
            if score >= best_score * MULTI_DOMAIN_RATIO
        ]

        # Filter out OOD — if OOD wins, return empty domains (fast rejection)
        domains = [d for d in domains if d != DOMAIN_OOD]

        return {
            "domains": domains,
            "needs_clarification": False,
            "sub_queries": {domain: query for domain in domains},
        }

    except Exception as e:

        logger.error(f"Semantic routing failed: {e}")

        return {
            "domains": [],
            "needs_clarification": False,
            "sub_queries": {},
        }