"""
clarification_options.py
Fetches distinct field values from ES indices to populate clarification messages.
Results are cached at import time — one ES aggregation per domain on startup.
"""

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

try:
    from common.es_client import es
    _ES_AVAILABLE = True
except Exception:
    _ES_AVAILABLE = False
    es = None


def _agg_terms(index: str, field: str, size: int = 50) -> List[str]:
    """Return sorted distinct values for a keyword field via ES terms aggregation."""
    if not _ES_AVAILABLE or es is None:
        return []
    try:
        resp = es.search(
            index=index,
            body={
                "size": 0,
                "aggs": {
                    "vals": {
                        "terms": {"field": field, "size": size}
                    }
                }
            }
        )
        buckets = resp["aggregations"]["vals"]["buckets"]
        return sorted(b["key"] for b in buckets if b["key"])
    except Exception as e:
        logger.warning(f"ES agg failed for {index}.{field}: {e}")
        return []


# ── Calendar ──────────────────────────────────────────────────────────────────

def get_calendar_terms() -> List[str]:
    """Distinct term values from iit_calendar — e.g. 'Spring 2026', 'Fall 2026'."""
    terms = _agg_terms("iit_calendar", "term")
    if terms:
        return _post_process_calendar_terms(terms)

    # Fallback: if `term` is mapped as `text` without doc-values, ES terms aggs
    # can fail (fielddata disabled). In that case, scan a bounded set of docs
    # and build the unique set from `_source.term`.
    try:
        resp = es.search(
            index="iit_calendar",
            body={"size": 500, "_source": ["term"]},
        )
        hits = resp.get("hits", {}).get("hits", []) or []
        raw = [h.get("_source", {}).get("term") for h in hits]
        terms_scanned = sorted({t for t in raw if t})
        return _post_process_calendar_terms(terms_scanned)
    except Exception:
        return []


def _post_process_calendar_terms(terms: List[str]) -> List[str]:
    main = [t for t in terms if "Coursera" not in t and "Calendar Year" not in t]
    coursera = [t for t in terms if "Coursera" in t]
    return main + coursera


_CALENDAR_TOKEN_STOPWORDS = {
    "of", "the", "and", "for", "a", "an", "to", "in", "on", "at", "or",
    "is", "are", "be", "by", "do", "no", "due", "day", "new", "may",
    "full", "with", "into", "combined", "converting", "students",
    "undergraduate", "online", "published", "observed", "observance",
    "starts", "begins", "begin", "monday", "noon", "pba", "cst",
    "mooc", "year", "session", "semester", "coursera", "university",
    "schedule", "schedules", "charges", "late", "early", "last",
}


def get_calendar_event_tokens() -> List[str]:
    """
    Extract distinct meaningful tokens from iit_calendar event_name values.
    Used to build a dynamic regex for slot validation instead of hardcoding event types.
    """
    if not _ES_AVAILABLE or es is None:
        return []
    try:
        resp = es.search(
            index="iit_calendar",
            body={
                "size": 0,
                "aggs": {
                    "names": {
                        "terms": {"field": "event_name.keyword", "size": 500}
                        if False  # event_name is text-only; use top hits instead
                        else "terms"
                    }
                }
            }
        )
    except Exception:
        pass

    # event_name is text-only (no keyword subfield) — fetch all docs and tokenize
    try:
        resp = es.search(
            index="iit_calendar",
            body={"size": 500, "_source": ["event_name"]},
        )
        hits = resp["hits"]["hits"]
        tokens = set()
        for h in hits:
            name = (h["_source"].get("event_name") or "").lower()
            for w in re.findall(r"\b[a-z]{3,}\b", name):
                if w not in _CALENDAR_TOKEN_STOPWORDS:
                    tokens.add(w)
        return sorted(tokens)
    except Exception as e:
        logger.warning(f"Calendar event token extraction failed: {e}")
        return []


# ── Tuition ───────────────────────────────────────────────────────────────────

def get_tuition_schools() -> List[str]:
    """Distinct school values from iit_tuition."""
    return _agg_terms("iit_tuition", "school")


def get_tuition_levels() -> List[str]:
    """Distinct level values from iit_tuition, excluding 'all'."""
    levels = _agg_terms("iit_tuition", "level")
    return [l for l in levels if l != "all"]


def get_tuition_years() -> List[str]:
    """Distinct academic_year values from iit_tuition, most recent first."""
    years = _agg_terms("iit_tuition", "academic_year")
    return sorted(years, reverse=True)


def get_tuition_fee_names() -> List[str]:
    """Distinct fee_name values from iit_tuition."""
    return _agg_terms("iit_tuition", "fee_name", size=100)


# ── Contacts ──────────────────────────────────────────────────────────────────

def get_contact_departments() -> List[str]:
    """Distinct department values from iit_contacts."""
    return _agg_terms("iit_contacts", "department.keyword")


def get_contact_categories() -> List[str]:
    """Distinct category values from iit_contacts."""
    return _agg_terms("iit_contacts", "category")


# ── Cached singletons (loaded once at startup) ────────────────────────────────

class _OptionsCache:
    """Lazy-loaded cache so ES is only hit once per process."""

    def __init__(self):
        self._calendar_terms = None
        self._calendar_event_tokens = None
        self._tuition_schools = None
        self._tuition_levels = None
        self._tuition_years = None
        self._tuition_fee_names = None
        self._contact_departments = None
        self._contact_categories = None

    @property
    def calendar_terms(self):
        if self._calendar_terms is None:
            self._calendar_terms = get_calendar_terms()
        return self._calendar_terms

    @property
    def calendar_event_tokens(self):
        if self._calendar_event_tokens is None:
            self._calendar_event_tokens = get_calendar_event_tokens()
        return self._calendar_event_tokens

    @property
    def tuition_schools(self):
        if self._tuition_schools is None:
            self._tuition_schools = get_tuition_schools()
        return self._tuition_schools

    @property
    def tuition_levels(self):
        if self._tuition_levels is None:
            self._tuition_levels = get_tuition_levels()
        return self._tuition_levels

    @property
    def tuition_years(self):
        if self._tuition_years is None:
            self._tuition_years = get_tuition_years()
        return self._tuition_years

    @property
    def tuition_fee_names(self):
        if self._tuition_fee_names is None:
            self._tuition_fee_names = get_tuition_fee_names()
        return self._tuition_fee_names

    @property
    def contact_departments(self):
        if self._contact_departments is None:
            self._contact_departments = get_contact_departments()
        return self._contact_departments

    @property
    def contact_categories(self):
        if self._contact_categories is None:
            self._contact_categories = get_contact_categories()
        return self._contact_categories


options_cache = _OptionsCache()