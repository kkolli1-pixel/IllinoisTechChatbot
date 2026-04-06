"""
Derive stable fee_kind values for iit_tuition from source fee_name only.

Used at index time and kept in ES as a keyword for filter/boost — avoids
maintaining long fee-name blocklists in search code.
"""

from __future__ import annotations

import re

# Stored in Elasticsearch `fee_kind` (keyword).
FEE_KIND_TUITION = "tuition"
FEE_KIND_CONTINUATION = "continuation"
FEE_KIND_OTHER = "other"


def derive_fee_kind(fee_name: object) -> str:
    """
    Map a row's fee_name to a coarse category.

    Primary per-credit tuition rows use fee_name exactly \"Tuition\" in the
    scraped data. Continuation / studies bundles are labeled separately.
    """
    if fee_name is None:
        return FEE_KIND_OTHER
    # pandas/JSON NaN
    try:
        if isinstance(fee_name, float) and fee_name != fee_name:
            return FEE_KIND_OTHER
    except Exception:
        pass
    s = str(fee_name).strip()
    if not s or s.lower() == "nan":
        return FEE_KIND_OTHER
    lower = s.lower()
    if lower == "tuition":
        return FEE_KIND_TUITION
    if "continuation" in lower:
        return FEE_KIND_CONTINUATION
    return FEE_KIND_OTHER


# Queries that target continuation / non-primary tuition — do not restrict to fee_kind=tuition.
_EXCLUDE_PRIMARY_FEE_KIND_FILTER = (
    "continuation",
    "continuation studies",
    "credit by proficiency",
    "proficiency exam",
    "graduate continuation",
    "all fees",
    "mandatory",
    "other fees",
)

_PRIMARY_TUITION_QUERY_RE = re.compile(
    r"\btuition\b|\b(per credit|credit hour|/credit)\b",
    re.IGNORECASE,
)


def should_filter_to_primary_tuition_fee_kind(query: str) -> bool:
    """
    When True, tuition search may add term filter fee_kind=tuition (with fallback if empty).

    Use only for broad \"how much is tuition / per credit\" style questions, not
    named ancillary or continuation fees, and NOT when the user asks for 'all' fees.
    """
    q = query.lower()
    if any(p in q for p in _EXCLUDE_PRIMARY_FEE_KIND_FILTER):
        return False
    # If the user asks for "fees" (plural) or "all", don't restrict to just pure tuition rows.
    if "all" in q or "fees" in q:
        return False
    return bool(_PRIMARY_TUITION_QUERY_RE.search(query))
