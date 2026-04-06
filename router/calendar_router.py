import re

from common.es_client import es
from search.calendar_search import calendar_rrf_search
from utils.reranker import rerank_chunks

try:
    from common.clarification_options import options_cache
    _OPTIONS_AVAILABLE = True
except Exception:
    _OPTIONS_AVAILABLE = False
    options_cache = None

index_name = "iit_calendar"

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

# Regex patterns
_re_full = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})\s*[,]?\s*(\d{4})\b",
    re.IGNORECASE,
)

_re_month_day = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})\b",
    re.IGNORECASE,
)

_re_month = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)


def detect_date(query: str):
    if not query:
        return None

    text = query.strip()

    # Full date
    m = _re_full.search(text)
    if m:
        month_name = m.group(1).lower()
        day = int(m.group(2))
        year = int(m.group(3))
        if month_name in MONTHS and 1 <= day <= 31:
            return {
                "type": "date_full",
                "year": year,
                "month": MONTHS[month_name],
                "day": day,
            }

    # Month + day
    m = _re_month_day.search(text)
    if m:
        month_name = m.group(1).lower()
        day = int(m.group(2))
        if month_name in MONTHS and 1 <= day <= 31:
            return {
                "type": "date_month_day",
                "month": MONTHS[month_name],
                "day": day,
            }

    # Month only
    m = _re_month.search(text)
    if m:
        month_name = m.group(1).lower()
        if month_name in MONTHS:
            # Look at the text strictly before the match
            prefix = text[:m.start()].strip().lower()
            last_word = prefix.split()[-1] if prefix.split() else ""            
            # Require a time preposition to prove this is a date
            valid_prepositions = {"in", "for", "during", "until", "by", "since", "of"}
            
            if last_word in valid_prepositions:
                return {
                    "type": "date_month",
                    "month": MONTHS[month_name],
                }

    return None


def date_search(date_info: dict):

    if date_info["type"] == "date_full":
        year = date_info["year"]
        month = date_info["month"]
        day = date_info["day"]
        date_str = f"{year}-{month:02d}-{day:02d}"

        filter_clause = [
            {"range": {"start_date": {"lte": date_str}}},
            {"range": {"end_date": {"gte": date_str}}},
        ]

    elif date_info["type"] == "date_month_day":
        month = date_info["month"]
        day = date_info["day"]

        filter_clause = [
            {
                "script": {
                    "script": {
                        "source": """
                            int startMonth = doc['start_date'].value.getMonthValue();
                            int startDay = doc['start_date'].value.getDayOfMonth();
                            int endMonth = doc['end_date'].value.getMonthValue();
                            int endDay = doc['end_date'].value.getDayOfMonth();

                            int qMonth = params.month;
                            int qDay = params.day;

                            boolean afterStart =
                                (qMonth > startMonth) ||
                                (qMonth == startMonth && qDay >= startDay);

                            boolean beforeEnd =
                                (qMonth < endMonth) ||
                                (qMonth == endMonth && qDay <= endDay);

                            return afterStart && beforeEnd;
                        """,
                        "params": {
                            "month": month,
                            "day": day,
                        },
                    }
                }
            }
        ]

    elif date_info["type"] == "date_month":
        month = date_info["month"]

        filter_clause = [
            {
                "script": {
                    "script": {
                        "source": "doc['start_date'].value.getMonthValue() == params.month",
                        "params": {"month": month},
                    }
                }
            }
        ]

    else:
        return []

    res = es.search(
        index=index_name,
        body={
            "size": 20,
            "sort": [{"start_date": "asc"}],
            "query": {"bool": {"filter": filter_clause}},
        },
    )

    hits = res["hits"]["hits"]
    for h in hits:
        if "_score" not in h or h.get("_score") is None:
            h["_score"] = 1.0

    return hits


# Pre-retrieval clarification: event present but no semester/term or year
_CALENDAR_EVENT_SLOTS = re.compile(
    r"\b(exam|exams|final|finals|midterm|midterms|break|graduation|commencement|"
    r"drop|withdraw|withdrawal|deadline|start|end|begin|close|class|classes|"
    r"register|registration|enroll|holiday|recess)\b",
    re.IGNORECASE,
)
_CALENDAR_TERM_OR_YEAR = re.compile(
    r"\b(spring|fall|summer|winter|20\d{2})\b",
    re.IGNORECASE,
)

_CALENDAR_NAMED_HOLIDAY = re.compile(
    r"\b(thanksgiving|christmas|labor day|memorial day|juneteenth|independence day|"
    r"martin luther king|mlk|new year|spring break|fall break|winter break)\b",
    re.IGNORECASE,
)


def _calendar_options():
    if _OPTIONS_AVAILABLE and options_cache:
        return options_cache.calendar_terms
    return []


def route_query(query: str):
    if not query or not query.strip():
        return []

    q = (query or "").strip().lower()
    has_event = bool(_CALENDAR_EVENT_SLOTS.search(q))
    has_term_or_year = bool(_CALENDAR_TERM_OR_YEAR.search(q))
    has_named_holiday = bool(_CALENDAR_NAMED_HOLIDAY.search(q))
    has_month = bool(_re_month.search(query))

    # Named holidays are unambiguous — retrieve directly without requiring term/year
    if has_named_holiday:
        return calendar_rrf_search(query)

    # Very short queries (≤2 words) without any temporal signal need clarification
    if len(q.split()) <= 2 and not has_term_or_year and not has_month:
        return {
            "needs_clarification": True,
            "message": "Which semester or year are you referring to?",
            "options": _calendar_options(),
        }

    # Event-related query with no semester/year AND no month — ask which semester
    if has_event and not has_term_or_year and not has_month:
        return {
            "needs_clarification": True,
            "message": "Which semester or year are you referring to?",
            "options": _calendar_options(),
        }

    date_info = detect_date(query)

    if date_info:
        raw_hits = date_search(date_info)
        if raw_hits:
            top_hits = raw_hits[:15]
            return rerank_chunks(query, top_hits, top_k=5)
        # No events found ON that date — the date in the query is a reference point
        # (e.g. "is April 10 before the withdrawal deadline?"), not the event to look up.
        # Fall through to semantic search to find the relevant deadline.

    return calendar_rrf_search(query)