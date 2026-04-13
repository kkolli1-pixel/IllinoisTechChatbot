"""
slot_filling.py
Domain-specific query validation with zero hardcoded domain values.

- Contacts strong units  → built from ES department names at startup
- Calendar event tokens  → built from ES event_name values at startup
- Tuition school/fee     → built from ES school + fee_name values at startup
- Documents topics       → loaded from config/documents_topics.json
"""

import json
import re
import string
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    from common.clarification_options import options_cache
    _OPTIONS_AVAILABLE = True
except Exception:
    _OPTIONS_AVAILABLE = False
    options_cache = None


# ── Documents config (only thing that can't come from ES cleanly) ─────────────

def _load_documents_config() -> Dict:
    """Load topic and doc_type keywords from JSON config — no hardcoding in Python."""
    for config_path in [
        Path(__file__).resolve().parent.parent / "config" / "documents_topics.json",
        Path(__file__).resolve().parent / "documents_topics.json",
    ]:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load documents_topics.json: {e}")
    logger.warning("documents_topics.json not found — document validation will be permissive.")
    return {"topic_keywords": [], "doc_type_keywords": []}


_DOCS_CONFIG = _load_documents_config()


# ── Pattern builder ───────────────────────────────────────────────────────────

def _build_pattern(terms: List[str], flags=re.IGNORECASE) -> "re.Pattern | None":
    """Build a word-boundary regex from a list of terms. Returns None if list is empty."""
    clean = [t.lower().strip() for t in terms if t and t.strip()]
    if not clean:
        return None
    escaped = sorted([re.escape(t) for t in clean], key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", flags)


# ── Department alias expansion ────────────────────────────────────────────────

def _expand_department_aliases(departments: List[str]) -> List[str]:
    """
    Generate short-form aliases from full ES department names so that
    partial queries like "computer science" match "Department of Computer Science".

    Strategy:
    1. Strip known prefixes ("Department of", "College of", etc.)
    2. Extract suffix from "X College of Y" → "Y"
    3. Add manual aliases for common short forms
    """
    STRIP_PREFIXES = [
        "department of ",
        "college of ",
        "school of ",
        "institute of ",
        "office of ",
    ]
    INFIX_RE = re.compile(r".+\b(?:college|school|institute)\b of (.+)", re.IGNORECASE)
    GENERIC_SKIP = {"law", "design", "letters", "science", "arts"}

    seen = {d.lower() for d in departments}
    aliases = list(departments)

    for dept in departments:
        lower = dept.lower().strip()

        # Strip leading prefix
        for prefix in STRIP_PREFIXES:
            if lower.startswith(prefix):
                short = lower[len(prefix):].strip()
                if short and short not in seen:
                    aliases.append(short)
                    seen.add(short)
                break

        # Extract suffix from "X College/School/Institute of Y"
        m = INFIX_RE.match(lower)
        if m:
            short = m.group(1).strip()
            if short and short not in seen and short not in GENERIC_SKIP:
                aliases.append(short)
                seen.add(short)

    # Manual aliases for common short forms used in queries
    MANUAL = [
        "financial aid", "registrar", "admissions", "bursar", "advising",
        "it helpdesk", "helpdesk", "kent", "kent law", "stuart", "armour",
        "lewis", "kaplan", "pritzker", "wiser", "ifsh",
        "student accounting", "graduate admission", "undergraduate admission",
        "student affairs", "academic affairs", "global services",
        "office of the registrar", "student accounting office",
        "dean of students", "dean of admissions", "dean of libraries",
    ]
    for alias in MANUAL:
        if alias not in seen:
            aliases.append(alias)
            seen.add(alias)

    return aliases


# Fee names too generic to count as specific tuition anchors
_GENERIC_FEE_NAMES = {"tuition", "fees", "fee", "cost", "costs", "rate", "rates"}


def _shorten_fee_names(fee_names: List[str], exclude_generic: bool = False) -> List[str]:
    """Truncate fee names to first 2 words and strip trailing punctuation.
    Optionally exclude generic names that don't add specificity."""
    result = []
    for f in fee_names:
        lower = f.lower().strip()
        if exclude_generic and lower in _GENERIC_FEE_NAMES:
            continue
        words = lower.split()
        short = " ".join(words[:2]) if len(words) > 2 else lower
        short = short.rstrip(":.,;")
        if short and (not exclude_generic or short not in _GENERIC_FEE_NAMES):
            result.append(short)
    return result


# ── Lazy pattern cache ────────────────────────────────────────────────────────

class _PatternCache:
    def __init__(self):
        self._contacts_strong  = None
        self._calendar_events  = None
        self._tuition_anchor   = None
        self._tuition_specific = None
        self._docs_topic       = None
        self._docs_type        = None

    def _get(self, attr: str, builder):
        val = getattr(self, attr)
        if val is None or val == []:
            result = builder()
            setattr(self, attr, result if result else [])
        return getattr(self, attr)

    @property
    def contacts_strong(self):
        def _build():
            departments = options_cache.contact_departments if _OPTIONS_AVAILABLE else []
            expanded = _expand_department_aliases(departments)
            return _build_pattern(expanded)
        return self._get("_contacts_strong", _build)

    @property
    def calendar_events(self):
        return self._get("_calendar_events", lambda: _build_pattern(
            options_cache.calendar_event_tokens if _OPTIONS_AVAILABLE else []
        ))

    @property
    def tuition_anchor(self):
        def _build():
            schools    = options_cache.tuition_schools   if _OPTIONS_AVAILABLE else []
            fee_names  = options_cache.tuition_fee_names if _OPTIONS_AVAILABLE else []
            levels     = options_cache.tuition_levels    if _OPTIONS_AVAILABLE else []
            years      = options_cache.tuition_years     if _OPTIONS_AVAILABLE else []
            structural = [
                "tuition", "fees", "fee", "rate", "rates", "cost", "costs",
                "price", "prices", "charge", "charges", "per credit", "credit hour",
                "mandatory", "activity fee", "health insurance", "student service",
                "graduate", "undergraduate", "full-time", "part-time",
                "full time", "part time", "billing", "llm", "mdes", "mdm",
                "per semester", "per term", "upass", "u-pass",
                "deposit", "admissions deposit", "application fee",
                "refund", "payment", "installment",
            ]
            short_fees = _shorten_fee_names(fee_names, exclude_generic=False)
            return _build_pattern(schools + short_fees + levels + years + structural)
        return self._get("_tuition_anchor", _build)

    @property
    def tuition_specific(self):
        """ES-derived anchors only — school, fee name, level, year.
        Used to detect underspecified queries that only match structural vocab."""
        def _build():
            schools   = options_cache.tuition_schools   if _OPTIONS_AVAILABLE else []
            fee_names = options_cache.tuition_fee_names if _OPTIONS_AVAILABLE else []
            levels    = options_cache.tuition_levels    if _OPTIONS_AVAILABLE else []
            years     = options_cache.tuition_years     if _OPTIONS_AVAILABLE else []
            short_fees = _shorten_fee_names(fee_names, exclude_generic=True)
            # Add common short forms that won't appear in 2-word truncations
            extra_specifics = ["upass", "u-pass", "deposit", "application fee"]
            return _build_pattern(schools + short_fees + levels + years + extra_specifics)
        return self._get("_tuition_specific", _build)

    @property
    def docs_topic(self):
        return self._get("_docs_topic", lambda: _build_pattern(
            _DOCS_CONFIG.get("topic_keywords", [])
        ))

    @property
    def docs_type(self):
        return self._get("_docs_type", lambda: _build_pattern(
            _DOCS_CONFIG.get("doc_type_keywords", [])
        ))


_patterns = _PatternCache()


# ── Shared helpers ────────────────────────────────────────────────────────────

def _options(values: List[str], max_items: int = 15) -> List[str]:
    if not _OPTIONS_AVAILABLE or not values:
        return []
    return values[:max_items]


def _clarify(message: str, options: List[str] = None) -> Dict[str, Any]:
    return {"needs_clarification": True, "message": message, "options": options or []}


def _match(pattern, text: str) -> bool:
    return bool(pattern.search(text)) if pattern else False


# ── Structural patterns (vocabulary, not domain data — always fixed) ──────────

_CALENDAR_TERM  = re.compile(r"\b(spring|fall|summer|winter|semester|term)\b", re.IGNORECASE)
_CALENDAR_WEAK  = re.compile(r"\b(deadline|due|start|end|begin|close|week|weeks|first day|last day|first class|last class|orientation)\b", re.IGNORECASE)
_CALENDAR_STRONG_HOLIDAY = re.compile(
    r"\b(thanksgiving|christmas|labor day|memorial day|juneteenth|independence day|"
    r"martin luther king|mlk|new year|spring break|fall break|winter break)\b",
    re.IGNORECASE
)
_CALENDAR_DATE  = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|20\d{2})\b", re.IGNORECASE
)
_CALENDAR_SPECIFIC_TERM = re.compile(
    r"\b(spring|fall|summer|winter|term a|term b|coursera)\b", re.IGNORECASE
)
_CONTACT_SLOTS  = re.compile(
    r"\b(phone|phones|email|emails|call|number|numbers|fax|contact|contacts|"
    r"address|addresses|handles|handle|manages|manage|responsible|reach|"
    r"in charge|who do i|who should i|who can i|who is|who's|is there a|are there)\b",
    re.IGNORECASE,
)
_LOCATION_SLOTS = re.compile(
    r"\b(where|building|buildings|location|locations|room|rooms|campus|floor|floors)\b",
    re.IGNORECASE,
)
_GENERIC_UNIT   = re.compile(
    r"\b(department|departments|office|offices|program|programs|school|schools|college|colleges)\b",
    re.IGNORECASE,
)
_GENERIC_MONEY  = re.compile(
    r"\b(pay|paying|money|afford|expensive|cheap|bill|billing)\b", re.IGNORECASE
)
_GENERIC_DOC    = re.compile(r"\b(document|documents|info|information)\b", re.IGNORECASE)


# Curated contacts clarification list (offices/schools shown when the query is underspecified)
CONTACT_DEPT_PICKER_OPTIONS = [
    "Registrar", "Financial Aid", "Student Accounting", "Academic Affairs",
    "Admissions", "Student Affairs", "Global Services",
    "Armour College of Engineering", "College of Computing",
    "College of Architecture", "Chicago-Kent College of Law",
    "Stuart School of Business", "Lewis College of Science and Letters",
    "Institute of Design", "Computer Science", "Physics", "Chemistry",
]


def contact_reply_matches_picker_option(text: str) -> bool:
    """
    True when the user message selects one of the contacts clarification options
    (exact label, or a single word that matches an option, punctuation-tolerant).
    Used so pending-clarification follow-ups are always treated as ANSWER, not NEW_TOPIC.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    lower = raw.lower()
    if any(lower == o.lower() for o in CONTACT_DEPT_PICKER_OPTIONS):
        return True
    words = raw.split()
    if len(words) == 1:
        w = words[0].strip(string.punctuation).lower()
        return any(w == o.lower() for o in CONTACT_DEPT_PICKER_OPTIONS)
    return any(lower == o.lower() for o in CONTACT_DEPT_PICKER_OPTIONS)


def _mentions_dept_picker_option(q_lower: str) -> bool:
    """Substring match for multi-word options (e.g. Chicago-Kent College of Law)."""
    for opt in CONTACT_DEPT_PICKER_OPTIONS:
        if opt.lower() in q_lower:
            return True
    return False


# ── Calendar ──────────────────────────────────────────────────────────────────

def calendar_query_validation(query: str) -> Dict[str, Any]:
    q = (query or "").lower()
    term_opts = _options(options_cache.calendar_terms if _OPTIONS_AVAILABLE else [])

    if len(q.split()) <= 1:
        return _clarify("Could you provide a bit more detail about your question?", term_opts)

    # Previous broad pass-through kept for easy rollback:
    # # 3+ word queries: the semantic router already picked CALENDAR — trust it
    # if len(q.split()) >= 3:
    #     return {"needs_clarification": False, "options": []}

    has_term = bool(_CALENDAR_TERM.search(q))
    has_specific_term = bool(_CALENDAR_SPECIFIC_TERM.search(q))
    has_weak_event = bool(_CALENDAR_WEAK.search(q))
    has_strong_event = _match(_patterns.calendar_events, q) or bool(_CALENDAR_STRONG_HOLIDAY.search(q))
    has_date = bool(_CALENDAR_DATE.search(q))
    has_specific_context = has_specific_term or has_date

    # Strong, event-specific requests are safe when term/date is explicit.
    if has_strong_event and has_specific_context:
        return {"needs_clarification": False, "options": []}

    # Holiday-only questions are usually specific enough to answer directly.
    if bool(_CALENDAR_STRONG_HOLIDAY.search(q)):
        return {"needs_clarification": False, "options": []}

    # Weak timeline/event asks without term/date are underspecified.
    if has_weak_event and not has_specific_context:
        return _clarify(
            "Which semester or year are you referring to?",
            term_opts,
        )

    # General calendar asks with explicit term/date can proceed.
    if has_specific_context:
        return {"needs_clarification": False, "options": []}

    # Generic "semester/term" mentions still need disambiguation.
    if has_term and not has_specific_context:
        return _clarify("Which semester or year are you referring to?", term_opts)

    return _clarify(
        "Could you clarify what part of the academic calendar you are asking about? "
        "For example: a specific exam period, drop deadline, break, or graduation date.",
        term_opts,
    )


# ── Contacts ──────────────────────────────────────────────────────────────────

def contacts_query_validation(query: str) -> Dict[str, Any]:
    
    q = (query or "").lower()

    # Show a curated list of common offices rather than ES department.keyword
    # values (which are individual job titles, not useful to a student).
    dept_opts = CONTACT_DEPT_PICKER_OPTIONS

    # One-word answers that match the clarification picker (e.g. "Physics") are enough to search.
    words = q.split()
    if len(words) == 1:
        w0 = words[0].strip(string.punctuation).lower()
        if any(opt.lower() == w0 for opt in dept_opts):
            return {"needs_clarification": False, "options": []}

    if len(q.split()) <= 1:
        return _clarify("Which department or office are you looking for?", dept_opts)

    # Previous broad pass-through kept for easy rollback:
    # # 3+ word queries: the semantic router already picked CONTACTS — trust it
    # if len(q.split()) >= 3:
    #     return {"needs_clarification": False, "options": []}

    has_strong_unit = _match(_patterns.contacts_strong, q)
    has_contact_slot = bool(_CONTACT_SLOTS.search(q))
    has_location_slot = bool(_LOCATION_SLOTS.search(q))
    has_generic_unit = bool(_GENERIC_UNIT.search(q))

    if has_strong_unit:
        return {"needs_clarification": False, "options": []}

    # Broad support asks ("who should I contact about X?") are still answerable.
    if has_contact_slot and not has_generic_unit:
        return {"needs_clarification": False, "options": []}

    # Location questions for a concrete unit can proceed.
    if has_location_slot and has_generic_unit and len(q.split()) >= 4:
        return {"needs_clarification": False, "options": []}

    # Reformulated follow-ups often name the department but omit "phone/email/contact"
    # (e.g. "What is the Physics department?") — still a concrete directory ask.
    # Also catches clarification answers like "I need to speak with Computer Science".
    if _mentions_dept_picker_option(q):
        return {"needs_clarification": False, "options": []}

    return _clarify(
        "Could you specify which department or office and what kind of information you need?",
        dept_opts,
    )

# ── Tuition ───────────────────────────────────────────────────────────────────

_TUITION_LEVEL = re.compile(
    r"\b(graduate|grad|undergraduate|undergrad|masters|doctoral|phd|llm|mdes|mdm)\b",
    re.IGNORECASE
)

# Short aliases for schools that ES aggregations store as full names
_TUITION_SCHOOL_ALIASES = re.compile(
    r"\b(kent|chicago[\s-]kent|law\s*school|stuart|business\s*school|mies|"
    r"institute\s*of\s*design|iep|intensive\s*english)\b",
    re.IGNORECASE
)

def tuition_query_validation(query: str) -> Dict[str, Any]:
    q = (query or "").lower()
    school_opts = _options(options_cache.tuition_schools if _OPTIONS_AVAILABLE else [])

    if len(q.split()) <= 1:
        w = q.strip(string.punctuation)
        if school_opts and any(w == opt.lower() for opt in school_opts):
            return {"needs_clarification": False, "options": []}
        return _clarify("Which school or program are you asking about?", school_opts)

    # Previous broad pass-through kept for easy rollback:
    # # 3+ word queries: the semantic router already picked TUITION — trust it
    # if len(q.split()) >= 3:
    #     return {"needs_clarification": False, "options": []}

    has_anchor = _match(_patterns.tuition_anchor, q)
    has_specific = (_match(_patterns.tuition_specific, q)
                    or bool(_TUITION_LEVEL.search(q))
                    or bool(_TUITION_SCHOOL_ALIASES.search(q)))
    has_generic_money = bool(_GENERIC_MONEY.search(q))
    has_pricing_intent = bool(re.search(r"\b(tuition|fee|fees|cost|costs|rate|rates|price|prices|billing|bill|per credit|credit hour)\b", q))

    # If routed here due overlap terms like "full-time/part-time" but no real tuition intent,
    # avoid forcing a tuition clarification and let downstream retrieval answer.
    if has_anchor and not has_pricing_intent:
        return {"needs_clarification": False, "options": []}

    if has_specific:
        return {"needs_clarification": False, "options": []}

    # Tuition intent exists, but no school/level/fee/year specificity.
    if has_anchor and not has_specific:
        return _clarify(
            "Could you specify which school or program level this is for?",
            school_opts,
        )

    if has_generic_money:
        return _clarify(
            "Could you specify what you need about tuition and fees? "
            "For example: which school, program level, or a specific fee.",
            school_opts,
        )

    return _clarify(
        "Could you specify what you need about tuition and fees? "
        "For example: which school, program level, or a specific fee.",
        school_opts,
    )


# ── Documents ─────────────────────────────────────────────────────────────────

def documents_query_validation(query: str) -> Dict[str, Any]:
    q = (query or "").lower()

    if len(q.split()) <= 1:
        return _clarify("Could you provide a bit more detail about your question?")

    # Previous broad pass-through kept for easy rollback:
    # # 3+ word queries: the semantic router already picked DOCUMENTS — trust it
    # if len(q.split()) >= 3:
    #     return {"needs_clarification": False, "options": []}

    has_topic    = _match(_patterns.docs_topic, q)
    has_doc_type = _match(_patterns.docs_type, q)
    has_generic_doc = bool(_GENERIC_DOC.search(q))
    if has_topic or has_doc_type:
        return {"needs_clarification": False, "options": []}

    # If the query is long and policy-like but doesn't match curated keywords,
    # let retrieval attempt an answer instead of over-clarifying.
    if len(q.split()) >= 5 and not has_generic_doc:
        return {"needs_clarification": False, "options": []}

    return _clarify(
        "Could you specify what university policy or document you are looking for? "
        "For example: GPA requirements, housing policy, visa rules, or academic probation."
    )