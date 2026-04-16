"""
FastAPI endpoint for Chatbot B (university chatbot backend).

Run with:
    uvicorn api_app:app --reload --host 0.0.0.0 --port 8000

Interactive docs at:
    http://localhost:8000/docs
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Project bootstrap (same as the Streamlit apps) ────────────────────────────
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)
load_dotenv(project_root / ".env")

# ── Stub out streamlit so the import of app_with_clarification_memory works ───
# That module uses `st.error`, `st.session_state`, `st.set_page_config`, etc.
# at import time and inside functions.  We provide a lightweight no-op shim
# so everything loads cleanly without an actual Streamlit runtime.
import types

_st_shim = types.ModuleType("streamlit")

# Generic no-op that can be used as a function call OR a context manager
class _NoOp:
    """Returned by every shimmed Streamlit function. Supports `with` blocks."""
    def __call__(self, *a, **kw):
        return self
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def __iter__(self):
        return iter([])
    def __bool__(self):
        return False

_noop = _NoOp()
_noop_fn = lambda *a, **kw: _noop

# Assign all commonly used Streamlit functions to the no-op
for _name in (
    "error", "stop", "set_page_config", "title", "chat_input", "chat_message",
    "markdown", "write_stream", "spinner", "expander", "columns", "form",
    "write", "json", "caption", "text_input", "slider", "text_area",
    "form_submit_button", "button", "info", "success", "warning", "rerun",
    "sidebar", "header", "subheader",
):
    setattr(_st_shim, _name, _noop_fn)

# Decorators: handle both @st.cache_resource and @st.cache_data(show_spinner=False)
def _make_cache_decorator(*args, **kwargs):
    # @st.cache_resource  (no parens) → args[0] is the function itself
    if args and callable(args[0]):
        return args[0]
    # @st.cache_resource() or @st.cache_data(show_spinner=False) → return pass-through
    return lambda fn: fn

_st_shim.cache_data = _make_cache_decorator
_st_shim.cache_resource = _make_cache_decorator

# session_state as a dict that also supports attribute access (like real Streamlit)
class _AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)

_st_shim.session_state = _AttrDict()

sys.modules["streamlit"] = _st_shim

# NOW import the real chatbot logic (it will see our shim as "streamlit")
from ui.app_with_clarification_memory import (
    classify_pending_response,
    get_answer,
    get_answer_for_domain,
    reformulate_query,
    GROQ_API_KEY,
    DOMAIN_TUITION,
    DOMAIN_CONTACTS,
    DOMAIN_CALENDAR,
)
from common.clarification_options import options_cache
from common.slot_filling import CONTACT_DEPT_PICKER_OPTIONS
from common.es_client import es

def _strip_markdown(text: str) -> str:
    """Convert markdown to plain text for frontends that don't render it."""
    s = text
    # Headers: ### Title → Title
    s = re.sub(r"^#{1,6}\s+", "", s, flags=re.MULTILINE)
    # Bold/italic: **text** or *text* → text
    s = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", s)
    # Bullet points: keep as-is (- item), they read fine as plain text
    # Strip trailing pleasantries the LLM sometimes adds despite instructions
    s = re.sub(
        r"\n*(?:Let me know if you [\w\s,!.]*"
        r"|Feel free to [\w\s,!.]*"
        r"|(?:I )?hope (?:this|that) helps[\w\s,!.]*"
        r"|If you (?:have|need) [\w\s,!.]*)\s*$",
        "", s, flags=re.IGNORECASE
    )
    return s.rstrip()

def _options_for_domain(domain: str) -> list[str]:
    """Re-derive clarification options from domain when frontend doesn't pass them back."""
    if domain == DOMAIN_TUITION:
        return list(options_cache.tuition_schools or [])
    if domain == DOMAIN_CONTACTS:
        return list(CONTACT_DEPT_PICKER_OPTIONS)
    if domain == DOMAIN_CALENDAR:
        return list(options_cache.calendar_terms or [])
    return []

def _is_known_contact_name(prompt: str) -> bool:

    words = [w for w in (prompt or "").split() if w]
    if len(words) < 2 or len(words) > 4:
        return False
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z'-]*", w) for w in words):
        return False
    try:
        res = es.search(
            index="iit_contacts",
            body={
                "size": 3,
                "query": {
                    "match_phrase": {
                        "name": prompt
                    }
                },
            },
        )
        target = " ".join(words).lower()
        for hit in res.get("hits", {}).get("hits", []):
            name = ((hit.get("_source", {}) or {}).get("name") or "").strip().lower()
            if name == target:
                return True
        return False
    except Exception:
        return False

def _extract_contact_candidate(prompt: str) -> str:

    text = (prompt or "").strip()
    if not text:
        return ""
    m = re.match(r"(?i)\s*(?:who\s+is|contact\s+(?:for|info\s+for)|info\s+for)\s+(.+?)\s*$", text)
    if not m:
        return ""
    candidate = m.group(1).strip().strip("?.!,")
    tokens = [t for t in candidate.split() if t]
    if 2 <= len(tokens) <= 4 and all(re.fullmatch(r"[A-Za-z][A-Za-z'-]*", t) for t in tokens):
        return " ".join(tokens)
    return ""

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="IIT University Chatbot API",
    description=(
        "Backend API for Chatbot B — Illinois Tech's university assistant. "
        "Handles calendar, contacts, tuition, and policy questions."
    ),
    version="1.0.0",
)

# Allow the hosted Streamlit app (or any frontend) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────

class PendingContext(BaseModel):
    """Clarification state passed back and forth between client and API."""
    original_query: str | None = None
    clarification_message: str | None = None
    domain: str | None = None
    clarification_options: list[str] = []

class ChatMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    """What the client sends to /ask."""
    prompt: str = Field(..., description="The user's question")
    topic: str | None = Field(None, description="Optional topic filter (e.g. 'Academic Calendar')")
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation turns for context",
    )
    pending_context: PendingContext | None = Field(
        None,
        description="If the previous response was a clarification, send this back with the user's answer",
    )

class AskResponse(BaseModel):
    """What the API returns from /ask."""
    response: str = Field(..., description="The chatbot's answer or clarification question")
    sources: list[str] = Field(default_factory=list, description="Source URLs for the answer")
    is_clarification: bool = Field(False, description="True if the response is asking for more info")
    pending_context: PendingContext | None = Field(
        None,
        description="Non-null when is_clarification=True. Send this back with the user's next message.",
    )
    route_details: dict[str, Any] = Field(default_factory=dict, description="Router debug info")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "model": "chatbot_b"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Main chatbot endpoint. Mirrors the logic in design_app.py's
    get_model_b_response, but stateless — clarification state is
    passed through pending_context.
    """
    prompt = (req.prompt or "").strip()
    if not prompt:
        return AskResponse(response="Please enter a question.")

    # Convert chat_history to the list[dict] format the backend expects
    chat_history = [{"role": m.role, "content": m.content} for m in req.chat_history]

    pending = req.pending_context

    answer = ""
    sources: list[str] = []
    route_details: dict = {}
    is_clarification = False
    clar_msg = ""
    clar_domain = ""

    if pending and pending.original_query and pending.clarification_message:
        # We're in a clarification follow-up
        opts = pending.clarification_options or _options_for_domain(pending.domain or "")
        action = classify_pending_response(
            pending.original_query,
            pending.clarification_message,
            prompt,
            opts,
        )

        if action == "CANCEL":
            return AskResponse(
                response="No problem — ask me another Illinois Tech question whenever you are ready."
            )

        if action == "NEW_TOPIC":
            # User changed topic — fresh query
            answer, sources, route_details, is_clarification, clar_msg, clar_domain, _clar_opts = get_answer(
                query=prompt, chat_history=chat_history
            )
        else:
            # ANSWER — combine original + user clarification
            combined = (
                reformulate_query(pending.original_query, prompt)
                if GROQ_API_KEY
                else f"{pending.original_query} {prompt}".strip()
            )
            answer, sources, route_details, is_clarification, clar_msg, clar_domain, _clar_opts = get_answer_for_domain(
                combined,
                pending.domain or "",
                chat_history=[],
            )
    else:
        # Fresh question — but if it's a bare known school/dept/term name, go straight
        # to the right domain so the rewrite pipeline doesn't lose the entity.
        bare = prompt.strip().lower()
        tuition_schools = [s.lower() for s in (options_cache.tuition_schools or [])]
        calendar_terms  = [t.lower() for t in (options_cache.calendar_terms  or [])]
        contact_opts    = [o.lower() for o in CONTACT_DEPT_PICKER_OPTIONS]

        # Detect bare proper names (e.g. "Yuhan Ding") — route straight to CONTACTS.
        # Guards: 2-3 purely alphabetic words, no digits (rules out "Fall 2026"),
        # no question/punctuation, and not a known calendar/academic keyword.
        # Case-insensitive: title-case the prompt for the check so "yuhan ding"
        # Require at least one word to start with uppercase — real names always
        # have at least one capital (Yuhan ding, YUHAN DING, Yuhan Ding all pass;
        # "hi there", "whats the cost", "help me" all fail and fall through normally).
        _NON_NAME_WORDS = {
            # Seasons / time
            "fall", "spring", "summer", "winter",
            # Months
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            # Academic calendar
            "holiday", "holidays", "break", "breaks", "schedule", "schedules",
            "deadline", "deadlines", "term", "semester", "session", "orientation",
            "finals", "final", "exam", "exams", "week", "day", "year",
            "commencement", "graduation", "convocation",
            "early", "departure", "midterm", "midterms", "grading", "begins",
            # Registration / academics
            "registration", "coursera", "campus", "course", "courses",
            "class", "classes", "credit", "credits", "load", "limit",
            "add", "drop", "withdraw", "withdrawal", "audit", "overload",
            "grade", "grades", "appeal", "transcript", "enrollment",
            "transfer", "abroad", "study", "research", "advising",
            # People/role words (not names)
            "student", "students", "faculty", "staff", "advisor", "dean",
            "professor", "instructor", "new", "office", "hours",
            # Misc
            "honor", "honors", "roll", "list", "labor", "policy", "policies",
            "fee", "fees", "never", "mind",
        }
        _DIRECT_NON_CONTACT_ANCHORS = {
            "tuition", "fee", "fees", "cost", "costs", "rate", "rates",
            "policy", "policies", "rule", "rules", "probation", "gpa",
            "housing", "visa", "handbook", "calendar", "holiday", "holidays",
            "deadline", "deadlines", "break", "breaks", "semester", "term",
            "document", "documents",
        }
        _prompt_words = prompt.split()
        _prompt_title = prompt.title()
        _has_non_contact_anchor = any(w.lower() in _DIRECT_NON_CONTACT_ANCHORS for w in _prompt_words)
        _is_proper_name = (
            len(_prompt_words) in (2, 3)
            and all(re.fullmatch(r"[A-Za-z][A-Za-z'-]*", w) for w in _prompt_words if w)
            and not any(c in prompt for c in ("?", "!", "@", ","))
            and not any(w.lower() in _NON_NAME_WORDS for w in _prompt_words)
            and not _has_non_contact_anchor
        )
        _candidate_name = _extract_contact_candidate(prompt)
        _is_known_contact = _is_known_contact_name(prompt) or (_is_known_contact_name(_candidate_name) if _candidate_name else False)

        if bare in tuition_schools:
            answer, sources, route_details, is_clarification, clar_msg, clar_domain, _clar_opts = get_answer_for_domain(
                f"What are the tuition rates for all student levels at {prompt}?", DOMAIN_TUITION, chat_history=[]
            )
        elif bare in calendar_terms:
            answer, sources, route_details, is_clarification, clar_msg, clar_domain, _clar_opts = get_answer_for_domain(
                prompt, DOMAIN_CALENDAR, chat_history=[]
            )
        elif bare in contact_opts:
            answer, sources, route_details, is_clarification, clar_msg, clar_domain, _clar_opts = get_answer_for_domain(
                prompt, DOMAIN_CONTACTS, chat_history=[]
            )
        elif _is_proper_name or _is_known_contact:
            name_for_query = _candidate_name if _candidate_name else _prompt_title
            answer, sources, route_details, is_clarification, clar_msg, clar_domain, _clar_opts = get_answer_for_domain(
                f"contact information for {name_for_query}", DOMAIN_CONTACTS, chat_history=[]
            )
        else:
            answer, sources, route_details, is_clarification, clar_msg, clar_domain, _clar_opts = get_answer(
                query=prompt, chat_history=chat_history
            )

    # Build response
    resp_pending = None
    if is_clarification:
        resp_pending = PendingContext(
            original_query=prompt,
            clarification_message=clar_msg,
            domain=clar_domain or None,
            clarification_options=_clar_opts or [],
        )

    answer = (answer or "").strip()
    if not answer:
        answer = (
            "I couldn't generate a reply just now. If searches keep failing, "
            "confirm Elasticsearch is running (port 9200) and try again."
        )

    # Strip markdown for frontends that render plain text (e.g. Azure)
    answer = _strip_markdown(answer)

    return AskResponse(
        response=answer,
        sources=sources,
        is_clarification=is_clarification,
        pending_context=resp_pending,
        route_details=route_details,
    )
