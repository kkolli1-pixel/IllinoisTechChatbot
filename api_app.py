"""
FastAPI endpoint for Chatbot B (university chatbot backend).

Run with:
    uvicorn api_app:app --reload --host 0.0.0.0 --port 8000

Interactive docs at:
    http://localhost:8000/docs
"""

import json
import os
import sys
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
)

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
        action = classify_pending_response(
            pending.original_query,
            pending.clarification_message,
            prompt,
            pending.clarification_options or [],
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
        # Fresh question
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

    return AskResponse(
        response=answer,
        sources=sources,
        is_clarification=is_clarification,
        pending_context=resp_pending,
        route_details=route_details,
    )
