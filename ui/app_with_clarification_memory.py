import logging
import os
import json
import re
import string
import sys
from pathlib import Path

_log = logging.getLogger(__name__)

import requests
import streamlit as st
from dotenv import load_dotenv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

avatar_path = project_root / "images" / "kHqY4UrI_400x400 Background Removed.png"
assistant_avatar = str(avatar_path) if avatar_path.exists() else None

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

load_dotenv(project_root / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

try:
    from router.router import get_routing_intent, DOMAIN_CALENDAR, DOMAIN_CONTACTS, DOMAIN_DOCUMENTS, DOMAIN_TUITION
    from router.calendar_router import route_query as calendar_route_query
    from search.calendar_search import calendar_rrf_search
    from search.contacts_search import contacts_rrf_search
    from search.documents_search import documents_rrf_search
    from search.tuition_search import tuition_rrf_search
except ImportError as e:
    st.error(f"Could not import necessary modules: {e}")
    st.stop()

import common.slot_filling as _slot_mod

_CONTACT_PICKER_OPTS = (
    "Registrar", "Financial Aid", "Student Accounting", "Academic Affairs",
    "Admissions", "Student Affairs", "Global Services",
    "Armour College of Engineering", "College of Computing",
    "College of Architecture", "Chicago-Kent College of Law",
    "Stuart School of Business", "Lewis College of Science and Letters",
    "Institute of Design", "Computer Science", "Physics", "Chemistry",
)

def _contact_reply_matches_picker_option_fallback(text: str) -> bool:
    """Same logic as slot_filling.contact_reply_matches_picker_option if that symbol is missing."""
    raw = (text or "").strip()
    if not raw:
        return False
    lower = raw.lower()
    if any(lower == o.lower() for o in _CONTACT_PICKER_OPTS):
        return True
    words = raw.split()
    if len(words) == 1:
        w = words[0].strip(string.punctuation).lower()
        return any(w == o.lower() for o in _CONTACT_PICKER_OPTS)
    return any(lower == o.lower() for o in _CONTACT_PICKER_OPTS)

contact_reply_matches_picker_option = getattr(
    _slot_mod,
    "contact_reply_matches_picker_option",
    _contact_reply_matches_picker_option_fallback,
)

# ── Small LLM helper (Groq only) ───────────────────────────────────────────────

def _groq_call(messages: list, max_tokens: int = 60) -> str:
    """Minimal Groq call for intent/reformulation tasks."""
    if not GROQ_API_KEY:
        return ""
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

def _openai_synthesis(messages: list) -> str:
    """Azure OpenAI synthesis call. Returns empty string on failure."""
    if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY and AZURE_OPENAI_DEPLOYMENT):
        return ""
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        _log.warning("Azure OpenAI synthesis failed: %s", e)
        return ""

def is_escape(clarification_message: str, user_response: str) -> bool:
    """
    Returns True if the user is abandoning the clarification rather than answering it.

    Short zero-content responses are caught by a filler-word check to avoid
    LLM misclassification. Anything longer or ambiguous goes to the LLM.
    """
    response = user_response.strip().lower()
    words = response.split()

    # Short responses made entirely of filler → escape without an LLM call.
    FILLER = {
        "no", "nope", "nah", "never", "mind", "nevermind",
        "never mind", "forget", "it", "forget it", "cancel",
        "stop", "actually", "hmm", "uh", "um", "ok", "okay",
        "sure", "wait", "not", "nothing",
    }
    if len(words) <= 3 and all(w in FILLER for w in words):
        return True

    # Level words are always valid answers — never treat as escape
    LEVEL_WORDS = {"graduate", "grad", "undergraduate", "undergrad", "masters", "doctoral", "phd"}
    if len(words) <= 2 and all(w in LEVEL_WORDS for w in words):
        return False

    result = _groq_call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a classification assistant. "
                    "Reply with only YES or NO — nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"A university chatbot asked: \"{clarification_message}\"\n"
                    f"User replied: \"{user_response}\"\n\n"
                    f"Does the user's reply contain specific information that helps answer "
                    f"the question (such as a semester, year, department name, school name, "
                    f"program name, fee type, or student level like graduate/undergraduate)?\n\n"
                    f"Important: proper nouns, institution names, school names, department "
                    f"names, student levels (graduate, undergraduate, grad, undergrad), "
                    f"and any direct answer to the question are always YES — even if "
                    f"they are short or unfamiliar words.\n\n"
                    f"Reply with only YES or NO."
                ),
            },
        ],
        max_tokens=5,
    )
    return result.strip().upper().startswith("NO")

def classify_pending_response(original_query: str, clarification_message: str, user_response: str, clarification_options: list = None) -> str:
    """
    Classify a response to a pending clarification as one of:
    - ANSWER: the user is answering the clarification
    - NEW_TOPIC: the user changed topic and asked a fresh question
    - CANCEL: the user is explicitly abandoning the pending clarification

    Falls back to the legacy escape-vs-answer behavior if Groq is unavailable.
    """
    response = (user_response or "").strip()
    if not response:
        return "ANSWER"

    if clarification_options:
        response_lower = response.lower()
        for opt in clarification_options:
            if response_lower == (opt or "").strip().lower():
                return "ANSWER"

    if contact_reply_matches_picker_option(user_response):
        return "ANSWER"

    if not GROQ_API_KEY:
        return "NEW_TOPIC" if is_escape(clarification_message, response) else "ANSWER"

    result = _groq_call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are classifying a user's response to a pending clarification in a "
                    "university chatbot. Reply with exactly one label: ANSWER, NEW_TOPIC, or CANCEL."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original user question: \"{original_query}\"\n"
                    f"Clarification asked by chatbot: \"{clarification_message}\"\n"
                    f"User response: \"{user_response}\"\n\n"
                    "Choose exactly one label:\n"
                    "- ANSWER: the user is still responding to the clarification, even if the response "
                    "is incomplete, ambiguous, or may require another clarification.\n"
                    "- NEW_TOPIC: the user asks a different university-related question instead of answering.\n"
                    "- CANCEL: the user explicitly abandons or dismisses the pending clarification.\n\n"
                    "Student levels like graduate/undergraduate, school names, department names, "
                    "dates, semesters, years, fee types, and short follow-up details count as ANSWER.\n"
                    "If the user refers to the same requested concept without adding enough detail, "
                    "such as saying 'the deadline' after being asked which deadline or semester, "
                    "that is still ANSWER, not NEW_TOPIC.\n"
                    "Replies like 'never mind' or 'forget it' count as CANCEL.\n"
                    "A fresh question like 'What is the plagiarism policy?' counts as NEW_TOPIC.\n\n"
                    "Reply with only one label."
                ),
            },
        ],
        max_tokens=5,
    ).strip().upper()

    if result in {"ANSWER", "NEW_TOPIC", "CANCEL"}:
        return result

    return "NEW_TOPIC" if is_escape(clarification_message, response) else "ANSWER"

def reformulate_query(original_query: str, user_clarification: str) -> str:
    """
    Use the LLM to combine the original question and the user's clarification
    into one clean, standalone question. Falls back to simple concatenation
    if the LLM call fails.
    """
    result = _groq_call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a query reformulation assistant for a university chatbot. "
                    "Combine the original question and the clarification into one clear, "
                    "complete, standalone question. Return only the reformulated question — "
                    "no explanation, no punctuation changes beyond what is natural. "
                    "The user's clarification is a slot value (school name, program level, "
                    "fee type, department name, etc.) that must be inserted directly into "
                    "the question as-is — do not paraphrase or reinterpret it."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {original_query}\n"
                    f"User clarification: {user_clarification}\n\n"
                    f"Reformulated question:"
                ),
            },
        ],
        max_tokens=80,
    )
    return result if result else f"{original_query} {user_clarification}".strip()

def is_followup_query(query: str, prev_user: str) -> bool:
    """
    Returns True if the query is a follow-up to the previous question
    rather than a standalone question. Uses LLM classification.
    Falls back to False if no prior context or Groq unavailable.
    """
    if not prev_user or not GROQ_API_KEY:
        return False

    result = _groq_call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a classification assistant. "
                    "Reply with only YES or NO — nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Previous question: \"{prev_user}\"\n"
                    f"Current query: \"{query}\"\n\n"
                    f"Is the current query a follow-up or continuation of the previous question "
                    f"(i.e. it references something from the previous question using pronouns, "
                    f"partial phrases, or implicit context)? "
                    f"A standalone question that could be asked independently is NOT a follow-up.\n\n"
                    f"Reply with only YES or NO."
                ),
            },
        ],
        max_tokens=5,
    )
    return result.strip().upper().startswith("YES")

def _looks_multi_part_query(query: str) -> bool:
    """
    Heuristic for user questions that are explicitly asking for multiple pieces
    of information. We keep secondary-domain clarifications alive for these.
    """
    q = (query or "").strip().lower()
    if not q:
        return False
    return bool(re.search(r"\b(and|also|plus)\b", q))

# ── Source and context builders ───────────────────────────────────────────────

def build_sources(hits):
    urls = []
    for h in hits:
        s = h.get("_source", {})
        source_urls = s.get("source_urls") or ([s.get("source_url")] if s.get("source_url") else [])
        for u in source_urls:
            u = (u or "").strip()
            if u and u not in urls:
                urls.append(u)
    return urls

def build_calendar_context(hits):
    context = []
    for h in hits:
        s = h.get("_source", {})
        start = (s.get("start_date") or "").strip()
        end = (s.get("end_date") or "").strip()
        event_name = (s.get("event_name") or "").strip()
        term = (s.get("term") or "").strip()
        semantic_text = (s.get("semantic_text") or "").strip()

        date_str = start if start and start == end else (f"{start} → {end}" if start or end else "N/A")
        header = f"[Calendar: {term} | {date_str}]"

        # Use semantic_text when available — it's enriched with synonyms and
        # plain-language descriptions that help the LLM interpret event names.
        # Fall back to event_name only if semantic_text is missing.
        body = semantic_text if semantic_text else event_name
        context.append(f"{header}\n{body}")
    return context

def build_contacts_context(hits):
    context = []
    for h in hits:
        s = h.get("_source", {})
        name = (s.get("name") or "").strip()
        dept = (s.get("department") or "").strip()
        email = (s.get("email") or "").strip()
        phone = (s.get("phone") or "").strip()
        building = (s.get("building") or "").strip()
        address = (s.get("address") or "").strip()
        category = (s.get("category") or "").strip()
        block = f"[Contact: {name}]"
        if category:
            block += f"\nCategory: {category}"
        if dept:
            block += f"\nDepartment: {dept}"
        if email:
            block += f"\nEmail: {email}"
        if phone:
            block += f"\nPhone: {phone}"
        if building:
            block += f"\nBuilding: {building}"
        if address:
            block += f"\nAddress: {address}"
        context.append(block)
    return context

def build_documents_context(hits):
    context = []
    for h in hits:
        s = h.get("_source", {})
        topic = (s.get("topic") or "").strip()
        doc_name = (s.get("doc_name") or "").strip()
        content = (s.get("content") or "").strip()
        block = f"[Policy Section: {topic} | Handbook: {doc_name}]\n{content}" if (topic or doc_name) else content
        context.append(block)
    return context

def build_tuition_context(hits):
    context = []
    for h in hits:
        s = h.get("_source", {})
        school = (s.get("school") or "").strip()
        level = (s.get("level") or "").strip()
        section = (s.get("section") or "").strip()
        fee_name = (s.get("fee_name") or "").strip()
        academic_year = (s.get("academic_year") or "").strip()
        term = (s.get("term") or "").strip()
        unit = (s.get("unit") or "").strip()
        amount = s.get("amount_value")
        amount_str = f"{amount}" if amount is not None else "N/A"
        content = (s.get("content") or "").strip()
        header = f"[Tuition: {school} | {level} | {academic_year}]"
        if term:
            header += f" {term}"
        line = f"{section} | {fee_name}: {amount_str}" + (f" ({unit})" if unit else "")
        block = f"{header}\n{line}"
        if content:
            block += f"\n{content}"
        context.append(block)
    return context

# ── Main answer orchestrator ──────────────────────────────────────────────────

def _format_clarification(message: str, options: list) -> str:
    """Append options to a clarification message as a readable suggestion line."""
    if not options:
        return message
    opts_str = " · ".join(options)
    return f"{message}\n\n*Options: {opts_str}*"

def _append_partial_clarification(answer: str, message: str, options: list) -> str:
    clarification_text = _format_clarification(message, options)
    answer_text = (answer or "").strip()
    if not answer_text:
        return clarification_text
    return (
        f"{answer_text}\n\n"
        f"To answer the rest of your question, I need one clarification:\n"
        f"{clarification_text}"
    )

def _off_topic_short_reply(query: str) -> str | None:
    """
    Short greetings / meta messages that should not hit the semantic router.
    Used when there is no pending clarification turn.
    """
    s = (query or "").strip().lower()
    if not s or len(s) > 140:
        return None
    exact = {
        "how are you",
        "how are you?",
        "what's up",
        "whats up",
        "whats up?",
        "what's up?",
        "sup",
        "hey",
        "hi",
        "hello",
        "hi there",
        "hello there",
        "thanks",
        "thanks!",
        "thank you",
        "thank you!",
        "thx",
        "ty",
        "ok",
        "okay",
        "ok!",
        "cool",
        "nice",
        "never mind",
        "nevermind",
        "forget it",
        "no thanks",
        "no thank you",
        "it's broken",
        "its broken",
        "this is broken",
    }
    if s in exact:
        return (
            "I am here for Illinois Tech calendar, contacts, tuition, and policy questions. "
            "What would you like to know?"
        )
    if s in {"what?", "what", "huh", "huh?"}:
        return (
            "I did not quite catch that — what do you need about the calendar, directory, tuition, or policies?"
        )
    return None

def _previous_user_utterance(chat_history: list | None) -> str:
    """Return the second-to-last user message (previous turn), or empty string."""
    if not chat_history:
        return ""
    users = [
        (t.get("content") or "").strip()
        for t in chat_history
        if t.get("role") == "user" and (t.get("content") or "").strip()
    ]
    if len(users) < 2:
        return ""
    return users[-2]

def rewrite_query(query: str, domains: list, context_hint: str = "") -> str:
    """
    Rewrite a natural language query into retrieval-ready language.
    Converts colloquial phrasing to policy/academic vocabulary that
    matches the indexed content — without hardcoding any specific terms.
    Falls back to original query if Groq call fails.
    """
    if not domains:
        return query

    domain_hints = {
        DOMAIN_CALENDAR: "academic calendar, dates, deadlines, registration, exams, breaks",
        DOMAIN_CONTACTS: "university departments, offices, phone numbers, emails, locations",
        DOMAIN_DOCUMENTS: "university policies, student handbook, academic rules, conduct, procedures",
        DOMAIN_TUITION: "tuition rates, fees, costs, billing, financial charges",
    }

    relevant_hints = " | ".join(domain_hints[d] for d in domains if d in domain_hints)
    hint = (context_hint or "").strip()

    system = (
        "You are a query rewriting assistant for a university information system. "
        "Rewrite the user's question using clear academic/administrative vocabulary "
        "that would match university policy documents and databases. "
        "Keep it concise — one sentence. Return only the rewritten query, nothing else. "
        "Preserve every season and year the user names (e.g. Fall 2026); never drop or substitute them. "
        "For academic-calendar questions about course registration or enrollment windows: calendar rows are "
        "titled like \"Fall Registration Begins\" for the term the student is registering for—keep that season "
        "in the rewrite (e.g. Fall 2026 registration, Fall registration begins). Do not replace it with the "
        "semester when the registration window actually opens."
    )
    if hint:
        system += (
            " If a previous user question is given, merge any entities it mentions "
            "(school, degree level, dates, offices, policies) into the rewritten query "
            "when the current query is a short follow-up."
        )

    user_content = (
        (
            f"Domains: {relevant_hints}\n"
            f"Previous question (for context only): {hint}\n"
            f"Current query: {query}\n\n"
            f"Rewritten query:"
        )
        if hint
        else (
            f"Domains: {relevant_hints}\n"
            f"Original query: {query}\n\n"
            f"Rewritten query:"
        )
    )

    result = _groq_call(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        max_tokens=60,
    )
    return result if result else query

def get_answer_for_domain(query: str, domain: str, chat_history: list = None) -> tuple[str, list, dict, bool, str, str, list]:
    """
    Bypass the router and search a specific domain directly.
    Used for clarification follow-ups where we already know the domain.
    Returns same 7-tuple as get_answer.
    """
    system_prompt = """You are Hawk, Illinois Tech's official university assistant. You help students, faculty, and staff with questions about the academic calendar, tuition and fees, staff contacts, and university policies.

Answer only from the context blocks provided. Never use your own training knowledge — even for questions that seem general or factual. If the information is not in the context, say: "I don't have that information available." Do not guess or extrapolate.

Cite sources naturally: "According to the academic calendar…", "The student handbook states…", "Based on the tuition schedule…" Never mention technical terms like "chunks", "retrieval", or "context blocks".

State exact figures from context — dollar amounts, credit hours, GPA thresholds, dates. Preserve any conditions or ranges exactly as stated (e.g. "admitted before Fall 2024").

When context covers multiple student levels (undergraduate, graduate, co-terminal) and the user has not specified theirs, list the rate for each level explicitly — do not pick just one. Co-terminal students are undergraduate students for tuition and registration status purposes unless the source document explicitly states otherwise.

Use conversation history only to resolve pronouns and short references in the current question ("when do they end?", "what about part time?", "who handles that?"). Do not let earlier turns bleed into the current answer.

Be direct and concise. Lead with the answer, then supporting detail. Do not open with filler phrases ("Great question", "Certainly", "Of course", "I'm happy to help"). When multiple relevant pieces exist, provide both.
"""

    q = (query or "").strip()
    if not q:
        return "Please enter a question.", [], {}, False, "", domain, []

    route_details = {"domains": [domain], "needs_clarification": False, "sub_queries": {domain: q}}

    context_parts = []
    all_sources = []
    total_hits = 0
    clarification_messages = []

    if domain == DOMAIN_CALENDAR:
        try:
            hits = calendar_route_query(q)
            if isinstance(hits, dict) and hits.get("needs_clarification"):
                clarification_messages.append((hits["message"], hits.get("options", [])))
                hits = []
            if hits:
                total_hits += len(hits)
                context_parts.extend(build_calendar_context(hits))
                all_sources.extend(build_sources(hits))
        except Exception as e:
            st.error(f"Calendar search error: {e}")

    elif domain == DOMAIN_CONTACTS:
        try:
            hits = contacts_rrf_search(q, top_k=10)
            if isinstance(hits, dict) and hits.get("needs_clarification"):
                clarification_messages.append((hits["message"], hits.get("options", [])))
                hits = []
            if hits:
                total_hits += len(hits)
                context_parts.extend(build_contacts_context(hits))
                all_sources.extend(build_sources(hits))
        except Exception as e:
            st.error(f"Contacts search error: {e}")

    elif domain == DOMAIN_DOCUMENTS:
        try:
            hits = documents_rrf_search(q)
            if isinstance(hits, dict) and hits.get("needs_clarification"):
                clarification_messages.append((hits["message"], hits.get("options", [])))
                hits = []
            if hits:
                total_hits += len(hits)
                context_parts.extend(build_documents_context(hits))
                all_sources.extend(build_sources(hits))
        except Exception as e:
            st.error(f"Documents search error: {e}")

    elif domain == DOMAIN_TUITION:
        try:
            hits = tuition_rrf_search(q, top_k=10)
            if isinstance(hits, dict) and hits.get("needs_clarification"):
                clarification_messages.append((hits["message"], hits.get("options", [])))
                hits = []
            if hits:
                total_hits += len(hits)
                context_parts.extend(build_tuition_context(hits))
                all_sources.extend(build_sources(hits))
        except Exception as e:
            st.error(f"Tuition search error: {e}")

    if total_hits == 0:
        if clarification_messages:
            raw_msg, options = clarification_messages[0]
            display_msg = _format_clarification(raw_msg, options)
            return display_msg, [], route_details, True, raw_msg, domain, options
        return "I couldn't find anything matching that. Could you try rephrasing?", [], route_details, False, "", "", []

    context_text = "\n\n---\n\n".join(context_parts)
    unique_sources = list(dict.fromkeys(all_sources))

    def _build_messages(system, history, question, context):
        messages = [{"role": "system", "content": system}]
        if history:
            last_turns = [
                t for t in history
                if t.get("role") in ("user", "assistant")
                and (t.get("content") or "").strip()
            ][-2:]
            for turn in last_turns:
                messages.append({"role": turn["role"], "content": turn["content"].strip()})
        messages.append({"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"})
        return messages

    messages = _build_messages(system_prompt, chat_history or [], q, context_text)

    # ── Primary: OpenAI gpt-4o-mini ──────────────────────────────────────────
    answer = _openai_synthesis(messages)

    # ── Fallback: Groq ────────────────────────────────────────────────────────
    if not answer and GROQ_API_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.0,
            )
            raw_content = getattr(resp.choices[0].message, "content", None)
            answer = (raw_content or "").strip()
            if not answer:
                raise ValueError("empty Groq completion")
        except Exception as e:
            _log.warning("Groq completion failed (domain follow-up): %s", e)

    if not answer:
        return f"**Context found:**\n\n{context_text}", unique_sources, route_details, False, "", "", []

    return answer, unique_sources, route_details, False, "", "", []

def get_answer(query: str, chat_history: list = None) -> tuple[str, list, dict, bool, str, str, list]:
    """
    Returns: (reply, sources, route_details, is_clarification, clarification_message, clarifying_domain, clarification_options)
    clarifying_domain: the domain that triggered clarification (empty string if none)
    """
    system_prompt = """You are Hawk, Illinois Tech's official university assistant. You help students, faculty, and staff with questions about the academic calendar, tuition and fees, staff contacts, and university policies.

Answer only from the context blocks provided. Never use your own training knowledge — even for questions that seem general or factual. If the information is not in the context, say: "I don't have that information available." Do not guess or extrapolate.

Cite sources naturally: "According to the academic calendar…", "The student handbook states…", "Based on the tuition schedule…" Never mention technical terms like "chunks", "retrieval", or "context blocks".

State exact figures from context — dollar amounts, credit hours, GPA thresholds, dates. Preserve any conditions or ranges exactly as stated (e.g. "admitted before Fall 2024").

When context covers multiple student levels (undergraduate, graduate, co-terminal) and the user has not specified theirs, list the rate for each level explicitly — do not pick just one. Co-terminal students are undergraduate students for tuition and registration status purposes unless the source document explicitly states otherwise.

Use conversation history only to resolve pronouns and short references in the current question ("when do they end?", "what about part time?", "who handles that?"). Do not let earlier turns bleed into the current answer.

Be direct and concise. Lead with the answer, then supporting detail. Do not open with filler phrases ("Great question", "Certainly", "Of course", "I'm happy to help"). When multiple relevant pieces exist, provide both.
"""

    q = (query or "").strip()
    if not q:
        return "Please enter a question.", [], {}, False, "", "", []

    small = _off_topic_short_reply(q)
    if small:
        return small, [], {}, False, "", "", []

    # Step 1: Route
    route_details = get_routing_intent(q)
    domains = route_details.get("domains", [])
    needs_clarification = route_details.get("needs_clarification", False)
    sub_queries = route_details.get("sub_queries", {})

    # Step 2: Router-level clarification (low confidence / out of scope)
    if needs_clarification or not domains:
        msg = "I can help with the university calendar, staff contacts, tuition and fees, and academic policies. Could you rephrase with more detail?"
        return msg, [], route_details, False, "", "", []

    # Step 2.5: Rewrite query into retrieval-ready language
    # prev_user = second-to-last user message (not current) for follow-up context
    prev_user = _previous_user_utterance(chat_history)
    retrieval_query = rewrite_query(q, domains, context_hint=prev_user)
    # Only treat as follow-up when the previous turn was a real answer (not clarification/error)
    # Prevents is_followup from firing after unrelated answered questions in long sessions
    last_was_answer = st.session_state.get("last_turn_was_answer", False)
    is_followup = last_was_answer and is_followup_query(q, prev_user)
    retrieval_sub_queries = {domain: retrieval_query for domain in domains}

    context_parts = []
    all_sources = []
    total_hits = 0
    clarification_messages = []  # list of (domain, message, options) tuples
    domain_hit_counts = {domain: 0 for domain in domains}
    primary_domain = domains[0] if domains else ""

    # Step 3: Retrieve
    if DOMAIN_CALENDAR in domains:
        cal_query = q
        try:
            hits = calendar_route_query(cal_query)
            if isinstance(hits, dict) and hits.get("needs_clarification"):
                if is_followup:
                    # Follow-up: bypass router clarification, search directly
                    hits = calendar_rrf_search(retrieval_query)
                else:
                    clarification_messages.append((DOMAIN_CALENDAR, hits["message"], hits.get("options", [])))
                    hits = []
            if hits and isinstance(hits, list):
                total_hits += len(hits)
                domain_hit_counts[DOMAIN_CALENDAR] += len(hits)
                context_parts.extend(build_calendar_context(hits))
                all_sources.extend(build_sources(hits))
        except Exception as e:
            st.error(f"Calendar search error: {e}")

    if DOMAIN_CONTACTS in domains:
        con_query = retrieval_sub_queries.get(DOMAIN_CONTACTS, retrieval_query)
        try:
            hits = contacts_rrf_search(con_query, top_k=10)
            if isinstance(hits, dict) and hits.get("needs_clarification"):
                if not is_followup:
                    clarification_messages.append((DOMAIN_CONTACTS, hits["message"], hits.get("options", [])))
                hits = []
            if hits and isinstance(hits, list):
                total_hits += len(hits)
                domain_hit_counts[DOMAIN_CONTACTS] += len(hits)
                context_parts.extend(build_contacts_context(hits))
                all_sources.extend(build_sources(hits))
        except Exception as e:
            st.error(f"Contacts search error: {e}")

    if DOMAIN_DOCUMENTS in domains:
        doc_query = retrieval_sub_queries.get(DOMAIN_DOCUMENTS, retrieval_query)
        try:
            hits = documents_rrf_search(doc_query)
            if isinstance(hits, dict) and hits.get("needs_clarification"):
                clarification_messages.append((DOMAIN_DOCUMENTS, hits["message"], hits.get("options", [])))
                hits = []
            if hits:
                total_hits += len(hits)
                domain_hit_counts[DOMAIN_DOCUMENTS] += len(hits)
                context_parts.extend(build_documents_context(hits))
                all_sources.extend(build_sources(hits))
        except Exception as e:
            st.error(f"Documents search error: {e}")

    if DOMAIN_TUITION in domains:
        tuition_query = retrieval_sub_queries.get(DOMAIN_TUITION, retrieval_query)
        try:
            hits = tuition_rrf_search(tuition_query, top_k=10)
            if isinstance(hits, dict) and hits.get("needs_clarification"):
                if not is_followup:
                    clarification_messages.append((DOMAIN_TUITION, hits["message"], hits.get("options", [])))
                hits = []
            if hits and isinstance(hits, list):
                total_hits += len(hits)
                domain_hit_counts[DOMAIN_TUITION] += len(hits)
                context_parts.extend(build_tuition_context(hits))
                all_sources.extend(build_sources(hits))
        except Exception as e:
            st.error(f"Tuition search error: {e}")

    # If the primary domain answered, don't let weaker secondary domains append
    # unrelated clarification prompts onto an otherwise grounded answer.
    if (
        primary_domain
        and domain_hit_counts.get(primary_domain, 0) > 0
        and clarification_messages
        and not _looks_multi_part_query(q)
    ):
        # Primary domain answered — suppress ALL secondary clarifications entirely
        clarification_messages = []

    # Step 4: Post-retrieval clarification
    # Show clarification if:
    # (a) nothing was retrieved at all, OR
    # (b) the PRIMARY routed domain needs clarification — even if secondary
    #     domains returned hits, we must not silently answer the wrong school/
    #     semester when the user's actual intent is underspecified.
    if clarification_messages:
        primary_domain = domains[0] if domains else None
        primary_clarifications = [
            (domain, msg, opts)
            for domain, msg, opts in clarification_messages
            if domain == primary_domain
        ]
        should_clarify = total_hits == 0 or bool(primary_clarifications)
        if should_clarify:
            clarifying_domain, raw_msg, options = (
                primary_clarifications[0]
                if primary_clarifications
                else clarification_messages[0]
            )
            display_msg = _format_clarification(raw_msg, options)
            return display_msg, [], route_details, True, raw_msg, clarifying_domain, options

    if total_hits == 0:
        msg = (
            "I don't have that information available."
            if is_followup
            else "I don't have information matching your question. Could you try rephrasing or adding more detail?"
        )
        return msg, [], route_details, False, "", "", []

    # Step 5: Synthesize
    context_text = "\n\n---\n\n".join(context_parts)
    unique_sources = list(dict.fromkeys(all_sources))

    def _build_messages(system: str, history: list, question: str, context: str) -> list:
        messages = [{"role": "system", "content": system}]
        if history:
            last_turns = [
                t for t in history
                if t.get("role") in ("user", "assistant")
                and (t.get("content") or "").strip()
            ][-2:]  # last 1 turn only — prevents contamination from earlier unrelated turns
            for turn in last_turns:
                messages.append({"role": turn["role"], "content": turn["content"].strip()})
        messages.append({
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}"
        })
        return messages

    history = chat_history or []
    messages = _build_messages(system_prompt, history, q, context_text)

    # ── Primary: OpenAI gpt-4o-mini ──────────────────────────────────────────
    answer = _openai_synthesis(messages)

    # ── Fallback: Groq ────────────────────────────────────────────────────────
    if not answer and GROQ_API_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.0,
            )
            raw_content = getattr(resp.choices[0].message, "content", None)
            answer = (raw_content or "").strip()
            if not answer:
                raise ValueError("empty Groq completion")
        except Exception as e:
            _log.warning("Groq completion failed: %s", e)

    # ── No LLM available: return raw context ─────────────────────────────────
    if not answer:
        answer = f"**Context found:**\n\n{context_text}"

    if clarification_messages:
        clarifying_domain, raw_msg, options = clarification_messages[0]
        reply = _append_partial_clarification(answer, raw_msg, options)
        return reply, unique_sources, route_details, True, raw_msg, clarifying_domain, options
    return answer, unique_sources, route_details, False, "", "", []

def stream_generator(text):
    import time
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="IIT University Chatbot")
st.title("University Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you navigate the university today?"}
    ]
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "pending_clarification_msg" not in st.session_state:
    st.session_state.pending_clarification_msg = None
if "pending_domain" not in st.session_state:
    st.session_state.pending_domain = None  # domain that triggered clarification
if "pending_clarification_options" not in st.session_state:
    st.session_state.pending_clarification_options = []
if "last_turn_was_answer" not in st.session_state:
    st.session_state.last_turn_was_answer = False  # True only when last turn was a real answer

# Display chat history
for msg in st.session_state.messages:
    avatar = assistant_avatar if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s}")
        if "router" in msg and msg["router"]:
            with st.expander("Router (testing)"):
                st.json(msg["router"])

if prompt := st.chat_input("E.g., When is spring break?"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    effective_query = prompt.strip()
    pending_action = None
    with st.chat_message("assistant", avatar=assistant_avatar):
        with st.spinner("Searching databases..."):

            pending = st.session_state.get("pending_query")
            pending_msg = st.session_state.get("pending_clarification_msg")
            pending_domain = st.session_state.get("pending_domain")

            if pending and pending_msg:
                pending_opts = st.session_state.get("pending_clarification_options", [])
                pending_action = classify_pending_response(pending, pending_msg, prompt, pending_opts)

                if pending_action == "CANCEL":
                    st.session_state.pending_query = None
                    st.session_state.pending_clarification_msg = None
                    st.session_state.pending_domain = None
                    st.session_state.pending_clarification_options = []
                    reply = "No problem. I cleared that question. Ask me something else whenever you're ready."
                    sources = []
                    route_details = {}
                    is_clarification = False
                    clarification_msg = ""
                    clarifying_domain = ""
                    clarification_options = []
                elif pending_action == "NEW_TOPIC":
                    effective_query = prompt.strip()
                    # User changed topic — fresh query through normal pipeline
                    st.session_state.pending_query = None
                    st.session_state.pending_clarification_msg = None
                    st.session_state.pending_domain = None
                    st.session_state.pending_clarification_options = []
                    reply, sources, route_details, is_clarification, clarification_msg, clarifying_domain, clarification_options = get_answer(
                        effective_query,
                        chat_history=st.session_state.messages,
                    )
                else:
                    # User is answering — combine original + clarification into one query
                    effective_query = (
                        reformulate_query(pending, prompt.strip())
                        if GROQ_API_KEY
                        else f"{pending} {prompt}".strip()
                    )

                    if pending_domain:
                        # Skip router — go directly to the domain that asked
                        st.session_state.pending_query = None
                        st.session_state.pending_clarification_msg = None
                        st.session_state.pending_domain = None
                        st.session_state.pending_clarification_options = []
                        reply, sources, route_details, is_clarification, clarification_msg, clarifying_domain, clarification_options = get_answer_for_domain(
                            effective_query,
                            pending_domain,
                            chat_history=[],
                        )
                    else:
                        st.session_state.pending_query = None
                        st.session_state.pending_clarification_msg = None
                        st.session_state.pending_domain = None
                        st.session_state.pending_clarification_options = []
                        reply, sources, route_details, is_clarification, clarification_msg, clarifying_domain, clarification_options = get_answer(
                            effective_query,
                            chat_history=st.session_state.messages,
                        )
            else:
                reply, sources, route_details, is_clarification, clarification_msg, clarifying_domain, clarification_options = get_answer(
                    effective_query,
                    chat_history=st.session_state.messages,
                )

        if is_clarification:
            st.session_state.pending_query = effective_query
            st.session_state.pending_clarification_msg = clarification_msg
            st.session_state.pending_domain = clarifying_domain or None
            st.session_state.pending_clarification_options = clarification_options or []
            st.session_state.last_turn_was_answer = False  # clarification, not an answer
        else:
            st.session_state.pending_query = None
            st.session_state.pending_clarification_msg = None
            st.session_state.pending_domain = None
            st.session_state.pending_clarification_options = []
            if pending_action == "CANCEL":
                st.session_state.last_turn_was_answer = False
            else:
                # Only mark as answered if we got real content (not "I'm not sure" / error)
                not_sure = "i'm not sure" in reply.lower() or "couldn't find" in reply.lower()
                st.session_state.last_turn_was_answer = not not_sure

        st.write_stream(stream_generator(reply))

        if sources:
            with st.expander("View Sources"):
                for s in sources:
                    st.write(f"- {s}")
        if route_details:
            with st.expander("Router (testing)"):
                st.json(route_details)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "sources": sources,
        "router": route_details,
    })