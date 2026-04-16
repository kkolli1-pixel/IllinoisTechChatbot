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

# ── Small LLM helper (Groq primary, GPT fallback) ──────────────────────────────

def _groq_call(messages: list, max_tokens: int = 60) -> str:
    """Groq call for intent/reformulation tasks. Falls back to GPT if Groq fails."""
    if GROQ_API_KEY:
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
            _log.warning("Groq call failed — falling back to GPT.")
    # GPT fallback
    if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY and AZURE_OPENAI_DEPLOYMENT:
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
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            _log.warning("GPT fallback also failed: %s", e)
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
                    "You are an intent classifier for a university chatbot. "
                    "Output exactly one token: YES or NO."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The chatbot asked the user: \"{clarification_message}\"\n"
                    f"The user replied: \"{user_response}\"\n\n"
                    f"Does the reply supply actionable information toward answering the "
                    f"chatbot's question? Actionable information includes: semester names, "
                    f"years, school names, department names, program names, fee types, "
                    f"student levels (graduate, undergraduate), proper nouns, or any "
                    f"direct selection from the offered choices.\n\n"
                    f"Short or single-word replies that match a valid category are YES.\n\n"
                    f"Output: YES or NO."
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

    # Heuristic: if the reply starts with a topic-switch signal word and contains
    # a wh-question or a policy/topic keyword, treat as NEW_TOPIC without LLM.
    _SWITCH_SIGNALS = re.compile(
        r"^\s*(actually|instead|wait|hold on|forget that|different question|"
        r"what about|never mind|on second thought)\b",
        re.IGNORECASE,
    )
    _WH_QUESTION = re.compile(r"\b(what|when|where|who|how|why|is there|are there|can I|do I)\b", re.IGNORECASE)
    if _SWITCH_SIGNALS.match(response) and _WH_QUESTION.search(response):
        return "NEW_TOPIC"

    # Heuristic: a complete question (ends with ?) that shares no words with
    # the clarification options is almost certainly a topic switch.
    if response.rstrip().endswith("?") and clarification_options:
        reply_words = set(re.findall(r"\b[a-z0-9]+\b", response.lower()))
        opt_words = set(
            w
            for opt in clarification_options
            for w in re.findall(r"\b[a-z0-9]+\b", (opt or "").lower())
            if len(w) > 2
        )
        # If fewer than 2 option words appear in the reply, it's a new topic
        overlap = reply_words & opt_words
        if len(overlap) < 2:
            return "NEW_TOPIC"

    if not GROQ_API_KEY:
        return "NEW_TOPIC" if is_escape(clarification_message, response) else "ANSWER"

    result = _groq_call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a dialogue-act classifier for a multi-turn university chatbot. "
                    "Output exactly one label: ANSWER, NEW_TOPIC, or CANCEL."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"<turn_context>\n"
                    f"Original question: \"{original_query}\"\n"
                    f"Chatbot clarification: \"{clarification_message}\"\n"
                    f"User reply: \"{user_response}\"\n"
                    f"</turn_context>\n\n"
                    "<label_definitions>\n"
                    "ANSWER — The reply provides information responsive to the clarification, "
                    "even if incomplete or requiring further refinement. This includes: school "
                    "names, department names, student levels, semesters, years, fee types, "
                    "partial answers, or references to the same concept the chatbot asked about.\n"
                    "NEW_TOPIC — The reply is an unrelated question on a DIFFERENT subject "
                    "that is NOT responsive to the clarification. Key signals: the reply asks "
                    "a full question about a different topic, begins with 'actually', 'instead', "
                    "'what about', or introduces a completely new subject (e.g., asking about "
                    "plagiarism when the clarification was about finals dates). If the reply "
                    "contains a question mark and asks about something unrelated to the "
                    "clarification, it is NEW_TOPIC.\n"
                    "CANCEL — The reply explicitly abandons the conversation "
                    "(e.g., \"never mind,\" \"forget it,\" \"stop\").\n"
                    "</label_definitions>\n\n"
                    "Output one label:"
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
                    "You are a query reformulation module in a retrieval pipeline. "
                    "Merge the original question and the user's clarification into one "
                    "self-contained question suitable for document retrieval. "
                    "Rules: (1) Output only the reformulated question — no commentary. "
                    "(2) Insert the clarification value verbatim — do not paraphrase, "
                    "translate, or reinterpret slot values such as school names, "
                    "departments, levels, or fee types. "
                    "(3) Preserve the original question's intent and scope."
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
                    "You are a co-reference detector for a multi-turn dialogue system. "
                    "Output exactly one token: YES or NO."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Turn N-1: \"{prev_user}\"\n"
                    f"Turn N:   \"{query}\"\n\n"
                    f"Does Turn N depend on Turn N-1 to be understood? "
                    f"Indicators: pronouns referring to entities in N-1, ellipsis, "
                    f"or implicit continuation of the same topic. "
                    f"A question that is fully self-contained is NO.\n\n"
                    f"Output: YES or NO."
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
        "hey what's up",
        "hey whats up",
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
        "goodbye",
        "bye",
        "bye!",
        "goodbye!",
        "see you",
        "see ya",
        "good bye",
        "hello?",
        "hey there",
        "heyy",
        "heyyy",
        "can you help me",
        "can you help me?",
        "can you help",
        "help me",
        "help",
        "help!",
        "help me please",
        "please help",
        "please help me",
        "i need help",
        "i need some help",
        "i need assistance",
        "idk where to start",
        "i dont know where to start",
        "i have no idea what im doing",
        "i have no idea",
        "i dont know what to do",
        "i dont know",
        "i'm lost",
        "im lost",
        "this is so confusing",
        "this is confusing",
        "so confused",
        "i'm confused",
        "im confused",
        "nobody told me about any of this",
        "i dont know who to ask",
        "i don't know who to ask",
        "wait how does this work",
        "how does this work",
        "how does this work?",
        "umm",
        "um",
        "uh",
        "i have a question",
        "i have a question!",
        "i need info",
        "i need information",
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
    # Self-identity questions
    if s in {"who are you", "who are you?", "what are you", "what are you?",
             "what is this", "what is this?", "what can you do", "what can you do?",
             "what do you do", "what do you do?",
             "what even is this chatbot", "what is this chatbot",
             "what does this chatbot do", "what does this do",
             "what does this do?",
             "what even are you", "what even are you?"}:
        return (
            "I am Hawk, the Illinois Institute of Technology virtual assistant. "
            "I can help with the academic calendar, staff contacts, tuition and fees, "
            "and university policies. What would you like to know?"
        )
    # Emotional / wellbeing — redirect compassionately to counseling
    _EMOTIONAL = re.compile(
        r"\b(cant focus|can't focus|overwhelmed|too much right now|don't want to be here"
        r"|dont want to be here|everything is too much|stressed out|breaking down"
        r"|falling apart|cant handle|can't handle|need someone to talk to"
        r"|nobody cares|no one cares|give up|giving up|want to quit"
        r"|mental health|anxiety|depression)\b",
        re.IGNORECASE,
    )
    if _EMOTIONAL.search(query):
        return (
            "It sounds like you are going through a tough time. "
            "Illinois Tech's Counseling Services are here to help — "
            "you can reach them at counseling@iit.edu or 312-567-5395. "
            "The Office of Student Affairs is also available at dos@illinoistech.edu or 312-567-3081."
        )
    # Homework / essay / personal task requests — only unambiguous academic dishonesty phrases
    _TASK_REQUEST = re.compile(
        r"\b(write|do|complete)\b.{0,30}\b(essay|homework|assignment)\b",
        re.IGNORECASE,
    )
    if _TASK_REQUEST.search(query):
        return (
            "I am not able to help with assignments or personal tasks. "
            "I can assist with Illinois Tech calendar, contacts, tuition, and policy questions."
        )
    # Clearly off-topic: math, translation, trivia, general knowledge
    _OOD_HARD = re.compile(
        r"\b(what is \d+[\s\+\-\*\/]+(plus|minus|times|divided)?\s*\d+|"
        r"what is \d+\s*(plus|minus|times|divided by)\s*\d+|"
        r"calculate|solve for|translate (this|to|into)|"
        r"weather (today|tomorrow|forecast)|who (won|is winning)|super bowl|world cup|"
        r"stock price|bitcoin|recipe for|how to cook|recommend (a |me a )?(restaurant|movie|song|book)|"
        r"capital of [a-z]+|president of (the )?(us|usa|united states|[a-z]+\s[a-z]+))\b",
        re.IGNORECASE,
    )
    if _OOD_HARD.search(query):
        return (
            "I can help with the university calendar, staff contacts, tuition and fees, "
            "and academic policies. Could you rephrase with more detail?"
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
        "You are a query rewriter in a university RAG pipeline. "
        "Transform the user's natural-language question into a single retrieval-optimized "
        "sentence using formal academic and administrative vocabulary. "
        "Rules: (1) Output only the rewritten query — no commentary or explanation. "
        "(2) Preserve all temporal references exactly as stated (e.g., Fall 2026, Spring 2025). "
        "Never substitute, drop, or infer a different term or year. "
        "(3) For registration-related queries, keep the target term in the rewrite "
        "(e.g., 'Fall 2026 registration begins'), not the term when the registration window opens. "
        "(4) Keep it to one concise sentence."
    )
    if hint:
        system += (
            " (5) A previous user question is provided for co-reference resolution. "
            "If the current query is a short follow-up, merge relevant entities from the "
            "previous question (school, degree level, dates, offices, policies) into the rewrite."
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
    system_prompt = """You are Hawk, the official Illinois Institute of Technology virtual assistant.

<role>
You serve students, faculty, and staff with authoritative answers about four domains: academic calendar, tuition & fees, staff directory, and university policies/handbook.
</role>

<constraints>
- Ground every claim in the provided context blocks. If the answer is not present in context, respond exactly: "I don't have that information available."
- Never use your own training knowledge, even for facts that seem common or obvious. Do not guess, infer, or extrapolate beyond what the context states.
- Never reveal implementation details. Do not mention "chunks," "retrieval," "context blocks," "search results," or any internal system terminology.
</constraints>

<formatting>
- Lead with the direct answer. Follow with supporting detail only when it adds value.
- State exact figures verbatim: dollar amounts, credit hours, GPA thresholds, dates, deadlines. Preserve any conditions or ranges exactly as written in context (e.g., "admitted before Fall 2024").
- Cite sources conversationally: "According to the academic calendar…," "The tuition schedule shows…," "The student handbook states…"
- Use bullet points and bold labels when listing multiple items. Do not use markdown headers (##, ###). Keep paragraphs short.
- Do not open with filler ("Great question!", "Certainly!", "Of course!", "I'm happy to help!"). Do not close with pleasantries ("Let me know if you need anything else!", "Let me know if you need further clarification!", "Feel free to ask!", or any variation). End with the last piece of factual content.
</formatting>

<multi_level_tuition>
When context contains tuition data for multiple student levels (undergraduate, graduate, co-terminal) and the user has not specified a level, list the rate for every level present — never select just one. Co-terminal students are classified as undergraduate for tuition and registration purposes unless the source explicitly states otherwise.
</multi_level_tuition>

<conversation_history>
Use prior turns only to resolve pronouns and short references in the current question ("when do they end?", "what about part time?", "who handles that?"). Never let information from earlier turns bleed into the current answer. Each response must be independently accurate given the current context.
</conversation_history>
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
    system_prompt = """You are Hawk, the official Illinois Institute of Technology virtual assistant.

<role>
You serve students, faculty, and staff with authoritative answers about four domains: academic calendar, tuition & fees, staff directory, and university policies/handbook.
</role>

<constraints>
- Ground every claim in the provided context blocks. If the answer is not present in context, respond exactly: "I don't have that information available."
- Never use your own training knowledge, even for facts that seem common or obvious. Do not guess, infer, or extrapolate beyond what the context states.
- Never reveal implementation details. Do not mention "chunks," "retrieval," "context blocks," "search results," or any internal system terminology.
</constraints>

<formatting>
- Lead with the direct answer. Follow with supporting detail only when it adds value.
- State exact figures verbatim: dollar amounts, credit hours, GPA thresholds, dates, deadlines. Preserve any conditions or ranges exactly as written in context (e.g., "admitted before Fall 2024").
- Cite sources conversationally: "According to the academic calendar…," "The tuition schedule shows…," "The student handbook states…"
- Use bullet points and bold labels when listing multiple items. Do not use markdown headers (##, ###). Keep paragraphs short.
- Do not open with filler ("Great question!", "Certainly!", "Of course!", "I'm happy to help!"). Do not close with pleasantries ("Let me know if you need anything else!", "Let me know if you need further clarification!", "Feel free to ask!", or any variation). End with the last piece of factual content.
</formatting>

<multi_level_tuition>
When context contains tuition data for multiple student levels (undergraduate, graduate, co-terminal) and the user has not specified a level, list the rate for every level present — never select just one. Co-terminal students are classified as undergraduate for tuition and registration purposes unless the source explicitly states otherwise.
</multi_level_tuition>

<conversation_history>
Use prior turns only to resolve pronouns and short references in the current question ("when do they end?", "what about part time?", "who handles that?"). Never let information from earlier turns bleed into the current answer. Each response must be independently accurate given the current context.
</conversation_history>
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
    # prev_user = second-to-last user message (not current) for follow-up context.
    # In the Streamlit path, the current question IS already appended to chat_history
    # before get_answer is called, so users[-2] is the previous turn.
    # In the API path, the current question is NOT in chat_history, so users[-1]
    # is already the previous turn — _previous_user_utterance would return "" (only 1 user).
    # Detect which path we're in and handle accordingly.
    prev_user = _previous_user_utterance(chat_history)
    if not prev_user and chat_history:
        # API path: fall back to the last user message in history
        api_users = [
            (t.get("content") or "").strip()
            for t in chat_history
            if t.get("role") == "user" and (t.get("content") or "").strip()
        ]
        if api_users:
            prev_user = api_users[-1]
    retrieval_query = rewrite_query(q, domains, context_hint=prev_user)
    # Only treat as follow-up when the previous turn was a real answer (not clarification/error)
    # Prevents is_followup from firing after unrelated answered questions in long sessions
    last_was_answer = st.session_state.get("last_turn_was_answer", False)
    # API path: infer from chat_history when session_state is unavailable
    if not last_was_answer and chat_history and len(chat_history) >= 2:
        last_asst = chat_history[-1].get("content", "") if chat_history[-1].get("role") == "assistant" else ""
        # If the last assistant message looks like a real answer (not a clarification prompt),
        # treat this as a follow-up candidate
        if last_asst and not any(kw in last_asst.lower() for kw in [
            "could you specify", "which semester", "which school",
            "could you clarify", "options:",
        ]):
            last_was_answer = True
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
        # Use the rewritten query for routing — it includes semester context
        # resolved from the previous turn (e.g. "when do they end?" → "when do
        # spring 2026 classes end?"), avoiding spurious clarification prompts.
        cal_query = retrieval_query if retrieval_query else q
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
            # Also search with original query when rewrite differs — the rewriter
            # can add noise that derails retrieval, so merge both result sets.
            if doc_query.lower() != q.lower():
                orig_hits = documents_rrf_search(q)
                if isinstance(orig_hits, list) and orig_hits:
                    seen_ids = {h["_id"] for h in (hits or [])}
                    for oh in orig_hits:
                        if oh["_id"] not in seen_ids:
                            hits = hits or []
                            hits.append(oh)
                            seen_ids.add(oh["_id"])
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