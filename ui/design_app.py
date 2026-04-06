from __future__ import annotations

import csv
import json
import os
import re
import sys
from datetime import date
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

from ui_utils.chat_panel import render_chat_panel
from ui_utils.layout import load_css, render_footer_actions, render_title_block
from ui.app_with_clarification_memory import (
    GROQ_API_KEY,
    classify_pending_response,
    get_answer,
    get_answer_for_domain,
    reformulate_query,
)

TOPIC_OPTIONS = [
    "Academic Calendar",
    "Tuition",
    "Directory",
    "Policies",
    "Handbook",
]

TOPIC_ICONS = {
    "Academic Calendar": "📅",
    "Tuition": "💲",
    "Directory": "📇",
    "Policies": "📜",
    "Handbook": "📘",
}


def _init_state() -> None:
    if "history_a" not in st.session_state:
        st.session_state.history_a = [
            {"role": "assistant", "content": "Welcome! Share a prompt and I will respond clearly in this panel."}
        ]
    if "history_b" not in st.session_state:
        st.session_state.history_b = [
            {"role": "assistant", "content": "Welcome! Share a prompt and I will respond clearly in this panel."}
        ]
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""
    if "selected_topic" not in st.session_state:
        st.session_state.selected_topic = None
    if "compare_result" not in st.session_state:
        st.session_state.compare_result = ""
    if "show_feedback_form" not in st.session_state:
        st.session_state.show_feedback_form = False
    if "feedback_status" not in st.session_state:
        st.session_state.feedback_status = ""
    if "last_turn_was_answer" not in st.session_state:
        st.session_state.last_turn_was_answer = False
    if "pending_query_b" not in st.session_state:
        st.session_state.pending_query_b = None
    if "pending_clarification_msg_b" not in st.session_state:
        st.session_state.pending_clarification_msg_b = None
    if "pending_domain_b" not in st.session_state:
        st.session_state.pending_domain_b = None
    if "session_id_a" not in st.session_state:
        st.session_state.session_id_a = None


def _timestamp_now() -> str:
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def _topic_list_text() -> str:
    return ", ".join(TOPIC_OPTIONS[:-1]) + f", and {TOPIC_OPTIONS[-1]}"


def _append_message(history_key: str, role: str, content: str, metadata: dict | None = None) -> None:
    st.session_state[history_key].append(
        {"role": role, "content": content, "timestamp": _timestamp_now(), "metadata": metadata}
    )


def _last_assistant_content(history: list[dict[str, Any]]) -> str:
    for message in reversed(history):
        if str(message.get("role", "")) == "assistant":
            return str(message.get("content", "")).strip()
    return ""


def _compare_chatbot_responses() -> str:
    response_a = _last_assistant_content(st.session_state.get("history_a", []))
    response_b = _last_assistant_content(st.session_state.get("history_b", []))

    if not response_a or not response_b:
        return "I cant compare yet. Please send a message first."

    words_a = set(re.findall(r"[a-zA-Z0-9]+", response_a.lower()))
    words_b = set(re.findall(r"[a-zA-Z0-9]+", response_b.lower()))
    overlap = len(words_a & words_b)
    overlap_pct = int((overlap / max(1, min(len(words_a), len(words_b)))) * 100)
    length_a = len(response_a)
    length_b = len(response_b)

    longer_text = "A" if length_a > length_b else ("B" if length_b > length_a else "Both are similar")
    return (
        f"Comparison summary:\n"
        f"- Response A length: {length_a} chars\n"
        f"- Response B length: {length_b} chars\n"
        f"- Vocabulary overlap: {overlap_pct}%\n"
        f"- More detailed response: {longer_text}"
    )


def _save_feedback(topic: str, rating: int, feedback_text: str) -> bool:
    feedback_dir = Path(__file__).parent / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    feedback_file = feedback_dir / "feedback_log.jsonl"
    payload = {
        "timestamp": datetime.now().isoformat(),
        "topic": topic,
        "rating": rating,
        "feedback": feedback_text.strip(),
        "latest_response_a": _last_assistant_content(st.session_state.get("history_a", [])),
        "latest_response_b": _last_assistant_content(st.session_state.get("history_b", [])),
    }
    try:
        with feedback_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return True
    except Exception:
        return False


def _handle_footer_actions(disabled: bool) -> None:
    compare_clicked, feedback_clicked = render_footer_actions(disabled=disabled)

    if compare_clicked:
        if st.session_state.selected_topic:
            st.session_state.compare_result = _compare_chatbot_responses()
        else:
            st.session_state.compare_result = "Please choose a topic and ask a question first."
    if feedback_clicked:
        st.session_state.show_feedback_form = not st.session_state.show_feedback_form

    if st.session_state.compare_result:
        st.info(st.session_state.compare_result)

    if st.session_state.show_feedback_form:
        with st.form("feedback_form"):
            st.markdown("### Share Feedback")
            rating = st.slider("Rating", min_value=1, max_value=5, value=4)
            feedback_text = st.text_area("Feedback", placeholder="Tell us what worked and what can improve.")
            submit_feedback = st.form_submit_button("Submit Feedback", use_container_width=True)
            if submit_feedback:
                saved = _save_feedback(st.session_state.selected_topic or "Unknown", rating, feedback_text)
                st.session_state.feedback_status = (
                    "Thanks! Your feedback was saved."
                    if saved
                    else "Could not save feedback right now. Please try again."
                )
                st.session_state.show_feedback_form = False
                st.rerun()

    if st.session_state.feedback_status:
        if "saved" in st.session_state.feedback_status.lower():
            st.success(st.session_state.feedback_status)
        else:
            st.warning(st.session_state.feedback_status)
        st.session_state.feedback_status = ""


def _topic_intro(topic: str) -> str:
    return f"Great choice. I am ready to help you with {topic} information."


def _reset_histories_for_topic(topic: str) -> None:
    intro = _topic_intro(topic)
    st.session_state.history_a = [{"role": "assistant", "content": intro, "timestamp": _timestamp_now()}]
    st.session_state.history_b = [{"role": "assistant", "content": intro, "timestamp": _timestamp_now()}]
    st.session_state.pending_prompt = ""
    st.session_state.is_generating = False
    st.session_state.compare_result = ""
    st.session_state.show_feedback_form = False
    st.session_state.feedback_status = ""
    st.session_state.pending_query_b = None
    st.session_state.pending_clarification_msg_b = None
    st.session_state.pending_domain_b = None


def _enqueue_user_prompt(user_prompt: str) -> None:
    clean_prompt = user_prompt.strip()
    if not clean_prompt:
        return
    _append_message("history_a", "user", clean_prompt)
    _append_message("history_b", "user", clean_prompt)
    st.session_state.pending_prompt = clean_prompt
    st.session_state.is_generating = True


def _render_topic_selection() -> None:
    st.markdown(
        """
        <div class="topic-prompt">
            <h2>What would you like help with today?</h2>
            <p>Choose a topic to get started, and I will guide you step by step.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top_row = st.columns(3, gap="small")
    for idx, topic in enumerate(TOPIC_OPTIONS[:3]):
        with top_row[idx]:
            label = f"{TOPIC_ICONS.get(topic, '')} {topic}".strip()
            if st.button(label, key=f"topic_{topic}", use_container_width=True):
                st.session_state.selected_topic = topic
                _reset_histories_for_topic(topic)
                st.rerun()

    spacer_left, mid_one, mid_two, spacer_right = st.columns([1, 1, 1, 1], gap="small")
    for col, topic in zip([mid_one, mid_two], TOPIC_OPTIONS[3:]):
        with col:
            label = f"{TOPIC_ICONS.get(topic, '')} {topic}".strip()
            if st.button(label, key=f"topic_{topic}", use_container_width=True):
                st.session_state.selected_topic = topic
                _reset_histories_for_topic(topic)
                st.rerun()


def _discover_files(name_tokens: tuple[str, ...], suffixes: tuple[str, ...]) -> list[Path]:
    root = Path(__file__).parent
    ignored_dirs = {".venv", ".cursor", "__pycache__", ".git", ".dist"}
    paths: list[Path] = []
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for filename in files:
            lower_name = filename.lower()
            if not lower_name.endswith(suffixes):
                continue
            if any(token in lower_name for token in name_tokens):
                paths.append(Path(current_root) / filename)
    paths.sort(key=lambda p: (len(str(p)), str(p)))
    return paths


def _load_json_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        if isinstance(payload, list):
            rows.extend([item for item in payload if isinstance(item, dict)])
        elif isinstance(payload, dict):
            for key in ("items", "data", "chunks", "rows"):
                nested = payload.get(key)
                if isinstance(nested, list):
                    rows.extend([item for item in nested if isinstance(item, dict)])
                    break
    return rows


@st.cache_data(show_spinner=False)
def _load_tuition_data() -> list[dict[str, Any]]:
    paths = _discover_files(("tuition",), (".json",))
    return _load_json_rows(paths)


@st.cache_data(show_spinner=False)
def _load_calendar_data() -> list[dict[str, Any]]:
    paths = _discover_files(("calendar",), (".json",))
    return _load_json_rows(paths)


@st.cache_data(show_spinner=False)
def _load_directory_data() -> list[dict[str, str]]:
    paths = _discover_files(("contact", "directory"), (".csv",))
    rows: list[dict[str, str]] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                rows.extend(list(csv.DictReader(f)))
        except Exception:
            continue
    return rows


@st.cache_data(show_spinner=False)
def _load_handbook_data() -> list[dict[str, Any]]:
    direct_paths = _discover_files(("handbook", "policy", "policies"), (".json",))
    rows = _load_json_rows(direct_paths)
    if rows:
        return rows
    fallback_paths = _discover_files(("chunk", "structured", "unstructured"), (".json",))
    return _load_json_rows(fallback_paths)


def _extract_keywords(prompt: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+", prompt.lower())
    skip = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "about",
        "that",
        "this",
        "what",
        "when",
        "where",
        "which",
        "please",
        "show",
        "give",
        "tell",
        "need",
        "help",
        "academic",
        "calendar",
        "iit",
        "tech",
        "illinois",
    }
    return [w for w in words if len(w) > 2 and w not in skip and not w.isdigit()]


def _normalize_casual_text(prompt: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", prompt.lower()).strip()
    cleaned = re.sub(r"(.)\1+", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_greeting(prompt: str) -> bool:
    cleaned = _normalize_casual_text(prompt)
    greetings = {
        "hi",
        "hey",
        "hello",
        "yo",
        "hiya",
        "good morning",
        "good afternoon",
        "good evening",
        "whats up",
        "what s up",
    }
    return cleaned in greetings


def _casual_response(prompt: str) -> str | None:
    cleaned = _normalize_casual_text(prompt)
    first_word = cleaned.split(" ")[0] if cleaned else ""
    greeting_words = {"hi", "hey", "hello", "helo", "yo", "hiya"}
    if (
        cleaned in {"good morning", "good afternoon", "good evening", "whats up", "what s up"}
        or first_word in greeting_words
        or first_word.startswith("hel")
    ):
        return "Hey! 👋 What’s up—how can I help you today?"
    if cleaned in {"how are you", "how are you doing"}:
        return "I’m doing great, thanks for asking! What can I help you with today?"
    if cleaned in {"thanks", "thank you", "thx"}:
        return "You’re welcome! Happy to help."
    if cleaned in {"bye", "goodbye", "see you", "see ya"}:
        return "See you! 👋 If you need anything later, I’m here."
    if cleaned in {"ok", "okay", "cool", "great", "nice"}:
        return "Awesome. Want to ask another question?"
    return None


def _autocorrect_prompt(prompt: str) -> str:
    substitutions = {
        r"\btuitiion\b": "tuition",
        r"\btuituin\b": "tuition",
        r"\baademic\b": "academic",
        r"\bcalender\b": "calendar",
        r"\bdirctory\b": "directory",
        r"\bpolcies\b": "policies",
        r"\bhandbok\b": "handbook",
    }
    corrected = prompt
    for wrong, right in substitutions.items():
        corrected = re.sub(wrong, right, corrected, flags=re.IGNORECASE)
    return corrected


def _is_tuition_query(prompt: str) -> bool:
    p = prompt.lower()
    return any(token in p for token in ("tuition", "tuiti", "tuituin", "fees", "fee", "cost"))


def _is_calendar_query(prompt: str) -> bool:
    p = prompt.lower()
    return any(token in p for token in ("calendar", "academic", "deadline", "semester", "term", "exam", "spring", "fall"))


def _is_directory_query(prompt: str) -> bool:
    p = prompt.lower()
    return any(token in p for token in ("directory", "contact", "department", "phone", "email", "office"))


def _is_policy_query(prompt: str) -> bool:
    p = prompt.lower()
    return any(
        token in p
        for token in (
            "policy",
            "policies",
            "academic integrity",
            "attendance",
            "withdraw",
            "probation",
            "conduct",
            "grade appeal",
        )
    )


def _is_handbook_query(prompt: str) -> bool:
    p = prompt.lower()
    return any(
        token in p
        for token in (
            "handbook",
            "student handbook",
            "coterminal handbook",
            "academic handbook",
        )
    )


def _top_matches(items: list[dict[str, Any]], prompt: str, fields: tuple[str, ...], limit: int = 3) -> list[dict[str, Any]]:
    keywords = _extract_keywords(prompt)
    scored: list[tuple[int, dict[str, Any]]] = []
    for item in items:
        haystack = " ".join(str(item.get(field, "")).lower() for field in fields)
        score = sum(1 for kw in keywords if kw in haystack)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:limit]]


def _format_money(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"${value:,.0f}"
    return "N/A"


def _tuition_response(prompt: str, variant: str) -> str:
    rows = _load_tuition_data()
    matches = _top_matches(rows, prompt, ("school", "section", "fee_name", "program", "content"), limit=3)
    matches.sort(
        key=lambda r: 0 if "tuition" in str(r.get("fee_name", "")).lower() else 1
    )
    if not matches:
        return (
            "I cant find the relevant information right now.\n\n"
            "Can you tell me a bit more about what you are looking for?"
        )
    lines = []
    for row in matches:
        school = row.get("school", "Illinois Tech")
        section = row.get("section", "Tuition")
        fee_name = row.get("fee_name") or "Tuition"
        amount = _format_money(row.get("amount_value"))
        unit = (row.get("unit") or "").replace("_", " ")
        source = row.get("source_url", "")
        detail = f"- {school}: {section} - {fee_name}: {amount}"
        if unit:
            detail += f" ({unit})"
        if source:
            detail += f"\n  Source: {source}"
        lines.append(detail)

    if variant == "A":
        intro = "Absolutely - here are the closest tuition details I found:"
        outro = "If you share your exact program and term, I can narrow this to one precise amount."
    else:
        intro = "Sure, I checked the tuition data and found these likely matches:"
        outro = "Want me to filter this by your degree level and semester so it reads like a single estimate?"
    return f"{intro}\n\n" + "\n".join(lines) + f"\n\n{outro}"


def _calendar_response(prompt: str, variant: str) -> str:
    rows = _load_calendar_data()
    lower_prompt = prompt.lower()
    term_hint = ""
    for season in ("spring", "summer", "fall", "winter"):
        if season in lower_prompt:
            term_hint = season
            break

    if term_hint:
        filtered_rows = [r for r in rows if term_hint in str(r.get("term", "")).lower()]
    else:
        filtered_rows = rows

    matches = _top_matches(filtered_rows, prompt, ("term", "event_name"), limit=4)

    if not matches:
        today = date.today().isoformat()
        upcoming = [
            r for r in rows if isinstance(r.get("start_date"), str) and r.get("start_date") >= today
        ][:4]
    else:
        upcoming = matches

    if not upcoming:
        return (
            "I cant find the relevant information right now. "
            "Can you tell me a bit more about the term or event you need?"
        )

    lines = []
    for row in upcoming:
        term = row.get("term", "Academic Calendar")
        event_name = row.get("event_name", "Event")
        start_date = row.get("start_date", "")
        source_urls = row.get("source_urls") or []
        source = source_urls[0] if source_urls else ""
        detail = f"- {start_date}: {event_name} ({term})"
        if source:
            detail += f"\n  Source: {source}"
        lines.append(detail)

    lead = (
        "Happy to help - here are the most relevant academic calendar events:"
        if variant == "A"
        else "Here are the calendar events that best match your question:"
    )
    follow_up = (
        "Can you tell me a bit more so I can narrow it down for you?"
        if variant == "A"
        else "Share a little more detail and I can make this more precise."
    )
    return f"{lead}\n\n" + "\n".join(lines) + f"\n\n{follow_up}"


def _directory_response(prompt: str, variant: str) -> str:
    rows = _load_directory_data()
    matches = _top_matches(rows, prompt, ("Name", "Department", "Description", "Email", "Phone"), limit=4)
    if not matches:
        matches = rows[:4]

    if not matches:
        return "I cant find the relevant information right now. Can you tell me a bit more about the department or office?"

    lines = []
    for row in matches:
        name = row.get("Name") or row.get("\ufeffName") or row.get("Department") or "Department"
        dept = row.get("Department", "")
        phone = row.get("Phone", "N/A")
        email = row.get("Email", "N/A")
        building = row.get("Building", "N/A")
        detail = f"- {name} ({dept})\n  Phone: {phone} | Email: {email} | Building: {building}"
        lines.append(detail)

    intro = (
        "Sure - here are the directory contacts that look most relevant:"
        if variant == "A"
        else "Got it. I found these likely contacts from the Illinois Tech directory:"
    )
    next_step = (
        "Can you tell me a bit more so I can find the best match for you?"
        if variant == "A"
        else "Share a little more detail and I will narrow this down for you."
    )
    return f"{intro}\n\n" + "\n".join(lines) + f"\n\n{next_step}"


def _policy_response(prompt: str, variant: str) -> str:
    rows = _load_handbook_data()
    matches = _top_matches(
        rows,
        prompt,
        ("title", "section", "content", "chunk_text", "text", "source", "source_url"),
        limit=3,
    )
    if not matches:
        return (
            "I cant find the relevant information right now.\n\n"
            "Can you tell me a bit more so I can help better?"
        )

    lines = []
    for row in matches:
        text = str(
            row.get("content")
            or row.get("chunk_text")
            or row.get("text")
            or "Policy detail found."
        ).strip()
        snippet = text[:260] + ("..." if len(text) > 260 else "")
        title = str(row.get("title") or row.get("section") or "Policy")
        source = str(row.get("source_url") or row.get("source") or "").strip()
        detail = f"- {title}: {snippet}"
        if source:
            detail += f"\n  Source: {source}"
        lines.append(detail)

    intro = (
        "Absolutely - here are the closest handbook/policy details I found:"
        if variant == "A"
        else "Sure, here are policy and handbook details relevant to your question:"
    )
    outro = (
        "Can you tell me a bit more so I can make this more precise?"
        if variant == "A"
        else "Share a little more detail and I can narrow this down for you."
    )
    return f"{intro}\n\n" + "\n".join(lines) + f"\n\n{outro}"


def _general_conversation_response(prompt: str, variant: str) -> str:
    casual = _casual_response(prompt)
    if casual:
        return casual
    topics_text = _topic_list_text()
    if variant == "A":
        return (
            f"Thanks for your question: \"{prompt.strip()}\".\n\n"
            f"I can help with these topics: {topics_text}. "
            "Tell me what you need, and I will help step by step."
        )
    return (
        f"Great question. You asked: \"{prompt.strip()}\".\n\n"
        f"I am currently tuned for these Illinois Tech topics: {topics_text}. "
        "Pick one area and I will give you a clear, conversational answer."
    )


def _detect_prompt_topic(prompt: str) -> str | None:
    if _is_tuition_query(prompt):
        return "Tuition"
    if _is_calendar_query(prompt):
        return "Academic Calendar"
    if _is_directory_query(prompt):
        return "Directory"
    if _is_handbook_query(prompt):
        return "Handbook"
    if _is_policy_query(prompt):
        return "Policies"
    return None


def _topic_locked_response(prompt: str, variant: str, selected_topic: str) -> str:
    corrected_prompt = _autocorrect_prompt(prompt)
    casual = _casual_response(corrected_prompt)
    if casual:
        return casual
    detected_topic = _detect_prompt_topic(corrected_prompt)
    if detected_topic and detected_topic != selected_topic:
        return (
            f"I am currently in {selected_topic} mode.\n\n"
            "Use 'Choose Topic' in the sidebar if you want to switch."
        )

    if selected_topic == "Tuition":
        return _tuition_response(corrected_prompt, variant)
    if selected_topic == "Academic Calendar":
        return _calendar_response(corrected_prompt, variant)
    if selected_topic == "Directory":
        return _directory_response(corrected_prompt, variant)
    if selected_topic in ("Policies", "Handbook"):
        return _policy_response(corrected_prompt, variant)
    return _general_conversation_response(corrected_prompt, variant)


def _mock_response(prompt: str, variant: str, selected_topic: str | None = None) -> str:
    corrected_prompt = _autocorrect_prompt(prompt)
    if selected_topic:
        return _topic_locked_response(corrected_prompt, variant, selected_topic)
    if _is_tuition_query(corrected_prompt):
        return _tuition_response(corrected_prompt, variant)
    if _is_calendar_query(corrected_prompt):
        return _calendar_response(corrected_prompt, variant)
    if _is_directory_query(corrected_prompt):
        return _directory_response(corrected_prompt, variant)
    if _is_policy_query(corrected_prompt) or _is_handbook_query(corrected_prompt):
        return _policy_response(corrected_prompt, variant)
    return _general_conversation_response(corrected_prompt, variant)


def _fetch_api_response(prompt: str, endpoint: str, fallback_variant: str, selected_topic: str | None) -> str:
    try:
        payload = {"prompt": prompt}
        if selected_topic:
            payload["topic"] = selected_topic
        response = requests.post(endpoint, json=payload, timeout=20)
        response.raise_for_status()
        payload = response.json()
        for key in ("response", "answer", "content", "text"):
            if key in payload and isinstance(payload[key], str):
                return payload[key]
        return str(payload)
    except Exception:
        return _mock_response(prompt, fallback_variant, selected_topic)


def get_model_a_response(prompt: str, use_api_mode: bool, selected_topic: str | None) -> tuple[str, dict]:
    if use_api_mode:
        raw_endpoint = os.getenv("MODEL_A_ENDPOINT", "").strip()
        # Clean up any trailing comments (e.g. "#group 6") from .env
        endpoint = raw_endpoint.split("#")[0].strip()

        if endpoint:
            try:
                # 1. Initialize session if missing
                if not st.session_state.get("session_id_a"):
                    # Derive session endpoint from query endpoint
                    session_url = endpoint.replace("/api/query", "/api/session/new")
                    try:
                        s_resp = requests.post(session_url, timeout=5)
                        if s_resp.ok:
                            st.session_state.session_id_a = s_resp.json().get("session_id")
                    except Exception:
                        # Continue even if session init fails; the main query might still work
                        pass

                # 2. Build chat_history in the format Model A expects
                history = [
                    {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                    for m in st.session_state.get("history_a", [])
                    if str(m.get("role", "")) in {"user", "assistant"} and str(m.get("content", "")).strip()
                ]

                payload = {
                    "question": prompt,
                    "method": "traffic_cop",
                    "session_id": st.session_state.get("session_id_a"),
                    "chat_history": history
                }

                resp = requests.post(endpoint, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                # Extract answer and session_id (update if it changed)
                answer = str(data.get("answer") or "").strip()
                new_session_id = data.get("session_id")
                if new_session_id:
                    st.session_state.session_id_a = new_session_id

                return answer, data
            except Exception as e:
                # Return the error message directly to the UI so we can debug the integration
                return f"Model A API Error: {str(e)}\n\n(Endpoint: {endpoint})", {}

        return f"Model A: No endpoint set in .env (MODEL_A_ENDPOINT is empty). Falling back to internal mock mode.", {}

    return _mock_response(prompt, "A", selected_topic), {}


def get_model_b_response(prompt: str, use_api_mode: bool, selected_topic: str | None) -> tuple[str, dict]:
    if use_api_mode:
        raw_endpoint = os.getenv("MODEL_B_ENDPOINT", "https://bessie-pinnate-noumenally.ngrok-free.dev/ask").strip()
        endpoint = raw_endpoint.split("#")[0].strip()
        try:
            # 1. Build chat_history
            history = [
                {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                for m in st.session_state.get("history_b", [])
                if str(m.get("role", "")) in {"user", "assistant"} and str(m.get("content", "")).strip()
            ]

            # 2. Build pending_context if we have one
            pending_payload = None
            pending_query = st.session_state.get("pending_query_b")
            pending_msg = st.session_state.get("pending_clarification_msg_b")
            pending_domain = st.session_state.get("pending_domain_b")
            if pending_query and pending_msg:
                pending_payload = {
                    "original_query": pending_query,
                    "clarification_message": pending_msg,
                    "domain": pending_domain,
                }

            payload = {
                "prompt": prompt,
                "topic": selected_topic,
                "chat_history": history,
                "pending_context": pending_payload
            }

            req_headers = {"Content-Type": "application/json"}
            if "ngrok" in endpoint.lower():
                req_headers["ngrok-skip-browser-warning"] = "true"

            resp = requests.post(endpoint, json=payload, timeout=30, headers=req_headers)
            resp.raise_for_status()
            data = resp.json()

            answer = str(data.get("response") or "").strip()
            is_clarification = data.get("is_clarification", False)
            new_pending = data.get("pending_context")

            if not answer:
                raise ValueError("Model B API returned an empty response")

            if is_clarification and new_pending:
                st.session_state.pending_query_b = new_pending.get("original_query")
                st.session_state.pending_clarification_msg_b = new_pending.get("clarification_message")
                st.session_state.pending_domain_b = new_pending.get("domain")
                st.session_state.last_turn_was_answer = False
            else:
                st.session_state.pending_query_b = None
                st.session_state.pending_clarification_msg_b = None
                st.session_state.pending_domain_b = None

                not_sure = "i don't have" in answer.lower() or "couldn't find" in answer.lower()
                st.session_state.last_turn_was_answer = not not_sure

            return answer, data
        except Exception as e:
            # Fall through to local mode on API failure
            pass

    # ── Local mode: direct import (for development / when no endpoint is set) ──
    _ = (use_api_mode, selected_topic)

    try:
        chat_history = [
            {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
            for m in st.session_state.get("history_b", [])
            if str(m.get("role", "")) in {"user", "assistant"} and str(m.get("content", "")).strip()
        ]

        pending = st.session_state.get("pending_query_b")
        pending_msg = st.session_state.get("pending_clarification_msg_b")
        pending_domain = st.session_state.get("pending_domain_b")

        answer = ""
        is_clarification = False
        clar_msg = ""
        clar_domain = ""
        metadata = {}

        if pending and pending_msg:
            action = classify_pending_response(pending, pending_msg, prompt)
            if action == "CANCEL":
                st.session_state.pending_query_b = None
                st.session_state.pending_clarification_msg_b = None
                st.session_state.pending_domain_b = None
                st.session_state.last_turn_was_answer = False
                return "No problem — ask me another Illinois Tech question whenever you are ready.", {}
            if action == "NEW_TOPIC":
                st.session_state.pending_query_b = None
                st.session_state.pending_clarification_msg_b = None
                st.session_state.pending_domain_b = None
                answer, sources, route_details, is_clarification, clar_msg, clar_domain = get_answer(
                    query=prompt, chat_history=chat_history
                )
                metadata = {"sources": sources, "route_details": route_details}
            else:
                combined = (
                    reformulate_query(pending, prompt.strip())
                    if GROQ_API_KEY
                    else f"{pending} {prompt}".strip()
                )
                st.session_state.pending_query_b = None
                st.session_state.pending_clarification_msg_b = None
                st.session_state.pending_domain_b = None
                answer, sources, route_details, is_clarification, clar_msg, clar_domain = get_answer_for_domain(
                    combined,
                    pending_domain,
                    chat_history=chat_history,
                )
                metadata = {"sources": sources, "route_details": route_details}
        else:
            answer, sources, route_details, is_clarification, clar_msg, clar_domain = get_answer(
                query=prompt, chat_history=chat_history
            )
            metadata = {"sources": sources, "route_details": route_details}

        if is_clarification:
            st.session_state.pending_query_b = prompt
            st.session_state.pending_clarification_msg_b = clar_msg
            st.session_state.pending_domain_b = clar_domain or None
            st.session_state.last_turn_was_answer = False
        else:
            not_sure = "i don't have" in answer.lower() or "couldn't find" in answer.lower()
            st.session_state.last_turn_was_answer = not not_sure

        return answer, metadata
    except Exception as e:
        st.session_state.last_turn_was_answer = False
        return f"Model B error: {str(e)}", {}


def main() -> None:
    st.set_page_config(page_title="Chatbot Comparison", page_icon=":speech_balloon:", layout="wide")
    css_path = Path(__file__).parent.parent / "ui_utils" / "styles.css"
    load_css(css_path)
    _init_state()

    # Use "API-ready" to call the endpoints from your .env or environment
    response_mode = "API-ready"
    with st.sidebar:
        st.markdown("### Actions")
        if st.session_state.selected_topic:
            st.caption(f"Current topic: {st.session_state.selected_topic}")
        if st.button("Choose Topic", use_container_width=True):
            st.session_state.selected_topic = None
            st.session_state.compare_result = ""
            st.session_state.show_feedback_form = False
            st.session_state.feedback_status = ""
            st.session_state.pending_query_b = None
            st.session_state.pending_clarification_msg_b = None
            st.session_state.pending_domain_b = None
            st.rerun()
        if st.button("Clear conversation", use_container_width=True):
            if st.session_state.selected_topic:
                _reset_histories_for_topic(st.session_state.selected_topic)
            else:
                st.session_state.pop("history_a", None)
                st.session_state.pop("history_b", None)
                st.session_state.compare_result = ""
                st.session_state.show_feedback_form = False
                st.session_state.feedback_status = ""
            st.session_state.pending_query_b = None
            st.session_state.pending_clarification_msg_b = None
            st.session_state.pending_domain_b = None
            st.rerun()

    st.markdown('<div class="app-shell">', unsafe_allow_html=True)
    render_title_block("center")
    if not st.session_state.selected_topic:
        _render_topic_selection()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown(
        f'<div class="selected-topic-banner">Topic: <strong>{st.session_state.selected_topic}</strong></div>',
        unsafe_allow_html=True,
    )
    show_typing_a = st.session_state.is_generating
    show_typing_b = st.session_state.is_generating

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        is_api = bool(os.getenv("MODEL_A_ENDPOINT"))
        model_name = "MODEL 1 (API)" if is_api else "MODEL 1 (MOCK)"
        render_chat_panel("CHATBOT A", model_name, "a", st.session_state.history_a, show_typing=show_typing_a)
    with col_b:
        is_api = bool(os.getenv("MODEL_B_ENDPOINT"))
        model_name = "MODEL 2 (API)" if is_api else "MODEL 2 (MOCK)"
        render_chat_panel("CHATBOT B", model_name, "b", st.session_state.history_b, show_typing=show_typing_b)

    with st.form("send_form", clear_on_submit=True):
        input_col, button_col = st.columns([6, 1])
        with input_col:
            prompt = st.text_input(
                "Compare prompt",
                placeholder=f"Ask your question about {st.session_state.selected_topic}...",
                label_visibility="collapsed",
                disabled=st.session_state.is_generating,
            )
            st.caption("Press Enter or click SEND")
        with button_col:
            submitted = st.form_submit_button(
                "SEND",
                use_container_width=True,
                type="primary",
                disabled=st.session_state.is_generating,
            )

    if submitted and prompt.strip() and not st.session_state.is_generating:
        _enqueue_user_prompt(prompt)
        st.rerun()

    if st.session_state.is_generating and st.session_state.pending_prompt:
        user_prompt = st.session_state.pending_prompt.strip()
        use_api_mode = response_mode == "API-ready"
        time.sleep(0.45)
        selected_topic = st.session_state.selected_topic
        response_a, meta_a = get_model_a_response(user_prompt, use_api_mode, selected_topic)
        response_b, meta_b = get_model_b_response(user_prompt, use_api_mode, selected_topic)
        _append_message("history_a", "assistant", response_a, metadata=meta_a)
        _append_message("history_b", "assistant", response_b, metadata=meta_b)

        st.session_state.pending_prompt = ""
        st.session_state.is_generating = False
        st.rerun()

    _handle_footer_actions(disabled=st.session_state.is_generating)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()