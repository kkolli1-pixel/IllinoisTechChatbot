import json
import os
import sys
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

load_dotenv(project_root / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
THETA_API_KEY = os.getenv("THETA_API_KEY")
THETA_API_URL = os.getenv("THETA_API_URL")

try:
    from router.calendar_router import route_query as calendar_route_query
except ImportError as e:
    st.error(f"Could not import calendar router: {e}")
    st.stop()


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
        body = semantic_text if semantic_text else event_name
        context.append(f"{header}\n{body}")
    return context


def theta_parse_response(raw: str) -> str | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or choices[0].get("delta", {})
            return (msg.get("content") or "").strip()
        return None
    except ValueError:
        pass
    chunks = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:") and (part := line[5:].strip()) and part != "[DONE]":
            try:
                obj = json.loads(part)
                for c in (obj.get("choices") or []):
                    d = c.get("delta") or c.get("message") or {}
                    if d.get("content"):
                        chunks.append(d["content"])
            except (ValueError, KeyError):
                pass
    return "".join(chunks).strip() if chunks else None


def get_reply(query: str) -> tuple[str, list]:
    q = (query or "").strip()
    if not q:
        return "Please enter a question.", []

    hits = calendar_route_query(q)
    if isinstance(hits, dict) and hits.get("needs_clarification"):
        opts = hits.get("options") or []
        msg = hits.get("message") or "Could you clarify?"
        if opts:
            msg = f"{msg}\n\n*Options: {' · '.join(opts)}*"
        return msg, []

    if not hits:
        return "I couldn't find matching calendar events.", []

    context_text = "\n\n---\n\n".join(build_calendar_context(hits))
    sources = list(dict.fromkeys(build_sources(hits)))

    system = (
        "You are a helpful university assistant for Illinois Institute of Technology. "
        "Answer using ONLY the context below. Be concise. "
        "Do not mention chunks or indices. Say \"The calendar states\" when appropriate."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {q}\n\nContext:\n{context_text}"},
    ]

    if GROQ_API_KEY:
        try:
            from groq import Groq

            client = Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip(), sources
        except Exception:
            pass

    if THETA_API_KEY and THETA_API_URL:
        try:
            theta_url = THETA_API_URL.strip().strip('"').strip("'")
            theta_key = THETA_API_KEY.strip()
            resp = requests.post(
                theta_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {theta_key}",
                },
                json={
                    "input": {
                        "max_tokens": 500,
                        "messages": messages,
                        "temperature": 0.0,
                        "top_p": 0.7,
                    },
                    "stream": False,
                    "variant": "quantized",
                },
                timeout=30,
            )
            resp.raise_for_status()
            answer = theta_parse_response(resp.text)
            if answer:
                return answer, sources
        except Exception:
            pass

    return f"**Context:**\n\n{context_text}", sources


st.set_page_config(page_title="IIT — Academic calendar")
st.title("Academic calendar")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask about dates, deadlines, breaks, or registration."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s}")

if prompt := st.chat_input("E.g., When is spring break?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    reply, sources = get_reply(prompt)
    with st.chat_message("assistant"):
        st.markdown(reply)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
