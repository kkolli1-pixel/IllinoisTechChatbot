import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

load_dotenv(project_root / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

try:
    from search.documents_search import documents_rrf_search
except ImportError as e:
    st.error(f"Could not import documents search: {e}")
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


def build_documents_context(hits):
    context = []
    for h in hits:
        s = h.get("_source", {})
        topic = (s.get("topic") or "").strip()
        doc_name = (s.get("doc_name") or "").strip()
        content = (s.get("content") or "").strip()
        block = (
            f"[Policy Section: {topic} | Handbook: {doc_name}]\n{content}"
            if (topic or doc_name)
            else content
        )
        context.append(block)
    return context


def get_reply(query: str) -> tuple[str, list]:
    q = (query or "").strip()
    if not q:
        return "Please enter a question.", []

    hits = documents_rrf_search(q)
    if isinstance(hits, dict) and hits.get("needs_clarification"):
        opts = hits.get("options") or []
        msg = hits.get("message") or "Could you clarify?"
        if opts:
            msg = f"{msg}\n\n*Options: {' · '.join(opts)}*"
        return msg, []

    if not hits:
        return "I couldn't find matching policy or handbook text.", []

    context_text = "\n\n---\n\n".join(build_documents_context(hits))
    sources = list(dict.fromkeys(build_sources(hits)))

    system = (
        "You are a helpful university assistant for Illinois Institute of Technology. "
        "Answer using ONLY the context below. Cite the handbook or topic when helpful. "
        "If the answer is not in the context, say you are not sure. "
        "When the user says \"the handbook\" or \"the document\" without specifying, "
        "use the document and section names in the context to decide which is relevant."
    )
    if not GROQ_API_KEY:
        return f"**Context:**\n\n{context_text}", sources

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {q}\n\nContext:\n{context_text}"},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip(), sources
    except Exception:
        return f"**Context:**\n\n{context_text}", sources


st.set_page_config(page_title="IIT — Policies & handbooks")
st.title("Handbooks & policies")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask about academic policies, GPA, conduct, housing, visas, etc.",
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s}")

if prompt := st.chat_input("E.g., What is academic probation?"):
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
