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
    from search.contacts_search import contacts_rrf_search
except ImportError as e:
    st.error(f"Could not import contacts search: {e}")
    st.stop()


def build_sources(hits):
    urls = []
    for h in hits:
        s = h.get("_source", {})
        u = (s.get("source_url") or "").strip()
        if u and u not in urls:
            urls.append(u)
    return urls


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


def get_reply(query: str) -> tuple[str, list]:
    q = (query or "").strip()
    if not q:
        return "Please enter a question.", []

    hits = contacts_rrf_search(q, top_k=10)
    if isinstance(hits, dict) and hits.get("needs_clarification"):
        opts = hits.get("options") or []
        msg = hits.get("message") or "Could you clarify?"
        if opts:
            msg = f"{msg}\n\n*Options: {' · '.join(opts)}*"
        return msg, []

    if not hits:
        return "I couldn't find a matching contact or department.", []

    context_text = "\n\n---\n\n".join(build_contacts_context(hits))
    sources = list(dict.fromkeys(build_sources(hits)))

    system = (
        "You are a helpful university assistant for Illinois Institute of Technology. "
        "Answer using ONLY the context below. Be concise. "
        "If the answer is not in the context, say you are not sure."
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


st.set_page_config(page_title="IIT — Staff contacts")
st.title("Staff & office contacts")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask for a department, office, phone, or email."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s}")

if prompt := st.chat_input("E.g., Office of the Registrar phone"):
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
