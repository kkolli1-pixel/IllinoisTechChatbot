"""
Generate gold_answers.json from raw data files for RAG evaluation.

For each of the 89 test questions, this script:
1. Finds relevant chunks from the source data (calendar, contacts, documents, tuition)
2. Uses Groq LLM to synthesize a factual gold answer grounded in those chunks
3. Outputs gold_answers.json with {section, query, relevant_chunks, gold_answer}

Usage:
    python generate_gold_answers.py              # generate all
    python generate_gold_answers.py "Calendar"   # generate one section
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
THETA_API_KEY = os.getenv("THETA_API_KEY")
THETA_API_URL = os.getenv("THETA_API_URL")

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_PATH = Path(__file__).parent / "gold_answers.json"


# ── Load raw data ──────────────────────────────────────────────────────────────

def load_calendar():
    with open(DATA_DIR / "calendar_chunks.json") as f:
        return json.load(f)

def load_contacts():
    rows = []
    with open(DATA_DIR / "Contacts data.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def load_documents():
    with open(DATA_DIR / "Unstructured chunks.json") as f:
        return json.load(f)

def load_tuition():
    with open(DATA_DIR / "tuition_data.json") as f:
        return json.load(f)


# ── Chunk matchers per domain ──────────────────────────────────────────────────

def _match_calendar(query, calendar_data):
    q = query.lower()
    matched = []

    term_filters = []
    if "spring 2026" in q:
        term_filters.append("Spring 2026")
    if "summer" in q and ("2026" in q or "1" in q):
        term_filters.append("Summer 2026")
    if "fall 2026" in q:
        term_filters.append("Fall 2026")
    if "november" in q or "late november" in q:
        term_filters.append("Fall 2026")
    if "april" in q:
        term_filters.append("Spring 2026")
    if "may" in q:
        term_filters.append("Spring 2026")
        term_filters.append("Summer 2026")
    if not term_filters:
        term_filters.extend(["Spring 2026", "Summer 2026", "Fall 2026"])

    event_keywords = []
    if "start" in q or "begin" in q:
        event_keywords.extend(["begin", "start"])
    if "final" in q and "exam" in q:
        event_keywords.extend(["final exam", "final grading"])
    if "commencement" in q or "graduation" in q:
        event_keywords.extend(["commencement", "conferral"])
    if "withdraw" in q:
        event_keywords.extend(["withdraw"])
    if "midterm" in q:
        event_keywords.extend(["midterm"])
    if "add" in q and "drop" in q:
        event_keywords.extend(["add/drop", "add a course", "drop"])
    if "registration" in q or "register" in q:
        event_keywords.extend(["registration"])
    if "spring break" in q:
        event_keywords.extend(["spring break"])
    if "no classes" in q or ("classes" in q and ("off" in q or "holiday" in q or "break" in q)):
        event_keywords.extend(["no classes", "break", "holiday"])
    if "last day" in q:
        event_keywords.extend(["last day"])
    if "lease" in q or "finish" in q or "end" in q:
        event_keywords.extend(["last day of", "courses"])
    if "wedding" in q or "november" in q:
        event_keywords.extend(["thanksgiving", "no classes", "break"])
    if "april" in q:
        event_keywords.extend(["no classes", "break"])

    for evt in calendar_data:
        term = evt.get("term", "")
        name = evt.get("event_name", "").lower()
        if "coursera" in term.lower():
            continue
        term_match = any(tf.lower() in term.lower() for tf in term_filters)
        if not term_match:
            continue
        if event_keywords:
            kw_match = any(ek in name for ek in event_keywords)
            if not kw_match:
                continue
        matched.append({
            "source": "calendar",
            "term": evt["term"],
            "event_name": evt["event_name"],
            "start_date": evt["start_date"],
            "end_date": evt["end_date"],
        })

    if not matched:
        for evt in calendar_data:
            if "coursera" in evt.get("term", "").lower():
                continue
            term_match = any(tf.lower() in evt.get("term", "").lower() for tf in term_filters)
            if term_match:
                matched.append({
                    "source": "calendar",
                    "term": evt["term"],
                    "event_name": evt["event_name"],
                    "start_date": evt["start_date"],
                    "end_date": evt["end_date"],
                })

    return matched[:15]


def _match_contacts(query, contacts_data):
    q = query.lower()
    matched = []

    search_terms = []
    if "registrar" in q:
        search_terms.extend(["registrar"])
    if "physics" in q:
        search_terms.extend(["physics"])
    if "pritzker" in q:
        search_terms.extend(["pritzker"])
    if "transcript" in q:
        search_terms.extend(["registrar", "transcript"])
    if "registration" in q or "register" in q:
        search_terms.extend(["registrar", "registration"])
    if "hold" in q:
        search_terms.extend(["registrar", "student accounting", "student affairs"])
    if "wanger" in q:
        search_terms.extend(["wanger"])
    if "semester off" in q or "leave" in q:
        search_terms.extend(["academic affairs", "registrar"])
    if "advisor" in q or "advising" in q:
        search_terms.extend(["academic affairs", "registrar", "advisor"])

    for row in contacts_data:
        name = (row.get("Name") or "").lower()
        dept = (row.get("Department") or "").lower()
        cat = (row.get("Category") or "").lower()
        desc = (row.get("Description") or "").lower()
        combined = f"{name} {dept} {cat} {desc}"
        if any(st in combined for st in search_terms):
            entry = {"source": "contacts", "name": row.get("Name", "")}
            for field in ("Department", "Phone", "Fax", "Email", "Building", "Address"):
                if row.get(field):
                    entry[field.lower()] = row[field]
            matched.append(entry)

    return matched[:10]


def _match_documents(query, documents_data):
    """Relevance-scored document matching with doc-name boosting."""
    q = query.lower()

    # Keywords to search for in chunk content
    kw_map = {
        "full-time": ["full-time", "full time", "full-time status"],
        "full time": ["full-time", "full time", "full-time status"],
        "credit": ["credit hour", "credit"],
        "pass/fail": ["pass/fail", "pass-fail", "pass fail"],
        "pass fail": ["pass/fail", "pass-fail", "pass fail"],
        "withdraw": ["withdraw", "withdrawal", "w grade"],
        "drop": ["drop", "add/drop", "dropping"],
        "transfer credit": ["transfer credit", "transfer"],
        "retake": ["repeat", "retake", "course repetition", "grade exclusion"],
        "repeat": ["repeat", "retake", "course repetition", "grade exclusion"],
        "intellectual property": ["intellectual property"],
        "coterminal": ["co-terminal", "coterminal"],
        "co-terminal": ["co-terminal", "coterminal"],
        "financial aid": ["financial aid"],
        "gpa": ["gpa", "grade point", "grade-point"],
        "transcript": ["transcript", "parchment"],
        "refund": ["refund", "tuition refund"],
        "late registration": ["late registration"],
        "registration": ["registration", "register"],
        "register": ["registration", "register"],
        "f-1": ["f-1", "visa", "immigration", "international"],
        "visa": ["f-1", "visa", "immigration"],
        "graduating": ["graduation", "degree conferral", "commencement"],
        "graduation": ["graduation", "degree conferral", "commencement"],
        "leave of absence": ["leave of absence"],
        "alumni": ["alumni", "former student"],
        "hold": ["hold", "registration prohibited"],
        "add/drop": ["add/drop", "add a course", "drop a course"],
        "add a course": ["add/drop", "add a course", "registration"],
        "incomplete": ["incomplete grade"],
        "probation": ["probation", "academic progress"],
        "overload": ["overload", "credit hour limit"],
    }

    content_keywords = []
    for trigger, kws in kw_map.items():
        if trigger in q:
            content_keywords.extend(kws)
    content_keywords = list(set(content_keywords))

    if not content_keywords:
        words = re.findall(r'\b[a-z]{4,}\b', q)
        stop = {"what", "when", "where", "does", "will", "that", "this", "from", "with",
                "have", "about", "there", "their", "they", "should", "would", "could",
                "more", "after", "before", "also", "than", "been", "into", "some",
                "still", "just", "many", "much", "like", "need", "want", "take"}
        content_keywords = [w for w in words if w not in stop][:6]

    # Map query topics -> preferred doc_name prefixes (boost these sources)
    preferred_docs = []
    if "transcript" in q or "alumni" in q:
        preferred_docs.extend(["transcripts", "alumni", "registrar"])
    if "withdraw" in q:
        preferred_docs.extend(["registration_drop_vs_withdraw", "registration_withdrawal",
                               "graduate_academic_policies", "policies_hardship"])
    if "drop" in q and "add" in q:
        preferred_docs.extend(["registration_add_drop", "registration_drop_vs_withdraw"])
    if "drop" in q:
        preferred_docs.extend(["registration_drop_vs_withdraw", "registration_add_drop"])
    if "refund" in q:
        preferred_docs.extend(["tuition", "policies_hardship", "financial_aid_policies"])
    if "pass/fail" in q or "pass fail" in q:
        preferred_docs.extend(["Taking a Course Pass/Fail", "graduate_academic_policies"])
    if "full-time" in q or "full time" in q:
        preferred_docs.extend(["Full-time Status", "Credit Hour Limits"])
    if "repeat" in q or "retake" in q:
        preferred_docs.extend(["registration_repeat"])
    if "transfer credit" in q or "transfer" in q:
        preferred_docs.extend(["graduate_academic_policies", "Coterminal"])
    if "coterminal" in q or "co-terminal" in q:
        preferred_docs.extend(["Co-terminal", "Coterminal", "handbook"])
    if "registration" in q or "register" in q:
        preferred_docs.extend(["registration", "Internet Course Registration",
                               "Late Registration Request"])
    if "gpa" in q or "grade" in q:
        preferred_docs.extend(["graduate_academic_policies", "registrar_grade_legend"])
    if "financial aid" in q:
        preferred_docs.extend(["financial_aid_policies"])
    if "hold" in q:
        preferred_docs.extend(["hold"])
    if "intellectual property" in q:
        preferred_docs.extend(["student handbook"])
    if "leave" in q or "semester off" in q:
        preferred_docs.extend(["registration_withdrawal", "policies_hardship"])
    if "commencement" in q or "graduation" in q:
        preferred_docs.extend(["commencement"])

    scored = []
    for chunk in documents_data:
        content = (chunk.get("content") or "").lower()
        topic = (chunk.get("Topic") or chunk.get("topic") or chunk.get("section_title") or "").lower()
        doc_name = (chunk.get("doc_name") or "").lower()
        chunk_id = (chunk.get("chunk_id") or "").lower()
        combined = f"{topic} {content}"

        # Skip very short chunks (TOC lines, titles, etc.)
        if len(content) < 40:
            continue

        # Count keyword hits
        kw_hits = sum(1 for kw in content_keywords if kw in combined) if content_keywords else 0
        if kw_hits == 0:
            continue

        # Base score from keyword density
        score = kw_hits

        # Boost preferred doc sources heavily (+5 per match)
        for pref in preferred_docs:
            if pref.lower() in doc_name or pref.lower() in chunk_id:
                score += 5
                break

        # Boost topic relevance (+2)
        if content_keywords and any(kw in topic for kw in content_keywords):
            score += 2

        # Penalize very long generic handbook chunks that mention keywords tangentially
        if doc_name == "student handbook.pdf" and kw_hits <= 1 and len(content) > 500:
            score -= 1

        text_preview = (chunk.get("content") or "")[:500]
        scored.append((score, {
            "source": "documents",
            "chunk_id": chunk.get("chunk_id", ""),
            "topic": chunk.get("Topic") or chunk.get("topic") or chunk.get("section_title", ""),
            "doc_name": chunk.get("doc_name", ""),
            "content_preview": text_preview,
        }))

    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:12]]


def _match_tuition(query, tuition_data):
    q = query.lower()
    matched = []

    level_filter = None
    if "graduate" in q and "undergraduate" not in q:
        level_filter = "graduate"
    elif "undergraduate" in q:
        level_filter = "undergrad"

    school_filter = None
    if "mies" in q or "main campus" in q:
        school_filter = "Mies"
    elif "kent" in q or "chicago-kent" in q:
        school_filter = "Chicago-Kent"
    elif "design" in q or "mdes" in q:
        school_filter = "Institute of Design"
    elif "stuart" in q or "business" in q:
        school_filter = "Stuart School of Business"

    fee_keywords = []
    if "tuition" in q:
        fee_keywords.append("tuition")
    if "refund" in q:
        fee_keywords.append("refund")
    if "fee" in q and "tuition" not in q:
        fee_keywords.append("fee")

    for row in tuition_data:
        level = (row.get("level") or "").lower()
        school = (row.get("school") or "").lower()
        fee_name = (row.get("fee_name") or "").lower()
        chunk_text = (row.get("chunk_text") or "").lower()

        if level_filter and level_filter not in level and level != "all":
            continue
        if school_filter and school_filter.lower() not in school:
            continue
        if fee_keywords and not any(fk in fee_name or fk in chunk_text for fk in fee_keywords):
            continue

        matched.append({
            "source": "tuition",
            "doc_id": row.get("doc_id", ""),
            "chunk_text": row.get("chunk_text", ""),
            "school": row.get("school", ""),
            "level": row.get("level", ""),
            "fee_name": row.get("fee_name", ""),
            "amount_value": row.get("amount_value"),
            "unit": row.get("unit", ""),
            "academic_year": row.get("academic_year", ""),
        })

    return matched[:10]


# ── Section → domain mapping ──────────────────────────────────────────────────

SECTION_DOMAINS = {
    "Calendar": ["calendar"],
    "Contact": ["contacts"],
    "Policy": ["documents"],
    "Registration": ["documents", "calendar", "contacts"],
    "Transcripts": ["documents", "contacts"],
    "Tuition": ["tuition", "documents"],
    "Financial Aid": ["documents", "tuition"],
    "Transfer Credit": ["documents"],
    "Multi-topic": ["documents", "calendar", "tuition", "contacts"],
}


def find_relevant_chunks(section, query, all_data):
    domains = SECTION_DOMAINS.get(section, ["documents"])
    chunks = []
    for domain in domains:
        if domain == "calendar":
            chunks.extend(_match_calendar(query, all_data["calendar"]))
        elif domain == "contacts":
            chunks.extend(_match_contacts(query, all_data["contacts"]))
        elif domain == "documents":
            chunks.extend(_match_documents(query, all_data["documents"]))
        elif domain == "tuition":
            chunks.extend(_match_tuition(query, all_data["tuition"]))
    return chunks


# ── Gold answer generation via Groq ────────────────────────────────────────────

def generate_gold_answer(query, chunks):
    if not GROQ_API_KEY:
        return "[NO GROQ KEY - fill manually]"

    chunks_text = ""
    for i, c in enumerate(chunks, 1):
        if c["source"] == "calendar":
            chunks_text += (
                f"\n[Chunk {i} - Calendar] Term: {c['term']}, "
                f"Event: {c['event_name']}, "
                f"Start: {c['start_date']}, End: {c['end_date']}"
            )
        elif c["source"] == "contacts":
            parts = [f"Name: {c['name']}"]
            for k in ("department", "phone", "fax", "email", "building", "address"):
                if c.get(k):
                    parts.append(f"{k.title()}: {c[k]}")
            chunks_text += f"\n[Chunk {i} - Contacts] {', '.join(parts)}"
        elif c["source"] == "documents":
            preview = c.get('content_preview', '')[:600]
            chunks_text += (
                f"\n[Chunk {i} - Documents] Topic: {c.get('topic','')}, "
                f"Doc: {c.get('doc_name','')}\n{preview}"
            )
        elif c["source"] == "tuition":
            chunks_text += f"\n[Chunk {i} - Tuition] {c.get('chunk_text','')}"

    if not chunks_text.strip():
        return "[No relevant chunks found in source data]"

    prompt = (
        "You are generating a gold-standard reference answer for evaluating a "
        "university chatbot (Illinois Institute of Technology).\n\n"
        "Using ONLY the data chunks below, write a concise, factual answer to "
        "the question. RULES:\n"
        "- Give a DIRECT answer (yes/no/specific value) before any explanation\n"
        "- Include specific dates, dollar amounts, names, phone numbers, or "
        "other concrete details from the data\n"
        "- If multiple chunks address the question, synthesize them into one "
        "coherent answer\n"
        "- Do NOT say 'the data does not state' if any chunk contains the "
        "answer — look carefully across all chunks\n"
        "- Only say information is missing if it truly is not in any chunk\n\n"
        f"Question: {query}\n\n"
        f"Source Data:{chunks_text}\n\n"
        "Write the gold answer (2-4 sentences, factual, specific):"
    )

    # Try Groq first (two models), then Theta as final fallback
    if GROQ_API_KEY:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        for model in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=300,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e):
                    print(f"    [Rate limited on {model}, trying next...]")
                    continue
                return f"[ERROR: {e}]"

    # Theta Cloud fallback
    if THETA_API_KEY and THETA_API_URL:
        print("    [Using Theta Cloud]")
        try:
            theta_url = THETA_API_URL.strip().strip('"').strip("'")
            theta_key = THETA_API_KEY.strip()
            messages = [{"role": "user", "content": prompt}]
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
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.text.strip()
            data = json.loads(raw)
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or choices[0].get("delta", {})
                answer = (msg.get("content") or "").strip()
                if answer:
                    return answer
        except Exception as e:
            return f"[ERROR Theta: {e}]"

    return "[ERROR: all LLMs unavailable]"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    from test_sections import questions

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    section_filter = args[0] if args else None
    force_all = "--force" in sys.argv

    print("Loading data files...")
    all_data = {
        "calendar": load_calendar(),
        "contacts": load_contacts(),
        "documents": load_documents(),
        "tuition": load_tuition(),
    }
    print(f"  Calendar:  {len(all_data['calendar'])} events")
    print(f"  Contacts:  {len(all_data['contacts'])} entries")
    print(f"  Documents: {len(all_data['documents'])} chunks")
    print(f"  Tuition:   {len(all_data['tuition'])} rows")

    existing = []
    if OUTPUT_PATH.exists() and not force_all:
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)
    existing_map = {(e["section"], e["query"]): e for e in existing}

    results = list(existing) if not force_all else []
    generated_count = 0

    for section, qs in questions.items():
        if section_filter and section != section_filter:
            continue

        print(f"\n--- {section} ({len(qs)} questions) ---")
        for q in qs:
            if not force_all:
                cached = existing_map.get((section, q))
                if cached:
                    ans = cached.get("gold_answer", "")
                    if ans and not ans.startswith("[ERROR"):
                        print(f"  [cached] {q[:60]}...")
                        continue
                    print(f"  [retry-error] {q[:60]}...")

            chunks = find_relevant_chunks(section, q, all_data)
            print(f"  [{len(chunks)} chunks] {q[:60]}...")

            gold_answer = generate_gold_answer(q, chunks)

            entry = {
                "section": section,
                "query": q,
                "relevant_chunks": chunks,
                "gold_answer": gold_answer,
            }

            if (section, q) in existing_map:
                results = [
                    r for r in results
                    if not (r["section"] == section and r["query"] == q)
                ]
            results.append(entry)
            existing_map[(section, q)] = entry
            generated_count += 1

            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            time.sleep(2)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Generated {generated_count} gold answers.")
    print(f"Total entries in {OUTPUT_PATH}: {len(results)}")


if __name__ == "__main__":
    main()
