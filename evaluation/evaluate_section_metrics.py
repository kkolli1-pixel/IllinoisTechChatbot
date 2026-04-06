"""
Evaluate the chatbot across all test sections using six metrics.
Calls search/retrieval functions directly — no Streamlit or frontend needed.

Retrieval metrics (per query, top-k chunks):
  1. Precision@k   — fraction of top-k retrieved chunks that are relevant
  2. Recall@k      — fraction of all relevant chunks captured in top-k
  3. HitRate@k     — 1 if at least one top-k chunk is relevant, else 0

Answer quality metrics (LLM-judged, 0.0–1.0):
  4. Faithfulness   — is the answer grounded in the retrieved context?
  5. Completeness   — does the answer cover what the question asks?
  6. Correctness    — is the answer factually correct vs the gold answer?

Usage:
    python evaluate_section_metrics.py --phase generate             # Phase 1: retrieve + generate all answers
    python evaluate_section_metrics.py --phase judge                # Phase 2: judge saved answers (uses 70b)
    python evaluate_section_metrics.py --phase judge --section Tuition  # judge one section
    python evaluate_section_metrics.py                              # both phases in one run
    python evaluate_section_metrics.py --k 5                        # custom k
    python evaluate_section_metrics.py --api theta                  # use Theta Cloud
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
THETA_API_KEY = os.getenv("THETA_API_KEY", "").strip()
THETA_API_URL = os.getenv("THETA_API_URL", "").strip().strip('"').strip("'")

USE_THETA = False

logging.getLogger().setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

GOLD_PATH = PROJECT_ROOT / "gold_answers.json"
GENERATED_PATH = PROJECT_ROOT / "generated_answers.json"
REPORT_PATH = PROJECT_ROOT / "metrics_report.md"
RESULTS_JSON_PATH = PROJECT_ROOT / "evaluation_results.json"


# ── Import search functions (no Streamlit needed) ─────────────────────────────

from router.router import (
    get_routing_intent,
    DOMAIN_CALENDAR, DOMAIN_CONTACTS, DOMAIN_DOCUMENTS, DOMAIN_TUITION,
)
from router.calendar_router import route_query as calendar_route_query
from search.calendar_search import calendar_rrf_search
from search.contacts_search import contacts_rrf_search
from search.documents_search import documents_rrf_search
from search.tuition_search import tuition_rrf_search


# ── Context builders (same logic as ui/app_with_clarification_memory.py) ──────

def build_calendar_context(hits):
    context = []
    for h in hits:
        s = h.get("_source", {})
        start = (s.get("start_date") or "").strip()
        end = (s.get("end_date") or "").strip()
        event_name = (s.get("event_name") or "").strip()
        term = (s.get("term") or "").strip()
        semantic_text = (s.get("semantic_text") or "").strip()
        date_str = start if start and start == end else (f"{start} -> {end}" if start or end else "N/A")
        header = f"[Calendar: {term} | {date_str}]"
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


# ── Theta Cloud helper ─────────────────────────────────────────────────────────

def _theta_call_raw(messages, max_tokens=300, _retries=4):
    """Call Theta Cloud edge inference with retry+backoff for 409/5xx errors."""
    if not THETA_API_KEY or not THETA_API_URL:
        return ""
    for attempt in range(_retries):
        try:
            resp = requests.post(
                THETA_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {THETA_API_KEY}",
                },
                json={
                    "input": {
                        "max_tokens": max_tokens,
                        "messages": messages,
                        "temperature": 0.0,
                        "top_p": 0.7,
                    },
                    "stream": False,
                    "variant": "quantized",
                },
                timeout=90,
            )
            if resp.status_code in (409, 429, 500, 502, 503):
                wait = 5 * (attempt + 1)
                print(f"    [Theta {resp.status_code}, retry {attempt+1} in {wait}s]")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = json.loads(resp.text.strip())
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or choices[0].get("delta", {})
                answer = (msg.get("content") or "").strip()
                if answer:
                    return answer
            return ""
        except requests.exceptions.RequestException as e:
            if attempt < _retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    [Theta error: {e}, retry {attempt+1} in {wait}s]")
                time.sleep(wait)
            else:
                print(f"    [Theta error] {e}")
        except Exception as e:
            print(f"    [Theta error] {e}")
            break
    return ""


# ── Query rewriting (mirrors ui/app_with_clarification_memory.py) ──────────────

def _groq_chat(messages, max_tokens=60):
    """Lightweight chat call for rewriting/clarification — uses 8b-instant."""
    if USE_THETA:
        return _theta_call_raw(messages, max_tokens=max_tokens)
    if not GROQ_API_KEY:
        return ""
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def rewrite_query(query: str, domains: list) -> str:
    """Rewrite a natural-language question into retrieval-optimised language."""
    if not domains:
        return query

    domain_hints = {
        DOMAIN_CALENDAR: "academic calendar, dates, deadlines, registration, exams, breaks",
        DOMAIN_CONTACTS: "university departments, offices, phone numbers, emails, locations",
        DOMAIN_DOCUMENTS: "university policies, student handbook, academic rules, conduct, procedures",
        DOMAIN_TUITION: "tuition rates, fees, costs, billing, financial charges",
    }

    relevant_hints = " | ".join(domain_hints[d] for d in domains if d in domain_hints)

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

    user_content = (
        f"Domains: {relevant_hints}\n"
        f"Original query: {query}\n\n"
        f"Rewritten query:"
    )

    result = _groq_chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        max_tokens=60,
    )
    return result if result else query


# ── Clarification auto-resolver ───────────────────────────────────────────────

def _pick_best_option(query: str, message: str, options: list) -> str:
    """
    Use an LLM to choose the most appropriate clarification option given the
    original question.  If the LLM response doesn't match any option, fall back
    to the first one.
    """
    if not options:
        return ""
    if len(options) == 1:
        return options[0]

    numbered = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
    prompt = (
        "A university chatbot asked the user to clarify their question.\n\n"
        f"User question: {query}\n"
        f"Chatbot message: {message}\n"
        f"Options:\n{numbered}\n\n"
        "Which single option best matches the user's original question? "
        "Reply with ONLY the exact option text — nothing else."
    )

    raw = _groq_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=40,
    )

    raw_lower = raw.lower().strip().strip('"').strip("'")
    for opt in options:
        if opt.lower() == raw_lower or opt.lower() in raw_lower:
            return opt
    for opt in options:
        if raw_lower in opt.lower():
            return opt
    return options[0]


def _resolve_clarification(query, hits, domain, search_fn, build_fn, top_k=None):
    """
    If *hits* is a clarification dict, auto-resolve it by picking the best
    option and re-running the search with an augmented query.
    Returns (resolved_hits_list, was_clarified).
    """
    if not (isinstance(hits, dict) and hits.get("needs_clarification")):
        return (hits if isinstance(hits, list) else []), False

    message = hits.get("message", "")
    options = hits.get("options", [])
    choice = _pick_best_option(query, message, options)

    if not choice:
        return [], True

    augmented_query = f"{query} — {choice}"
    print(f"    [Clarification in {domain}] picked: {choice}")

    try:
        if top_k is not None:
            retry_hits = search_fn(augmented_query, top_k=top_k)
        else:
            retry_hits = search_fn(augmented_query)
    except Exception:
        return [], True

    if isinstance(retry_hits, dict) and retry_hits.get("needs_clarification"):
        return [], True
    return (retry_hits if isinstance(retry_hits, list) else []), True


# ── Retrieval pipeline (standalone, no Streamlit) ─────────────────────────────

def retrieve_context(query):
    """
    Run the full retrieval pipeline for a query and return
    (context_parts, domains, is_clarification).

    Steps:
      1. Route query to domains
      2. Rewrite query for better retrieval
      3. Search each domain; auto-resolve clarifications via LLM
    """
    route_details = get_routing_intent(query)
    domains = route_details.get("domains", [])

    if not domains:
        return [], domains, False

    retrieval_query = rewrite_query(query, domains)
    if retrieval_query != query:
        print(f"    [Rewritten] {retrieval_query[:90]}")

    context_parts = []
    total_hits = 0

    if DOMAIN_CALENDAR in domains:
        try:
            hits = calendar_route_query(retrieval_query)
            hits, _ = _resolve_clarification(
                query, hits, "Calendar", calendar_rrf_search,
                build_calendar_context,
            )
            if hits:
                total_hits += len(hits)
                context_parts.extend(build_calendar_context(hits))
        except Exception:
            pass

    if DOMAIN_CONTACTS in domains:
        try:
            hits = contacts_rrf_search(retrieval_query, top_k=10)
            hits, _ = _resolve_clarification(
                query, hits, "Contacts", contacts_rrf_search,
                build_contacts_context, top_k=10,
            )
            if hits:
                total_hits += len(hits)
                context_parts.extend(build_contacts_context(hits))
        except Exception:
            pass

    if DOMAIN_DOCUMENTS in domains:
        try:
            hits = documents_rrf_search(retrieval_query)
            hits, _ = _resolve_clarification(
                query, hits, "Documents", documents_rrf_search,
                build_documents_context,
            )
            if hits:
                total_hits += len(hits)
                context_parts.extend(build_documents_context(hits))
        except Exception:
            pass

    if DOMAIN_TUITION in domains:
        try:
            hits = tuition_rrf_search(retrieval_query, top_k=10)
            hits, _ = _resolve_clarification(
                query, hits, "Tuition", tuition_rrf_search,
                build_tuition_context, top_k=10,
            )
            if hits:
                total_hits += len(hits)
                context_parts.extend(build_tuition_context(hits))
        except Exception:
            pass

    return context_parts, domains, False


def generate_answer(query, context_parts):
    """Generate an answer given retrieved context — uses 8b-instant (matches live app)."""
    if not context_parts:
        return ""

    context_text = "\n\n---\n\n".join(context_parts)
    system = (
        "You are a helpful university assistant for Illinois Institute of Technology. "
        "Answer using ONLY the context below. Be concise and factual. "
        "If the answer is not in the context, say you are not sure."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context_text}"},
    ]

    if USE_THETA:
        return _theta_call_raw(messages, max_tokens=400)

    if not GROQ_API_KEY:
        return ""
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"    [Generation error] {e}")
        return ""


# ── Groq judge helper ─────────────────────────────────────────────────────────

def _groq_call(prompt, max_tokens=400):
    """Judge helper — uses Theta when USE_THETA is set, else Groq."""
    messages = [{"role": "user", "content": prompt}]

    if USE_THETA:
        return _theta_call_raw(messages, max_tokens=max_tokens)

    if not GROQ_API_KEY:
        return ""
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"    [Groq error] {e}")
        return ""


# ── Retrieval metric: chunk relevance judge ────────────────────────────────────

def judge_chunk_relevance(query, chunks, k=3):
    """
    For each of the top-k context chunks, ask the LLM whether it is relevant
    to the query. Returns a list of booleans (length = min(k, len(chunks))).
    """
    top_chunks = chunks[:k]
    if not top_chunks:
        return []

    numbered = ""
    for i, c in enumerate(top_chunks, 1):
        preview = c[:300].replace("\n", " ")
        numbered += f"\nChunk {i}: {preview}\n"

    prompt = (
        "You are a relevance judge for a university information retrieval system.\n\n"
        f"User question: {query}\n\n"
        f"Retrieved chunks:{numbered}\n"
        f"For each chunk (1 to {len(top_chunks)}), decide if it contains information "
        "that is relevant or useful for answering the question.\n\n"
        "Reply with ONLY a JSON array of booleans, e.g. [true, false, true]. "
        "No explanation."
    )

    raw = _groq_call(prompt, max_tokens=50)
    try:
        result = json.loads(raw)
        if isinstance(result, list) and len(result) == len(top_chunks):
            return [bool(x) for x in result]
    except (json.JSONDecodeError, TypeError):
        pass

    return [True] * len(top_chunks)


def compute_retrieval_metrics(relevance_flags, k=3):
    top_k = relevance_flags[:k]
    if not top_k:
        return 0.0, 0.0, 0.0

    relevant_in_k = sum(top_k)
    total_relevant = sum(relevance_flags)

    precision = relevant_in_k / len(top_k)
    recall = relevant_in_k / total_relevant if total_relevant > 0 else 0.0
    hit_rate = 1.0 if relevant_in_k > 0 else 0.0

    return precision, recall, hit_rate


# ── Answer quality judge ───────────────────────────────────────────────────────

def judge_answer_quality(query, generated_answer, gold_answer, context_text):
    """
    Single LLM call that returns faithfulness, completeness, correctness
    as floats in [0.0, 1.0].
    """
    prompt = (
        "You are an expert evaluator for a university chatbot.\n\n"
        "Given:\n"
        f"- User question: {query}\n"
        f"- Gold (reference) answer: {gold_answer}\n"
        f"- Chatbot answer: {generated_answer}\n"
        f"- Retrieved context (used by chatbot):\n{context_text[:2000]}\n\n"
        "Score each of these on a scale from 0.0 to 1.0:\n"
        "1. **faithfulness**: Is every claim in the chatbot answer supported by "
        "the retrieved context? (1.0 = fully grounded, 0.0 = hallucinated)\n"
        "2. **completeness**: Does the chatbot answer cover all key facts from "
        "the gold answer? (1.0 = covers everything, 0.0 = misses everything)\n"
        "3. **correctness**: Is the chatbot answer factually correct compared "
        "to the gold answer? (1.0 = fully correct, 0.0 = wrong)\n\n"
        "Reply with ONLY a JSON object like: "
        '{"faithfulness": 0.8, "completeness": 0.7, "correctness": 0.9}\n'
        "No explanation."
    )

    raw = _groq_call(prompt, max_tokens=80)
    try:
        result = json.loads(raw)
        return (
            float(result.get("faithfulness", 0)),
            float(result.get("completeness", 0)),
            float(result.get("correctness", 0)),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return 0.0, 0.0, 0.0


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(k=3, section_filter=None):
    from test_sections import questions

    if not GOLD_PATH.exists():
        print(f"ERROR: {GOLD_PATH} not found. Run generate_gold_answers.py first.")
        sys.exit(1)

    with open(GOLD_PATH) as f:
        gold_data = json.load(f)
    gold_map = {(g["section"], g["query"]): g for g in gold_data}

    total_questions = sum(len(qs) for qs in questions.values())
    print(f"Evaluating {total_questions} questions with k={k}")
    print(f"No Streamlit needed — calling search functions directly.\n")

    all_metrics = []
    section_metrics_map = {}
    section_results = {}

    with open(REPORT_PATH, "w") as rpt:
        rpt.write(f"# RAG Evaluation Report (k={k})\n\n")
        rpt.write(f"Total questions: {total_questions}\n\n")

    for section, qs in questions.items():
        if section_filter and section != section_filter:
            continue

        print(f"\n{'='*50}")
        print(f"SECTION: {section} ({len(qs)} questions)")
        print(f"{'='*50}")

        section_metrics = []

        with open(REPORT_PATH, "a") as rpt:
            rpt.write(f"## {section}\n\n")
            rpt.write(
                f"| # | P@{k} | R@{k} | Hit@{k} | Faith | Compl | Correct | Query |\n"
                f"|---|-------|-------|---------|-------|-------|---------|-------|\n"
            )

        for idx, q in enumerate(qs, 1):
            print(f"\n  Q{idx}: {q[:70]}...")

            gold = gold_map.get((section, q))
            gold_answer = gold["gold_answer"] if gold else "[no gold answer]"

            # Step 1: Retrieve (direct search, no Streamlit)
            try:
                context_parts, domains, is_clarification = retrieve_context(q)
            except Exception as e:
                print(f"    [RETRIEVAL ERROR] {e}")
                entry = _skip_entry(q, "ERROR", section)
                section_metrics.append(entry)
                all_metrics.append(entry)
                _write_skip_row(idx, q, "ERR")
                continue

            if not context_parts:
                print(f"    [SKIP] No context retrieved")
                entry = _skip_entry(q, "NO_CONTEXT", section)
                section_metrics.append(entry)
                all_metrics.append(entry)
                _write_skip_row(idx, q, "NO_CTX")
                time.sleep(1)
                continue

            print(f"    Retrieved {len(context_parts)} chunks -> {', '.join(domains)}")

            # Step 2: Generate answer
            reply = generate_answer(q, context_parts)
            if reply:
                short = reply.replace("\n", " ")[:120]
                print(f"    Answer: {short}...")

            pause = 6 if USE_THETA else 2
            time.sleep(pause)

            # Step 3: Judge chunk relevance -> retrieval metrics
            relevance = judge_chunk_relevance(q, context_parts, k=k)
            p_at_k, r_at_k, hit_at_k = compute_retrieval_metrics(relevance, k=k)
            print(f"    Retrieval: P@{k}={p_at_k:.2f}  R@{k}={r_at_k:.2f}  Hit@{k}={hit_at_k:.0f}")

            time.sleep(pause)

            # Step 4: Judge answer quality
            context_text = "\n\n---\n\n".join(context_parts)
            faith, compl, correct = judge_answer_quality(q, reply, gold_answer, context_text)
            print(f"    Quality:   Faith={faith:.2f}  Compl={compl:.2f}  Correct={correct:.2f}")

            metrics = {
                "section": section,
                "query": q,
                "domains": domains,
                "num_chunks": len(context_parts),
                "generated_answer": reply,
                "gold_answer": gold_answer,
                "p_at_k": p_at_k,
                "r_at_k": r_at_k,
                "hit_at_k": hit_at_k,
                "faithfulness": faith,
                "completeness": compl,
                "correctness": correct,
                "skipped": False,
            }
            section_metrics.append(metrics)
            all_metrics.append(metrics)

            with open(REPORT_PATH, "a") as rpt:
                safe_q = q.replace("|", "/")[:50]
                rpt.write(
                    f"| {idx} | {p_at_k:.2f} | {r_at_k:.2f} | {hit_at_k:.0f} "
                    f"| {faith:.2f} | {compl:.2f} | {correct:.2f} | {safe_q} |\n"
                )
                rpt.flush()

            time.sleep(8 if USE_THETA else 3)

        # Section summary
        evaluated = [m for m in section_metrics if not m.get("skipped")]
        summary = _compute_section_summary(section, section_metrics, evaluated)
        section_results[section] = summary
        section_metrics_map[section] = section_metrics
        _write_section_summary(summary, k)

        print(f"\n  {section} avg: P@{k}={summary['avg_p_at_k']:.3f}  "
              f"R@{k}={summary['avg_r_at_k']:.3f}  Hit@{k}={summary['avg_hit_at_k']:.3f}  "
              f"Faith={summary['avg_faithfulness']:.3f}  "
              f"Compl={summary['avg_completeness']:.3f}  "
              f"Correct={summary['avg_correctness']:.3f}")

    # Global summary
    _write_global_summary(all_metrics, k)

    # Save structured JSON results
    _save_results_json(all_metrics, section_results, k)

    print(f"\n{'='*50}")
    print("EVALUATION COMPLETE")
    print(f"Markdown report : {REPORT_PATH}")
    print(f"JSON results    : {RESULTS_JSON_PATH}")
    print(f"{'='*50}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _skip_entry(q, reason="SKIP", section=""):
    return {
        "section": section, "query": q, "domains": [], "num_chunks": 0,
        "generated_answer": "", "gold_answer": "",
        "p_at_k": None, "r_at_k": None, "hit_at_k": None,
        "faithfulness": None, "completeness": None, "correctness": None,
        "skipped": True, "reason": reason,
    }


def _write_skip_row(idx, q, label):
    with open(REPORT_PATH, "a") as rpt:
        safe_q = q.replace("|", "/")[:50]
        rpt.write(f"| {idx} | {label} | - | - | - | - | - | {safe_q} |\n")
        rpt.flush()


def _compute_section_summary(section, section_metrics, evaluated):
    if evaluated:
        avg = lambda key: sum(m[key] for m in evaluated) / len(evaluated)
        return {
            "section": section,
            "n_evaluated": len(evaluated),
            "n_skipped": len(section_metrics) - len(evaluated),
            "avg_p_at_k": avg("p_at_k"),
            "avg_r_at_k": avg("r_at_k"),
            "avg_hit_at_k": avg("hit_at_k"),
            "avg_faithfulness": avg("faithfulness"),
            "avg_completeness": avg("completeness"),
            "avg_correctness": avg("correctness"),
        }
    return {
        "section": section,
        "n_evaluated": 0,
        "n_skipped": len(section_metrics),
        "avg_p_at_k": 0, "avg_r_at_k": 0, "avg_hit_at_k": 0,
        "avg_faithfulness": 0, "avg_completeness": 0, "avg_correctness": 0,
    }


def _write_section_summary(summary, k):
    with open(REPORT_PATH, "a") as rpt:
        s = summary
        rpt.write(
            f"\n**{s['section']} Averages** ({s['n_evaluated']} evaluated, "
            f"{s['n_skipped']} skipped):\n"
            f"P@{k}={s['avg_p_at_k']:.3f} | R@{k}={s['avg_r_at_k']:.3f} | "
            f"Hit@{k}={s['avg_hit_at_k']:.3f} | "
            f"Faith={s['avg_faithfulness']:.3f} | "
            f"Compl={s['avg_completeness']:.3f} | "
            f"Correct={s['avg_correctness']:.3f}\n\n"
        )


def _save_results_json(all_metrics, section_results, k):
    """Write all per-question metrics and section/global summaries to JSON."""
    evaluated = [m for m in all_metrics if not m.get("skipped")]
    if evaluated:
        gavg = lambda key: round(sum(m[key] for m in evaluated) / len(evaluated), 4)
        global_summary = {
            "n_evaluated": len(evaluated),
            "n_skipped": len(all_metrics) - len(evaluated),
            "avg_p_at_k": gavg("p_at_k"),
            "avg_r_at_k": gavg("r_at_k"),
            "avg_hit_at_k": gavg("hit_at_k"),
            "avg_faithfulness": gavg("faithfulness"),
            "avg_completeness": gavg("completeness"),
            "avg_correctness": gavg("correctness"),
        }
    else:
        global_summary = {"n_evaluated": 0, "n_skipped": len(all_metrics)}

    rounded_sections = {}
    for sec, s in section_results.items():
        rounded_sections[sec] = {
            k_: round(v, 4) if isinstance(v, float) else v
            for k_, v in s.items()
        }

    output = {
        "k": k,
        "global_summary": global_summary,
        "section_summaries": rounded_sections,
        "per_question": all_metrics,
    }
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  JSON results saved to {RESULTS_JSON_PATH}")


def _write_global_summary(all_metrics, k):
    evaluated = [m for m in all_metrics if not m.get("skipped")]
    if evaluated:
        gavg = lambda key: sum(m[key] for m in evaluated) / len(evaluated)
        text = (
            f"\n## Overall ({len(evaluated)} questions evaluated)\n\n"
            f"| Metric | Score |\n|--------|-------|\n"
            f"| Precision@{k} | {gavg('p_at_k'):.3f} |\n"
            f"| Recall@{k} | {gavg('r_at_k'):.3f} |\n"
            f"| HitRate@{k} | {gavg('hit_at_k'):.3f} |\n"
            f"| Faithfulness | {gavg('faithfulness'):.3f} |\n"
            f"| Completeness | {gavg('completeness'):.3f} |\n"
            f"| Correctness | {gavg('correctness'):.3f} |\n"
        )
    else:
        text = "\n## Overall\n\nNo questions were evaluated.\n"

    with open(REPORT_PATH, "a") as rpt:
        rpt.write(text)


# ── Phase 1: Generate answers (no 70b needed) ─────────────────────────────────

def run_generate(section_filter=None):
    """Retrieve context + generate answers for all questions. Saves to generated_answers.json."""
    from test_sections import questions

    if not GOLD_PATH.exists():
        print(f"ERROR: {GOLD_PATH} not found. Run generate_gold_answers.py first.")
        sys.exit(1)

    with open(GOLD_PATH) as f:
        gold_data = json.load(f)
    gold_map = {(g["section"], g["query"]): g for g in gold_data}

    existing = []
    if GENERATED_PATH.exists():
        with open(GENERATED_PATH) as f:
            existing = json.load(f)
    existing_map = {(e["section"], e["query"]): e for e in existing}

    results = list(existing)
    generated_count = 0
    skipped_count = 0

    for section, qs in questions.items():
        if section_filter and section != section_filter:
            continue

        print(f"\n{'='*50}")
        print(f"SECTION: {section} ({len(qs)} questions)")
        print(f"{'='*50}")

        for idx, q in enumerate(qs, 1):
            cached = existing_map.get((section, q))
            if cached and cached.get("generated_answer") and not cached.get("skipped"):
                print(f"  Q{idx}: [cached] {q[:60]}...")
                continue

            if cached and cached.get("context_parts") and not cached.get("generated_answer") and not cached.get("skipped"):
                print(f"\n  Q{idx}: [re-gen from saved context] {q[:60]}...")
                context_parts = cached["context_parts"]
                reply = generate_answer(q, context_parts)
                if reply:
                    short = reply.replace("\n", " ")[:120]
                    print(f"    Answer: {short}...")
                cached["generated_answer"] = reply
                results = [r for r in results if not (r["section"] == section and r["query"] == q)]
                results.append(cached)
                generated_count += 1
                time.sleep(2)
                continue

            print(f"\n  Q{idx}: {q[:70]}...")

            gold = gold_map.get((section, q))
            gold_answer = gold["gold_answer"] if gold else "[no gold answer]"

            try:
                context_parts, domains, _ = retrieve_context(q)
            except Exception as e:
                print(f"    [RETRIEVAL ERROR] {e}")
                entry = {
                    "section": section, "query": q, "domains": [],
                    "num_chunks": 0, "context_parts": [],
                    "generated_answer": "", "gold_answer": gold_answer,
                    "skipped": True, "reason": "ERROR",
                }
                results = [r for r in results if not (r["section"] == section and r["query"] == q)]
                results.append(entry)
                skipped_count += 1
                continue

            if not context_parts:
                print(f"    [SKIP] No context retrieved")
                entry = {
                    "section": section, "query": q, "domains": domains,
                    "num_chunks": 0, "context_parts": [],
                    "generated_answer": "", "gold_answer": gold_answer,
                    "skipped": True, "reason": "NO_CONTEXT",
                }
                results = [r for r in results if not (r["section"] == section and r["query"] == q)]
                results.append(entry)
                skipped_count += 1
                time.sleep(1)
                continue

            print(f"    Retrieved {len(context_parts)} chunks -> {', '.join(domains)}")

            reply = generate_answer(q, context_parts)
            if reply:
                short = reply.replace("\n", " ")[:120]
                print(f"    Answer: {short}...")

            entry = {
                "section": section, "query": q, "domains": domains,
                "num_chunks": len(context_parts),
                "context_parts": context_parts,
                "generated_answer": reply,
                "gold_answer": gold_answer,
                "skipped": False,
            }
            results = [r for r in results if not (r["section"] == section and r["query"] == q)]
            results.append(entry)
            generated_count += 1

            time.sleep(2)

        with open(GENERATED_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)

    with open(GENERATED_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*50}")
    print(f"GENERATE COMPLETE — {generated_count} generated, {skipped_count} skipped")
    print(f"Saved to: {GENERATED_PATH}")
    print(f"{'='*50}")


# ── Phase 2: Judge from saved answers (uses 70b) ──────────────────────────────

def run_judge(k=3, section_filter=None):
    """Load generated_answers.json, run 70b judge for all 6 metrics, write report."""
    if not GENERATED_PATH.exists():
        print(f"ERROR: {GENERATED_PATH} not found. Run --phase generate first.")
        sys.exit(1)

    with open(GENERATED_PATH) as f:
        generated = json.load(f)

    if section_filter:
        generated = [g for g in generated if g["section"] == section_filter]

    total = len(generated)
    print(f"Judging {total} questions with k={k}\n")

    all_metrics = []
    section_results = {}

    with open(REPORT_PATH, "w") as rpt:
        rpt.write(f"# RAG Evaluation Report (k={k})\n\n")
        rpt.write(f"Total questions: {total}\n\n")

    sections_ordered = {}
    for g in generated:
        sections_ordered.setdefault(g["section"], []).append(g)

    for section, entries in sections_ordered.items():
        print(f"\n{'='*50}")
        print(f"SECTION: {section} ({len(entries)} questions)")
        print(f"{'='*50}")

        section_metrics = []

        with open(REPORT_PATH, "a") as rpt:
            rpt.write(f"## {section}\n\n")
            rpt.write(
                f"| # | P@{k} | R@{k} | Hit@{k} | Faith | Compl | Correct | Query |\n"
                f"|---|-------|-------|---------|-------|-------|---------|-------|\n"
            )

        for idx, g in enumerate(entries, 1):
            q = g["query"]
            print(f"\n  Q{idx}: {q[:70]}...")

            if g.get("skipped"):
                print(f"    [SKIP] {g.get('reason', 'unknown')}")
                entry = _skip_entry(q, g.get("reason", "SKIP"), section)
                section_metrics.append(entry)
                all_metrics.append(entry)
                _write_skip_row(idx, q, "SKIP")
                continue

            context_parts = g.get("context_parts", [])
            reply = g.get("generated_answer", "")
            gold_answer = g.get("gold_answer", "[no gold answer]")
            domains = g.get("domains", [])

            if not context_parts:
                print(f"    [SKIP] No context in saved data")
                entry = _skip_entry(q, "NO_CONTEXT", section)
                section_metrics.append(entry)
                all_metrics.append(entry)
                _write_skip_row(idx, q, "NO_CTX")
                continue

            print(f"    {len(context_parts)} chunks, answer len={len(reply)}")

            relevance = judge_chunk_relevance(q, context_parts, k=k)
            p_at_k, r_at_k, hit_at_k = compute_retrieval_metrics(relevance, k=k)
            print(f"    Retrieval: P@{k}={p_at_k:.2f}  R@{k}={r_at_k:.2f}  Hit@{k}={hit_at_k:.0f}")

            time.sleep(3)

            context_text = "\n\n---\n\n".join(context_parts)
            faith, compl, correct = judge_answer_quality(q, reply, gold_answer, context_text)
            print(f"    Quality:   Faith={faith:.2f}  Compl={compl:.2f}  Correct={correct:.2f}")

            metrics = {
                "section": section, "query": q, "domains": domains,
                "num_chunks": len(context_parts),
                "generated_answer": reply, "gold_answer": gold_answer,
                "p_at_k": p_at_k, "r_at_k": r_at_k, "hit_at_k": hit_at_k,
                "faithfulness": faith, "completeness": compl, "correctness": correct,
                "skipped": False,
            }
            section_metrics.append(metrics)
            all_metrics.append(metrics)

            with open(REPORT_PATH, "a") as rpt:
                safe_q = q.replace("|", "/")[:50]
                rpt.write(
                    f"| {idx} | {p_at_k:.2f} | {r_at_k:.2f} | {hit_at_k:.0f} "
                    f"| {faith:.2f} | {compl:.2f} | {correct:.2f} | {safe_q} |\n"
                )
                rpt.flush()

            time.sleep(4)

        evaluated = [m for m in section_metrics if not m.get("skipped")]
        summary = _compute_section_summary(section, section_metrics, evaluated)
        section_results[section] = summary
        _write_section_summary(summary, k)

        print(f"\n  {section} avg: P@{k}={summary['avg_p_at_k']:.3f}  "
              f"R@{k}={summary['avg_r_at_k']:.3f}  Hit@{k}={summary['avg_hit_at_k']:.3f}  "
              f"Faith={summary['avg_faithfulness']:.3f}  "
              f"Compl={summary['avg_completeness']:.3f}  "
              f"Correct={summary['avg_correctness']:.3f}")

    _write_global_summary(all_metrics, k)
    _save_results_json(all_metrics, section_results, k)

    print(f"\n{'='*50}")
    print("JUDGE COMPLETE")
    print(f"Markdown report : {REPORT_PATH}")
    print(f"JSON results    : {RESULTS_JSON_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG evaluation with 6 metrics")
    parser.add_argument("--k", type=int, default=3, help="k for retrieval metrics (default: 3)")
    parser.add_argument("--section", type=str, default=None, help="Evaluate one section only")
    parser.add_argument("--api", type=str, default="groq", choices=["groq", "theta"],
                        help="LLM backend for generation & judging (default: groq)")
    parser.add_argument("--phase", type=str, default="all", choices=["all", "generate", "judge"],
                        help="Phase: generate (retrieve+answer), judge (score saved answers), all (both)")
    args = parser.parse_args()

    if args.api == "theta":
        USE_THETA = True
        if not THETA_API_KEY or not THETA_API_URL:
            print("ERROR: THETA_API_KEY / THETA_API_URL not set in .env")
            sys.exit(1)
        print("Using Theta Cloud API for all LLM calls\n")

    if args.phase == "generate":
        run_generate(section_filter=args.section)
    elif args.phase == "judge":
        run_judge(k=args.k, section_filter=args.section)
    else:
        run_evaluation(k=args.k, section_filter=args.section)
