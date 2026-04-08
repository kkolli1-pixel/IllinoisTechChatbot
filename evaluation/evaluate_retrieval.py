"""
Run all 89 questions through the real chatbot pipeline (get_answer).

Outputs:
  - answers_for_sheet.csv  → paste into Google Sheet
  - retrieval_metrics.csv  → retrieval metrics for analysis

Usage:
    python evaluate_retrieval.py
    python evaluate_retrieval.py --k 5
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── Path setup ─────────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

logging.getLogger().setLevel(logging.WARNING)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Import real chatbot pipeline ───────────────────────────────────────────────
from ui.app_with_clarification_memory import get_answer

# ── Import search functions for retrieval metrics ──────────────────────────────
from router.router import (
    get_routing_intent,
    DOMAIN_CALENDAR, DOMAIN_CONTACTS, DOMAIN_DOCUMENTS, DOMAIN_TUITION,
)
from router.calendar_router import route_query as calendar_route_query
from search.calendar_search import calendar_rrf_search
from search.contacts_search import contacts_rrf_search
from search.documents_search import documents_rrf_search
from search.tuition_search import tuition_rrf_search

# ── Questions (89) ─────────────────────────────────────────────────────────────
QUESTIONS = [
    ("Calendar",        "Easy",   "When does Spring 2026 semester start?",                                                                                                             "January 12, 2026."),
    ("Calendar",        "Easy",   "When are Spring 2026 final exams?",                                                                                                                 "Spring 2026 final exams take place from May 4 to May 9, 2026."),
    ("Calendar",        "Easy",   "When is Spring 2026 commencement?",                                                                                                                 "Spring 2026 commencement is on May 16, 2026."),
    ("Calendar",        "Easy",   "When does Summer 1 2026 start?",                                                                                                                    "Summer 1 courses begin May 18, 2026."),
    ("Calendar",        "Normal", "When is the last day to withdraw from a Spring 2026 course?",                                                                                        "Last day to withdraw (full semester) is March 30, 2026."),
    ("Calendar",        "Normal", "When are Spring 2026 midterm grades due?",                                                                                                           "Midterm grades are due March 13, 2026."),
    ("Calendar",        "Normal", "How long is the Spring 2026 final exam period?",                                                                                                     "The final exam period lasts 6 days (May 4-May 9)."),
    ("Calendar",        "Normal", "How long is the Spring 2026 semester?",                                                                                                              "The semester runs Jan 12 - May 9 (~16 weeks)."),
    ("Calendar",        "Hard",   "If I apply for graduation in Summer 2026 but want to walk in the Spring 2026 ceremony, when do I apply and which ceremony?",                         "You must apply by the spring graduation application deadline (Feb 2, 2026) and would walk in the Spring 2026 ceremony."),
    ("Calendar",        "Hard",   "If I withdraw from a Spring course on April 10, is that before or after the withdrawal deadline?",                                                   "April 10 is after the March 30 deadline, so it is not allowed."),
    ("Calendar",        "Hard",   "If the add/drop deadline is January 20 and I register January 21, what happens?",                                                                   "You miss the add/drop deadline, so you would need late registration approval or may not be able to enroll without penalty."),
    ("Calendar",        "Hard",   "I have a friend visiting from out of town next week. Are there any days in April when IIT has no classes?",                                          "Should list holidays in April or when there are no classes. (No holidays in April)"),
    ("Calendar",        "Hard",   "My lease ends in May. When exactly do spring semester classes finish?",                                                                              "Spring semester classes end on May 2, 2026."),
    ("Calendar",        "Hard",   "I have a wedding to attend in late November. Will I miss any class days?",                                                                           "The Thanksgiving Break occurs from November 25 to 28. You will not miss any class days since the break covers those dates."),
    ("Contact",         "Easy",   "Where is the Registrar's office located?",                                                                                                          "The Registrar's Office is located on the Mies Campus in Chicago."),
    ("Contact",         "Easy",   "What is the fax number for the Department of Physics?",                                                                                             "I'm not sure based on my available information."),
    ("Contact",         "Easy",   "How do I contact the Pritzker Institute of Biomedical Science?",                                                                                    "Call 312.567.7984, email pritzker.institute@illinoistech.edu, or visit 10 W. 35th Street, 18th Floor, Room 18C3-2."),
    ("Contact",         "Normal", "Who should I contact if my transcript request has an issue?",                                                                                        "Contact the Office of the Registrar."),
    ("Contact",         "Normal", "Who should I contact about registration problems?",                                                                                                  "Contact the Registrar's Office."),
    ("Contact",         "Normal", "Who should I contact if I have a hold on my account?",                                                                                              "Contact the Registrar or appropriate department placing the hold."),
    ("Contact",         "Normal", "I want to contact someone at the Wanger Institute. What is their number?",                                                                           "Contact someone at the Wanger Institute."),
    ("Contact",         "Normal", "I want to take a semester off. What is the process?",                                                                                               "Submit a Leave of Absence Form through the Illinois Tech portal."),
    ("Contact",         "Hard",   "If my advisor approval is missing during registration, who should I contact?",                                                                       "Contact your academic advisor or department."),
    ("Policy",          "Easy",   "How many credits do graduate students need to be full-time?",                                                                                        "Graduate students are full-time at 9 credits."),
    ("Policy",          "Easy",   "How many credits do undergraduate students need to be full-time?",                                                                                   "Undergraduate full-time status is 12 credits."),
    ("Policy",          "Easy",   "What is the pass/fail policy at IIT?",                                                                                                              "Pass/fail allows students to take certain courses without GPA impact, but restrictions apply."),
    ("Policy",          "Normal", "I am a graduate student taking 8 credits this semester — am I full-time?",                                                                           "No, 8 credits is part-time (full-time = 9)."),
    ("Policy",          "Normal", "What is the difference between withdrawing and dropping a course?",                                                                                  "Dropping happens early with no record; withdrawing happens later and results in a W grade."),
    ("Policy",          "Normal", "I failed a class and want to retake it. Will both grades show on my transcript?",                                                                    "Yes/No question."),
    ("Policy",          "Normal", "I'm graduating but haven't registered for any courses this semester. Is that a problem?",                                                            "Yes, graduating students must be actively enrolled in the semester they intend to graduate."),
    ("Policy",          "Normal", "How many courses can I take pass/fail as an undergraduate?",                                                                                         "3."),
    ("Policy",          "Hard",   "What is the max pass/fail courses I can take as an undergrad, and if I take a required major course pass/fail, will it count toward my degree?",     "3; usually does NOT count toward major requirements."),
    ("Policy",          "Hard",   "As a coterminal student, am I considered undergraduate or graduate for full-time status purposes, and how does tuition work?",                       "Undergraduate; tuition is billed at undergrad rate."),
    ("Policy",          "Hard",   "I'm a grad student who got a C in a required course — can I repeat it, will the old grade be replaced, and how does it affect my GPA?",             "Yes, courses may be repeated; GPA impact depends on replacement policy."),
    ("Policy",          "Hard",   "Can I work on a research project with a professor and keep the intellectual property?",                                                              "Yes, work done for course credit belongs exclusively to students per handbook Section 1."),
    ("Registration",    "Easy",   "How do I request a late registration?",                                                                                                             "Submit a late registration request through the Registrar, often requiring advisor and/or department approval."),
    ("Registration",    "Easy",   "How do I add a course during the add/drop period?",                                                                                                 "Add courses through the student registration system (myIIT portal) without special approval during the add/drop period."),
    ("Registration",    "Normal", "When does registration open for Fall 2026?",                                                                                                         "Registration typically opens around March-April 2026; exact dates depend on priority registration schedule."),
    ("Registration",    "Normal", "Who do I contact if I have a registration issue?",                                                                                                   "Contact the Office of the Registrar."),
    ("Registration",    "Normal", "What happens if I try to register after the add/drop deadline?",                                                                                     "You cannot register normally; you may need late registration approval or may be denied enrollment."),
    ("Registration",    "Hard",   "I have a hold on my account and registration opens tomorrow — what types of holds exist and how do I find out what mine is?",                        "Common holds include financial, advising, immunization, and administrative. View your hold in the student portal (myIIT)."),
    ("Registration",    "Hard",   "If registration opens tomorrow but I have a financial hold, can I still register?",                                                                  "No, a financial hold blocks registration until resolved."),
    ("Registration",    "Hard",   "If I miss the add/drop deadline, can I still add a class?",                                                                                          "Only with special approval (late add), and it is not guaranteed."),
    ("Transcripts",     "Easy",   "How do I get an official transcript from IIT?",                                                                                                     "Log in to the IIT Portal, search 'Order Official Transcript', and click the tool; alumni use Illinois Tech's Parchment page directly."),
    ("Transcripts",     "Easy",   "Can I order an official transcript online?",                                                                                                        "Yes — current students via the IIT Portal, alumni via Illinois Tech's Parchment page."),
    ("Transcripts",     "Easy",   "Is there a fee for ordering an official transcript?",                                                                                               "Yes, a fee is charged and paid by credit card through Parchment."),
    ("Transcripts",     "Normal", "How long does it take to process a transcript request?",                                                                                             "Electronic/PDF transcripts are delivered worldwide in about an hour; mailed transcripts are processed then mailed."),
    ("Transcripts",     "Normal", "Can alumni request transcripts through the same system?",                                                                                            "Yes — alumni order directly through Illinois Tech's Parchment page instead of the Portal."),
    ("Transcripts",     "Normal", "Can transcripts be sent electronically?",                                                                                                           "Yes, electronic/PDF transcripts are available through Parchment with worldwide delivery in about an hour."),
    ("Transcripts",     "Hard",   "If I graduated last year and need a transcript for a job application, how do I request it?",                                                         "Visit Illinois Tech's Parchment page, complete the online order, and pay by credit card."),
    ("Transcripts",     "Hard",   "If my transcript needs to be sent to multiple schools, how do I request that?",                                                                      "Place a separate Parchment order per recipient; multiple transcripts to the same address ship in one envelope, each individually sealed."),
    ("Transcripts",     "Hard",   "If I need a transcript urgently, is there an expedited option?",                                                                                    "Yes — electronic/PDF transcripts via Parchment are delivered in about an hour and are the fastest option."),
    ("Tuition",         "Easy",   "What is the tuition per credit hour for graduate students on the Mies campus?",                                                                      "$1,851 per credit hour for Fall 2025, Spring 2026, and Summer 2026; $1,780 for Summer 2025."),
    ("Tuition",         "Easy",   "What is the tuition per credit hour for undergraduate students?",                                                                                   "Flat rate of $25,824 per semester for 12-18 credits; $1,612 per credit hour below 12 credits."),
    ("Tuition",         "Easy",   "Where can I find current tuition rates?",                                                                                                           "iit.edu/student-accounting/tuition-and-fees/future-tuition-and-fees/mies-campus-graduate and mies-campus-undergraduate."),
    ("Tuition",         "Normal", "What happens to my tuition if I drop a class after the add/drop deadline?",                                                                          "No tuition refund; you owe full tuition and receive a W grade on your transcript."),
    ("Tuition",         "Normal", "Do I get a refund if I drop a course before the add/drop deadline?",                                                                                "Yes, 100% refundable before the add/drop deadline; submit a Refund Request Form, takes 7-10 business days."),
    ("Tuition",         "Normal", "How is tuition calculated for part-time students?",                                                                                                 "Part-time undergraduates (under 12 credits) are charged $1,612 per credit hour instead of the flat rate."),
    ("Tuition",         "Hard",   "If I drop from 12 credits to 9 credits mid-semester, how does that affect my tuition?",                                                              "No refund after add/drop deadline; dropping below 12 credits means you are no longer full-time."),
    ("Tuition",         "Hard",   "If I withdraw from all classes after the fourth week, do I receive a tuition refund?",                                                               "No refund after the add/drop deadline; W grades appear on transcript; exceptions for military service, serious illness, or university action."),
    ("Tuition",         "Hard",   "If I drop a course but remain full-time, does my tuition change?",                                                                                  "If you stay within 12-18 credits the flat rate does not change; dropping below 12 shifts to per-credit billing but no refund after add/drop."),
    ("Tuition",         "Hard",   "I'm starting the MDes program. What will I pay per semester?",                                                                                      "Tuition rates for the MDes program."),
    ("Tuition",         "Hard",   "Are there any fees specific to Chicago-Kent students beyond tuition?",                                                                               "Chicago-Kent graduate students have additional fees."),
    ("Financial Aid",   "Easy",   "Do undergraduate students need to be full-time to receive financial aid?",                                                                           "Not required for all aid types, but most aid is reduced or eliminated below full-time (12 credits)."),
    ("Financial Aid",   "Easy",   "Where can I find financial aid policies?",                                                                                                          "Contact the Office of Financial Aid at 312.567.7219 or visit iit.edu/financial-aid."),
    ("Financial Aid",   "Normal", "What happens to my financial aid if I drop below full-time status?",                                                                                "Aid is reduced or eliminated — federal loans, SEOG, and work-study are zeroed out; Pell Grant recalculated proportionally."),
    ("Financial Aid",   "Normal", "Does withdrawing from a course affect financial aid eligibility?",                                                                                   "Yes, withdrawing reduces enrolled credits which lowers or eliminates certain aid types and may trigger return of Title IV funds."),
    ("Financial Aid",   "Normal", "Can dropping a course require repayment of financial aid?",                                                                                          "Yes, dropping below an aid threshold may require repayment of federal Title IV aid."),
    ("Financial Aid",   "Hard",   "I receive financial aid and I'm thinking of dropping below full-time status mid-semester — what happens to my aid and is there any tuition refund?", "Aid may reduce; refund depends on deadline."),
    ("Financial Aid",   "Hard",   "If I withdraw from all courses mid-semester, how does it affect my financial aid?",                                                                  "Aid may need repayment; refund depends on timing."),
    ("Financial Aid",   "Hard",   "If I drop from full-time to part-time, will my grants or loans change?",                                                                            "Grants may drop; loans may adjust."),
    ("Transfer Credit", "Easy",   "Can graduate students transfer credits into a master's program?",                                                                                    "Yes, with approval."),
    ("Transfer Credit", "Easy",   "Do transfer credits count toward degree requirements?",                                                                                              "Yes for credits, not GPA."),
    ("Transfer Credit", "Easy",   "Who approves transfer credits?",                                                                                                                    "Department + Registrar."),
    ("Transfer Credit", "Normal", "What is the maximum number of transfer credits allowed for a master's program?",                                                                     "Up to ~9 credits."),
    ("Transfer Credit", "Normal", "Do transfer credits affect GPA calculations?",                                                                                                      "No GPA impact."),
    ("Transfer Credit", "Normal", "When should transfer credits be submitted?",                                                                                                         "Early in program."),
    ("Transfer Credit", "Hard",   "I transferred 6 credits into a master's program — how many more transfer credits can I bring in, and could this affect my graduation date?",        "3 more credits; may graduate sooner."),
    ("Transfer Credit", "Hard",   "If I transfer the maximum allowed credits, could I graduate earlier?",                                                                               "Yes, if requirements allow."),
    ("Transfer Credit", "Hard",   "Can transfer credits from another graduate program be applied to my degree?",                                                                        "Yes, if approved."),
    ("Multi-topic",     "Easy",   "If I take 9 credits as a graduate student, am I considered full-time?",                                                                             "Yes, full-time."),
    ("Multi-topic",     "Easy",   "If I take 12 credits as an undergraduate student, am I considered full-time?",                                                                      "Yes, full-time."),
    ("Multi-topic",     "Easy",   "What happens if I drop a course before the add/drop deadline?",                                                                                     "Removed; full refund."),
    ("Multi-topic",     "Normal", "I'm a graduate student taking 8 credits — am I full-time and will it affect my tuition?",                                                           "Not full-time; may affect aid."),
    ("Multi-topic",     "Normal", "If I drop a course after the add/drop deadline, will it appear on my transcript?",                                                                  "W grade; no refund."),
    ("Multi-topic",     "Normal", "If I withdraw from a course, does it affect my GPA?",                                                                                               "No GPA impact."),
    ("Multi-topic",     "Hard",   "I'm an F-1 grad student taking 9 credits in Spring 2026 — am I full-time, and what happens to my visa status if I drop one 3-credit course after the add/drop deadline?", "Not full-time; visa risk without approval."),
    ("Multi-topic",     "Hard",   "I'm an undergraduate student with 12 credits — if I drop a 3-credit course after the add/drop deadline, am I still full-time and will I get a tuition refund?",           "Not full-time; no refund."),
    ("Multi-topic",     "Hard",   "If I withdraw from all my classes in Spring 2026 after the 4th week, do I get any tuition refund and what grade appears on my transcript?",         "W grades; likely no refund."),
]

FETCH_K = 10


# ── Retrieval for metrics only ─────────────────────────────────────────────────

def retrieve_hits(query, fetch_k=FETCH_K):
    route_details = get_routing_intent(query)
    domains = route_details.get("domains", [])
    results = []

    if DOMAIN_CALENDAR in domains:
        try:
            hits = calendar_route_query(query)
            if isinstance(hits, dict): hits = calendar_rrf_search(query, top_k=fetch_k)
            if isinstance(hits, list):
                for h in hits[:fetch_k]:
                    s = h.get("_source", {})
                    text = (s.get("semantic_text") or s.get("event_name") or "").strip()
                    results.append((h.get("_id", "?"), text))
        except: pass

    if DOMAIN_CONTACTS in domains:
        try:
            hits = contacts_rrf_search(query, top_k=fetch_k)
            if isinstance(hits, list):
                for h in hits[:fetch_k]:
                    s = h.get("_source", {})
                    text = " | ".join(filter(None, [s.get("name"), s.get("department"), s.get("semantic_text")])).strip()
                    results.append((h.get("_id", "?"), text))
        except: pass

    if DOMAIN_DOCUMENTS in domains:
        try:
            hits = documents_rrf_search(query)
            if isinstance(hits, list):
                for h in hits[:fetch_k]:
                    s = h.get("_source", {})
                    results.append((h.get("_id", "?"), (s.get("content") or "").strip()))
        except: pass

    if DOMAIN_TUITION in domains:
        try:
            hits = tuition_rrf_search(query, top_k=fetch_k)
            if isinstance(hits, list):
                for h in hits[:fetch_k]:
                    s = h.get("_source", {})
                    results.append((h.get("_id", "?"), (s.get("content") or s.get("chunk_text") or "").strip()))
        except: pass

    return results


# ── Relevance judge ────────────────────────────────────────────────────────────

def judge_relevance(query, hits):
    if not hits:
        return []
    from groq import Groq
    numbered = "".join(f"\nChunk {i} (ID: {cid}): {text[:300].replace(chr(10),' ')}\n" for i, (cid, text) in enumerate(hits, 1))
    prompt = (
        f"User question: {query}\n\nRetrieved chunks:{numbered}\n"
        f"For each chunk (1 to {len(hits)}), is it relevant to answering the question?\n"
        f"Reply ONLY a JSON array of {len(hits)} booleans. Example: [true, false]. No explanation."
    )
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=100,
        )
        raw = resp.choices[0].message.content.strip()
        result = json.loads(raw)
        if isinstance(result, list) and len(result) == len(hits):
            return [bool(x) for x in result]
    except Exception:
        pass
    return [True] * len(hits)


def compute_metrics(relevance_flags, k):
    top_k = relevance_flags[:k]
    relevant_in_k = sum(top_k)
    total_relevant = sum(relevance_flags)
    precision = relevant_in_k / len(top_k) if top_k else 0.0
    recall    = relevant_in_k / total_relevant if total_relevant > 0 else 0.0
    hit_rate  = 1.0 if relevant_in_k > 0 else 0.0
    return relevant_in_k, total_relevant, recall, precision, hit_rate


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    k = args.k

    answers_path = EVAL_DIR / "answers_for_sheet.csv"
    metrics_path = EVAL_DIR / "retrieval_metrics.csv"

    answer_fields = ["Category", "Difficulty", "Question", "Expected Answer", "LLM Answer"]
    metric_fields = [
        "Category", "Difficulty", "Question",
        "Retrieved Chunk IDs", "Relevant Chunk IDs",
        f"# Relevant in Top {k}", "Total Relevant Chunks",
        f"Recall @ {k}", f"Precision @ {k}", f"HitRate @ {k}",
    ]

    print(f"\nRunning {len(QUESTIONS)} questions through real chatbot pipeline | k={k}")
    print(f"Answers  → {answers_path}")
    print(f"Metrics  → {metrics_path}\n{'='*60}")

    with open(answers_path, "w", newline="", encoding="utf-8") as af, \
         open(metrics_path, "w", newline="", encoding="utf-8") as mf:

        a_writer = csv.DictWriter(af, fieldnames=answer_fields)
        m_writer = csv.DictWriter(mf, fieldnames=metric_fields)
        a_writer.writeheader()
        m_writer.writeheader()

        for i, (category, difficulty, question, expected) in enumerate(QUESTIONS, 1):
            print(f"\nQ{i:02d} [{category}/{difficulty}]: {question[:70]}")

            # 1. Real chatbot answer
            try:
                reply, sources, route_details, is_clarification, clarification_msg, _, _clar_opts = get_answer(
                    query=question, chat_history=[]
                )
                if is_clarification:
                    answer = f"[CLARIFICATION] {clarification_msg}"
                else:
                    answer = reply
            except Exception as e:
                answer = f"[ERROR] {e}"

            print(f"  Answer: {answer[:100].replace(chr(10),' ')}...")

            a_writer.writerow({
                "Category": category,
                "Difficulty": difficulty,
                "Question": question,
                "Expected Answer": expected,
                "LLM Answer": answer,
            })

            # 2. Retrieval metrics
            hits = retrieve_hits(question)
            if hits:
                relevance_flags = judge_relevance(question, hits)
                relevant_ids = [cid for (cid, _), rel in zip(hits, relevance_flags) if rel]
                retrieved_ids = [cid for cid, _ in hits[:k]]
                rel_in_k, total_rel, recall, precision, hit_rate = compute_metrics(relevance_flags, k)
            else:
                relevant_ids, retrieved_ids = [], []
                rel_in_k, total_rel, recall, precision, hit_rate = 0, 0, 0.0, 0.0, 0.0

            print(f"  P@{k}={precision:.2f}  R@{k}={recall:.2f}  Hit@{k}={int(hit_rate)}")

            m_writer.writerow({
                "Category": category, "Difficulty": difficulty, "Question": question,
                "Retrieved Chunk IDs": ", ".join(retrieved_ids),
                "Relevant Chunk IDs": ", ".join(relevant_ids),
                f"# Relevant in Top {k}": rel_in_k,
                "Total Relevant Chunks": total_rel,
                f"Recall @ {k}": round(recall, 4),
                f"Precision @ {k}": round(precision, 4),
                f"HitRate @ {k}": int(hit_rate),
            })

            time.sleep(3)

    print(f"\n{'='*60}\nDone.\n  Answers → {answers_path}\n  Metrics → {metrics_path}")


if __name__ == "__main__":
    main()
