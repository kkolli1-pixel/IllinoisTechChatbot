"""
Comprehensive 50-question test — all types, all domains, all formats.
Single-turn and multi-turn. Tests current live system.
"""

import json
import requests
import time

API = "https://gpunodecjd79jliorokn-nfsz9bk0140k74a9phshq7dihf3t.tec-s1.onthetaedgecloud.com/ask"

def ask(prompt, chat_history=None, pending_context=None):
    payload = {"prompt": prompt}
    if chat_history:
        payload["chat_history"] = chat_history
    if pending_context:
        payload["pending_context"] = pending_context
    try:
        r = requests.post(API, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"response": f"ERROR: {e}", "is_clarification": False, "sources": [], "pending_context": None}

def run_conversation(steps):
    chat_history = []
    pending_context = None
    results = []
    for step in steps:
        resp = ask(step["prompt"], chat_history=chat_history, pending_context=pending_context)
        time.sleep(1)
        tag = "CLAR" if resp.get("is_clarification") else "ANS"
        results.append({
            "prompt": step["prompt"],
            "expected": step["expected"],
            "tag": tag,
            "response": resp.get("response", "")[:300],
            "pending_context": resp.get("pending_context"),
        })
        print(f"  Q: {step['prompt']}")
        print(f"  Expected: {step['expected']}")
        print(f"  [{tag}] {resp.get('response', '')[:200]}\n")
        chat_history.append({"role": "user", "content": step["prompt"]})
        chat_history.append({"role": "assistant", "content": resp.get("response", "")})
        pending_context = resp.get("pending_context") if resp.get("is_clarification") else None
    return results

tests = [

    # ── 1-word / 2-word inputs ──────────────────────────────────────────────
    {"name": "1. One word - Hi",          "steps": [{"prompt": "Hi",              "expected": "Greeting response"}]},
    {"name": "2. One word - Tuition",     "steps": [{"prompt": "Tuition",         "expected": "Clarification or tuition info"}]},
    {"name": "3. One word - Calendar",    "steps": [{"prompt": "Calendar",        "expected": "Some helpful response about calendar"}]},
    {"name": "4. Two words - proper name","steps": [{"prompt": "Yuhan Ding",      "expected": "Contact info for Yuhan Ding"}]},
    {"name": "5. Two words - Ioan Raicu", "steps": [{"prompt": "Ioan Raicu",      "expected": "Contact info for Ioan Raicu"}]},
    {"name": "6. One word - Graduate",    "steps": [{"prompt": "Graduate",        "expected": "Some helpful response, not contacts"}]},
    {"name": "7. Two words - Fall 2026",  "steps": [{"prompt": "Fall 2026",       "expected": "Fall 2026 academic calendar info"}]},
    {"name": "8. Two words - Spring 2026","steps": [{"prompt": "Spring 2026",     "expected": "Spring 2026 calendar info"}]},

    # ── OOD / off-topic ─────────────────────────────────────────────────────
    {"name": "9. OOD - weather",          "steps": [{"prompt": "What is the weather in Chicago?",      "expected": "OOD rejection"}]},
    {"name": "10. OOD - sports",          "steps": [{"prompt": "Who won the NBA Finals?",              "expected": "OOD rejection"}]},
    {"name": "11. OOD - trivia",          "steps": [{"prompt": "What is the capital of France?",       "expected": "OOD rejection"}]},
    {"name": "12. OOD - essay",           "steps": [{"prompt": "Can you write my essay for me?",       "expected": "OOD rejection, not 'I don't have info'"}]},
    {"name": "13. OOD - homework",        "steps": [{"prompt": "Do my homework for me",                "expected": "OOD rejection"}]},
    {"name": "14. OOD - CEO",             "steps": [{"prompt": "Who is the CEO of Tesla?",             "expected": "OOD rejection"}]},
    {"name": "15. OOD - restaurant",      "steps": [{"prompt": "Best pizza near campus",               "expected": "OOD rejection"}]},

    # ── Greetings / meta ────────────────────────────────────────────────────
    {"name": "16. Who are you",           "steps": [{"prompt": "Who are you?",                         "expected": "Introduces itself as IIT chatbot"}]},
    {"name": "17. What can you do",       "steps": [{"prompt": "What can you do?",                     "expected": "Explains capabilities"}]},
    {"name": "18. Thank you",             "steps": [{"prompt": "Thank you!",                           "expected": "Polite response"}]},
    {"name": "19. Goodbye",              "steps": [{"prompt": "Goodbye",                              "expected": "Polite response"}]},

    # ── Direct single-turn questions ────────────────────────────────────────
    {"name": "20. Plagiarism policy",     "steps": [{"prompt": "What is the plagiarism policy?",       "expected": "Answer about academic integrity"}]},
    {"name": "21. FERPA",                 "steps": [{"prompt": "What is FERPA?",                       "expected": "Answer about FERPA"}]},
    {"name": "22. Amnesty policy",        "steps": [{"prompt": "What is the amnesty policy?",          "expected": "Answer about amnesty policy"}]},
    {"name": "23. Parking",              "steps": [{"prompt": "How does parking work on campus?",      "expected": "Answer about parking permits"}]},
    {"name": "24. Degree verification",  "steps": [{"prompt": "How do I verify my degree?",            "expected": "Answer about NSC degree verification"}]},
    {"name": "25. Registrar contact",    "steps": [{"prompt": "How do I contact the Registrar?",       "expected": "Registrar contact info"}]},
    {"name": "26. Financial Aid contact","steps": [{"prompt": "Who do I contact for financial aid?",   "expected": "Financial Aid contact info"}]},
    {"name": "27. Leave of absence",     "steps": [{"prompt": "How do I take a leave of absence?",     "expected": "Leave of absence policy"}]},
    {"name": "28. Incomplete grade",     "steps": [{"prompt": "What happens if I get an incomplete?",  "expected": "Incomplete grade policy"}]},
    {"name": "29. Housing requirement",  "steps": [{"prompt": "Is there a housing requirement?",       "expected": "Freshmen housing requirement"}]},
    {"name": "30. GPA for graduation",   "steps": [{"prompt": "What GPA do I need to graduate?",       "expected": "Graduation GPA requirement"}]},
    {"name": "31. Latin honors",         "steps": [{"prompt": "What GPA do I need for Latin honors?",  "expected": "Summa/Magna/Cum Laude thresholds"}]},
    {"name": "32. Registration hold",    "steps": [{"prompt": "I have a registration hold, what do I do?", "expected": "How to clear a hold"}]},
    {"name": "33. Co-terminal program",  "steps": [{"prompt": "What is the co-terminal program?",      "expected": "Explain co-terminal program"}]},
    {"name": "34. Max course load",      "steps": [{"prompt": "What is the maximum course load?",      "expected": "Max credits per semester"}]},

    # ── Clarification flows ─────────────────────────────────────────────────
    {"name": "35. Tuition → Mies",
     "steps": [
         {"prompt": "How much is tuition?",     "expected": "Clarification: which school?"},
         {"prompt": "Mies",                     "expected": "Mies tuition rates"},
     ]},
    {"name": "36. Calendar → Spring 2026",
     "steps": [
         {"prompt": "When does the semester start?", "expected": "Clarification: which semester?"},
         {"prompt": "Spring 2026",                   "expected": "Spring 2026 start date"},
     ]},
    {"name": "37. Contact → Physics",
     "steps": [
         {"prompt": "I need to speak with someone", "expected": "Clarification: which department?"},
         {"prompt": "Physics",                      "expected": "Physics department contact info"},
     ]},
    {"name": "38. Finals → Fall 2026",
     "steps": [
         {"prompt": "When are finals?",  "expected": "Clarification: which semester?"},
         {"prompt": "Fall 2026",         "expected": "Fall 2026 finals dates"},
     ]},

    # ── Topic switches ───────────────────────────────────────────────────────
    {"name": "39. Topic switch mid clarification",
     "steps": [
         {"prompt": "When is the drop deadline?",            "expected": "Clarification: which semester?"},
         {"prompt": "Actually what is academic probation?",  "expected": "Topic switch — answers probation"},
     ]},
    {"name": "40. Topic switch tuition → policy",
     "steps": [
         {"prompt": "How much is tuition?",                  "expected": "Clarification: which school?"},
         {"prompt": "What is the plagiarism policy?",        "expected": "Topic switch — answers plagiarism"},
     ]},

    # ── Context carryover ───────────────────────────────────────────────────
    {"name": "41. Context - hardship withdrawal approver",
     "steps": [
         {"prompt": "What is the hardship withdrawal policy?", "expected": "Hardship withdrawal policy"},
         {"prompt": "How do I apply for it?",                  "expected": "Application steps (context)"},
     ]},
    {"name": "42. Context - co-terminal GPA",
     "steps": [
         {"prompt": "What is the co-terminal program?",  "expected": "Explains co-terminal"},
         {"prompt": "What GPA do I need?",               "expected": "3.0 GPA requirement (context)"},
     ]},
    {"name": "43. Context - honors chain",
     "steps": [
         {"prompt": "What is the highest graduation honor?",  "expected": "Summa cum laude"},
         {"prompt": "What GPA do I need for that?",          "expected": "3.90 GPA (context)"},
     ]},

    # ── Vague / ambiguous ───────────────────────────────────────────────────
    {"name": "44. Vague - fees",          "steps": [{"prompt": "What are the fees?",                   "expected": "Clarification or fee info"}]},
    {"name": "45. Vague - deadline",      "steps": [{"prompt": "What is the deadline?",                "expected": "Clarification about which deadline"}]},
    {"name": "46. Vague - office hours",  "steps": [{"prompt": "What are the office hours?",           "expected": "Clarification or office hours info"}]},

    # ── Multi-part questions ────────────────────────────────────────────────
    {"name": "47. Multi-part",
     "steps": [{"prompt": "What is the attendance policy and who do I contact if I have issues?",
                "expected": "Answers both attendance policy and contact info"}]},

    # ── Spelling errors ─────────────────────────────────────────────────────
    {"name": "48. Spelling - probashion",  "steps": [{"prompt": "What is the academic probashion policy?", "expected": "Academic probation policy"}]},
    {"name": "49. Spelling - tuision",     "steps": [{"prompt": "How much is tuision at Mies?",            "expected": "Mies tuition rates"}]},

    # ── Cancel / reset ──────────────────────────────────────────────────────
    {"name": "50. Cancel flow",
     "steps": [
         {"prompt": "How much is tuition?",  "expected": "Clarification: which school?"},
         {"prompt": "Never mind",            "expected": "Resets gracefully"},
     ]},
]

all_results = []
pass_count = 0
fail_count = 0

for test in tests:
    print(f"\n{'='*60}")
    print(f"Running: {test['name']}")
    print(f"{'='*60}")
    results = run_conversation(test["steps"])
    all_results.append({"name": test["name"], "turns": results})

output_path = "evaluation/comprehensive_50q_results.json"
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n\nResults saved to {output_path}")
print(f"Total tests: {len(tests)}")
print(f"Total turns: {sum(len(t['steps']) for t in tests)}")
