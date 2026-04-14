"""
Fresh test suite — new questions not previously tested.
Covers: name lookups, greetings, edge cases, new clarification flows.
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

def run(steps):
    chat_history = []
    pending_context = None
    results = []
    for step in steps:
        resp = ask(step["prompt"], chat_history=chat_history, pending_context=pending_context)
        time.sleep(1)
        tag = "CLAR" if resp.get("is_clarification") else "ANS"
        result = {
            "prompt": step["prompt"],
            "expected": step["expected"],
            "response": resp.get("response", ""),
            "is_clarification": resp.get("is_clarification", False),
            "pending_context": resp.get("pending_context"),
        }
        results.append(result)
        chat_history.append({"role": "user", "content": step["prompt"]})
        chat_history.append({"role": "assistant", "content": resp.get("response", "")})
        pending_context = resp.get("pending_context") if resp.get("is_clarification") else None
        print(f"\n  Q: {step['prompt']}")
        print(f"  Expected: {step['expected']}")
        print(f"  [{tag}] {resp.get('response', '')[:250]}")
    return results

conversations = []

# --- Name lookups ---
conversations.append({"name": "1. Faculty name lookup", "steps": [
    {"prompt": "Yuhan Ding", "expected": "Contact info for Yuhan Ding from contacts"},
]})

conversations.append({"name": "2. Faculty name with question", "steps": [
    {"prompt": "How do I contact Ioan Raicu?", "expected": "Contact info for Ioan Raicu"},
]})

# --- Greetings / chitchat ---
conversations.append({"name": "3. Hello greeting", "steps": [
    {"prompt": "Hello", "expected": "Friendly greeting, asks what they need"},
]})

conversations.append({"name": "4. Thank you", "steps": [
    {"prompt": "Thank you!", "expected": "Polite response, no info dump"},
]})

conversations.append({"name": "5. Who are you?", "steps": [
    {"prompt": "Who are you?", "expected": "Introduces itself as IIT chatbot"},
]})

# --- New clarification flows ---
conversations.append({"name": "6. Scholarship info", "steps": [
    {"prompt": "What scholarships are available?", "expected": "Answer or clarification about scholarships"},
]})

conversations.append({"name": "7. Internship credit", "steps": [
    {"prompt": "How do I get credit for an internship?", "expected": "Answer about internship/co-op credit policy"},
]})

conversations.append({"name": "8. GPA for honors → follow up", "steps": [
    {"prompt": "What GPA do I need for Latin honors?", "expected": "Answer about honors GPA thresholds"},
    {"prompt": "What is summa cum laude?", "expected": "Answer: summa cum laude GPA threshold (context)"},
]})

conversations.append({"name": "9. Tuition → Chicago-Kent → follow up", "steps": [
    {"prompt": "How much is tuition?", "expected": "Clarification: which school?"},
    {"prompt": "Chicago-Kent", "expected": "Chicago-Kent tuition rates"},
    {"prompt": "What about part time?", "expected": "Chicago-Kent part time rate (context)"},
]})

conversations.append({"name": "10. Leave of absence", "steps": [
    {"prompt": "How do I take a leave of absence?", "expected": "Answer about leave of absence policy"},
]})

conversations.append({"name": "11. Incomplete grade", "steps": [
    {"prompt": "What happens if I get an incomplete grade?", "expected": "Answer about incomplete grade policy"},
]})

conversations.append({"name": "12. Registration hold", "steps": [
    {"prompt": "I have a registration hold, what do I do?", "expected": "Answer about clearing a registration hold"},
]})

conversations.append({"name": "13. Degree verification", "steps": [
    {"prompt": "How do I verify my degree for an employer?", "expected": "Answer about degree verification / National Student Clearinghouse"},
]})

conversations.append({"name": "14. Stuart tuition → undergrad follow up", "steps": [
    {"prompt": "How much is tuition at Stuart?", "expected": "Stuart tuition rates"},
    {"prompt": "What about undergrad?", "expected": "Stuart undergrad tuition (context)"},
]})

conversations.append({"name": "15. OOD — personal advice", "steps": [
    {"prompt": "Can you write my essay for me?", "expected": "OOD rejection — polite decline"},
]})

conversations.append({"name": "16. OOD — weather", "steps": [
    {"prompt": "What is the weather in Chicago today?", "expected": "OOD rejection"},
]})

conversations.append({"name": "17. Campus housing policy", "steps": [
    {"prompt": "Is there a housing requirement for freshmen?", "expected": "Answer about on-campus housing requirement"},
]})

conversations.append({"name": "18. FERPA", "steps": [
    {"prompt": "What is FERPA?", "expected": "Answer about FERPA student records policy"},
]})

conversations.append({"name": "19. Coursera term → follow up", "steps": [
    {"prompt": "When does Coursera Spring 2026 Term A start?", "expected": "Answer with Coursera Term A start date"},
]})

conversations.append({"name": "20. Topic switch mid clarification", "steps": [
    {"prompt": "When is the add/drop deadline?", "expected": "Clarification: which semester?"},
    {"prompt": "Actually, what is the FERPA policy?", "expected": "Topic switch detected, answers FERPA"},
]})

# Run all
all_results = []
for conv in conversations:
    print(f"\n{'='*60}\nRunning: {conv['name']}\n{'='*60}")
    results = run(conv["steps"])
    all_results.append({"conversation": conv["name"], "turns": results})

output_path = "evaluation/fresh_test_results.json"
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n\nResults saved to {output_path}")
print(f"Total conversations: {len(conversations)}")
print(f"Total turns: {sum(len(c['steps']) for c in conversations)}")
