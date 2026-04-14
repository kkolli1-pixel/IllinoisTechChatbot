"""
Clarification & context-carryover test suite.
Runs conversational flows against the live API and saves results.
"""

import json
import requests
import time

API = "https://gpunodecjd79jliorokn-nfsz9bk0140k74a9phshq7dihf3t.tec-s1.onthetaedgecloud.com/ask"

def ask(prompt, chat_history=None, pending_context=None):
    """Send a question to the API."""
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
    """
    Run a multi-turn conversation.
    steps: list of dicts with 'prompt' and 'expected' keys.
    Returns list of results.
    """
    chat_history = []
    pending_context = None
    results = []

    for step in steps:
        prompt = step["prompt"]
        expected = step["expected"]

        resp = ask(prompt, chat_history=chat_history, pending_context=pending_context)
        time.sleep(1)  # rate limit

        result = {
            "prompt": prompt,
            "expected": expected,
            "response": resp.get("response", ""),
            "is_clarification": resp.get("is_clarification", False),
            "sources": resp.get("sources", []),
            "pending_context": resp.get("pending_context"),
        }
        results.append(result)

        # Build chat history for next turn
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": resp.get("response", "")})

        # Carry pending_context if clarification
        if resp.get("is_clarification") and resp.get("pending_context"):
            pending_context = resp["pending_context"]
        else:
            pending_context = None

    return results


# ── Define test conversations ────────────────────────────────────────────────

conversations = []

# --- Group 1: Simple clarifications ---
conversations.append({
    "name": "1. When does the semester start?",
    "steps": [
        {"prompt": "When does the semester start?", "expected": "Clarification: Which semester/year?"},
    ]
})

conversations.append({
    "name": "2. When is graduation?",
    "steps": [
        {"prompt": "When is graduation?", "expected": "Clarification: Which semester/year?"},
    ]
})

conversations.append({
    "name": "3. When does the term end?",
    "steps": [
        {"prompt": "When does the term end?", "expected": "Clarification: Which semester/year?"},
    ]
})

conversations.append({
    "name": "4. How much is tuition?",
    "steps": [
        {"prompt": "How much is tuition?", "expected": "Clarification: undergrad or grad? which semester?"},
    ]
})

conversations.append({
    "name": "5. How many credits is full time?",
    "steps": [
        {"prompt": "How many credits is full time?", "expected": "Clarification: grad or undergrad?"},
    ]
})

conversations.append({
    "name": "6. When can I register?",
    "steps": [
        {"prompt": "When can I register?", "expected": "Clarification: Which semester?"},
    ]
})

conversations.append({
    "name": "7. When is the last day to withdraw?",
    "steps": [
        {"prompt": "When is the last day to withdraw?", "expected": "Clarification: Which semester?"},
    ]
})

conversations.append({
    "name": "8. How does parking work?",
    "steps": [
        {"prompt": "How does parking work?", "expected": "Clarification: student permits, visitor parking, or daily rates?"},
    ]
})

conversations.append({
    "name": "9. When are finals?",
    "steps": [
        {"prompt": "When are finals?", "expected": "Clarification: Which semester/year?"},
    ]
})

# --- Group 2: Spring 2026 classes + context follow-up ---
conversations.append({
    "name": "10-11. Spring 2026 classes start + when do they end (context)",
    "steps": [
        {"prompt": "When do spring 2026 classes start?", "expected": "Answer first day of spring 2026"},
        {"prompt": "When do they end?", "expected": "Answer last day of spring 2026 courses (context carryover)"},
    ]
})

# --- Group 3: Hardship withdrawal + who approves (context) ---
conversations.append({
    "name": "12-13. Hardship withdrawal + who approves it (context)",
    "steps": [
        {"prompt": "What is the hardship withdrawal policy?", "expected": "Answer hardship withdrawal policy"},
        {"prompt": "Who approves it?", "expected": "Provide office or contact that handles it (context)"},
    ]
})

# --- Group 4: Tuition clarification → Graduate → Never Mind (reset) ---
conversations.append({
    "name": "14-16. Tuition → Graduate → Never Mind (context reset)",
    "steps": [
        {"prompt": "How much is tuition?", "expected": "Clarification: ask which school/level"},
        {"prompt": "Graduate", "expected": "Uses context to narrow to graduate tuition"},
        {"prompt": "Never Mind", "expected": "Clear pending context, respond helpfully without answering tuition"},
    ]
})

# --- Group 5: Academic probation + how long (context) ---
conversations.append({
    "name": "17-18. Academic probation + how long (context)",
    "steps": [
        {"prompt": "What is academic probation?", "expected": "Answer academic probation policy"},
        {"prompt": "How long can I be on it?", "expected": "Answer consecutive semester limit (context)"},
    ]
})

# --- Group 6: Tuition → Mies gradute (spelling error) ---
conversations.append({
    "name": "19-20. Tuition → Mies gradute (spelling error handling)",
    "steps": [
        {"prompt": "How much is tuition?", "expected": "Clarification: ask which school/level"},
        {"prompt": "Mies gradute", "expected": "Handle spelling error, answer Mies graduate tuition"},
    ]
})

# --- Group 7: Finals → topic switch to plagiarism ---
conversations.append({
    "name": "21-22. Finals → plagiarism policy (topic switch)",
    "steps": [
        {"prompt": "When are finals?", "expected": "Clarification: ask which semester"},
        {"prompt": "Actually, what is the plagiarism policy?", "expected": "Detect topic change, answer plagiarism policy"},
    ]
})

# --- Group 8: Co-terminal program + GPA (context) ---
conversations.append({
    "name": "23-24. Co-terminal program + GPA requirement (context)",
    "steps": [
        {"prompt": "What is the co-terminal program?", "expected": "Explain co-terminal program"},
        {"prompt": "What GPA do I need to apply?", "expected": "Answer co-terminal GPA requirement (3.0) (context)"},
    ]
})

# --- Group 9: Drop deadline → topic switch to probation ---
conversations.append({
    "name": "25-26. Drop deadline → probation (topic switch)",
    "steps": [
        {"prompt": "When is the drop deadline?", "expected": "Clarification: ask which semester"},
        {"prompt": "What is the academic probation policy?", "expected": "Detect topic change, answer probation policy"},
    ]
})

# --- Group 10: Max course load → Graduate (context) ---
conversations.append({
    "name": "27-28. Max course load → Graduate (context)",
    "steps": [
        {"prompt": "What is the maximum course load I can take each semester?", "expected": "Clarification or answer about max credits"},
        {"prompt": "Graduate", "expected": "Answer for graduate max course load (context)"},
    ]
})

# --- Group 11: GPA for graduation + honors + highest honor (context chain) ---
conversations.append({
    "name": "29-31. Graduation GPA → honors → highest honor (context chain)",
    "steps": [
        {"prompt": "What is the GPA requirement for graduation?", "expected": "Answer graduation GPA requirement"},
        {"prompt": "What about for honors?", "expected": "Answer honors GPA thresholds (context)"},
        {"prompt": "What is the highest honor?", "expected": "Answer highest honor GPA threshold (context)"},
    ]
})

# --- Group 12: Amnesty policy + graduate applicability (context) ---
conversations.append({
    "name": "32-33. Amnesty policy + does it apply to grad students (context)",
    "steps": [
        {"prompt": "What is the amnesty policy?", "expected": "Answer amnesty policy"},
        {"prompt": "Does it apply to graduate students too?", "expected": "Answer yes/no with context (context)"},
    ]
})


# ── Run all conversations ────────────────────────────────────────────────────

all_results = []
for conv in conversations:
    print(f"\n{'='*60}")
    print(f"Running: {conv['name']}")
    print(f"{'='*60}")

    results = run_conversation(conv["steps"])

    for r in results:
        tag = "CLAR" if r["is_clarification"] else "ANS"
        print(f"\n  Q: {r['prompt']}")
        print(f"  Expected: {r['expected']}")
        print(f"  [{tag}] {r['response'][:200]}")

    all_results.append({
        "conversation": conv["name"],
        "turns": results,
    })

# Save results
output_path = "evaluation/clarification_context_test_33.json"
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n\nResults saved to {output_path}")
print(f"Total conversations: {len(conversations)}")
total_turns = sum(len(c["steps"]) for c in conversations)
print(f"Total turns: {total_turns}")
