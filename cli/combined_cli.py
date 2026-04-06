import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_root))

load_dotenv(_root / ".env")

# Import router and search
from router.router import (

    get_routing_intent,
    DOMAIN_CALENDAR,
    DOMAIN_CONTACTS,
    DOMAIN_DOCUMENTS,
    DOMAIN_TUITION
)

from router.calendar_router import route_query as calendar_route_query
from search.contacts_search import contacts_rrf_search
from search.documents_search import documents_rrf_search
from search.tuition_search import tuition_rrf_search

# print calendar hits
def print_calendar_hits(hits):

    sources = []
    score = max((h.get("_score") or 0.0 for h in hits), default=0.0)

    print(f"\n[CALENDAR] Top score: {score:.4f} | Hits: {len(hits)}\n")

    for h in hits:

        s = h["_source"]

        start = s.get("start_date") or "N/A"
        end = s.get("end_date") or "N/A"

        date_str = start if start == end else f"{start} → {end}"

        print(f"  {date_str} | {s.get('event_name') or 'N/A'}")

        urls = s.get("source_urls") or ([s.get("source_url")] if s.get("source_url") else [])

        for url in urls:
            u = (url or "").strip()
            if u and u not in sources:
                sources.append(u)

    if sources:
        print("\nSources:")
        for u in sources:
            print(f"  - {u}")

    print()

# print contacts hits
def print_contacts_hits(hits):

    sources = []
    score = max((h.get("_score") or 0.0 for h in hits), default=0.0)

    print(f"\n[CONTACTS] Top score: {score:.4f} | Hits: {len(hits)}\n")

    for h in hits:

        s = h["_source"]

        print(f"[{s.get('category','').upper()}] {s.get('name','N/A')}")
        print(f"Department: {s.get('department','N/A')}")

        if s.get("description"):
            print(f"Description: {s['description']}")

        if s.get("phone"):
            print(f"Phone: {s['phone']}")

        if s.get("fax"):
            print(f"Fax: {s['fax']}")

        if s.get("email"):
            print(f"Email: {s['email']}")

        if s.get("building"):
            print(f"Building: {s['building']}")

        if s.get("address"):
            print(f"Address: {s['address']}")

        url = s.get("source_url")
        if url and url not in sources:
            sources.append(url)

        print()

    if sources:
        print("\nSources:")
        for u in sources:
            print(f"  - {u}")

    print()

# print tuition hits
def print_tuition_hits(hits):

    sources = []
    score = max((h.get("_score") or 0.0 for h in hits), default = 0.0)

    print(f"\n[TUITION] Top score: {score:.4f} | Hits: {len(hits)}\n")

    for h in hits:

        s = h["_source"]

        print(f"[{s.get('school','').upper()}] {s.get('level','N/A')}")
        print(f"Section: {s.get('section','N/A')}")
        print(f"Fee: {s.get('fee_name','N/A')}")
        print(f"Year: {s.get('academic_year','N/A')}")
        print(f"Term: {s.get('term','N/A')}")
        print(f"Enrollment: {s.get('enrollment','N/A')}")
        print(f"Program: {s.get('program','N/A')}")
        print(f"Unit: {s.get('unit','N/A')}")
        print(f"Amount: {s.get('amount_value','N/A')}")

        if s.get("content"):
            print(f"Content: {s['content']}")

        url = s.get("source_url")
        if url and url not in sources:
            sources.append(url)

        print()

    if sources:
        print("\nSources:")
        for u in sources:
            print(f"  - {u}")

    print()

# print documents hits
def print_documents_hits(hits):

    sources = []
    score = max((h.get("_score") or 0.0 for h in hits), default=0.0)

    print(f"\n[DOCUMENTS] Top score: {score:.4f} | Hits: {len(hits)}\n")

    for h in hits:

        s = h["_source"]

        print(f"[{s.get('doc_type','').upper()}] {s.get('doc_name','N/A')}")
        print(f"Topic: {s.get('topic','N/A')}")
        print(f"Page {s.get('page_start','?')}-{s.get('page_end','?')}")

        if s.get("content"):
            print(f"Content: {s['content']}")

        url = s.get("source_url")
        if url and url not in sources:
            sources.append(url)

        print()

    if sources:
        print("\nSources:")
        for u in sources:
            print(f"  - {u}")

    print()

# main cli
def main():

    print("CLI Search)\n")

    while True:

        try:
            query = input("Query: ").strip()

        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query or query.lower() == "quit":
            break

        try:

            # getting intent for the router
            intent = get_routing_intent(query)
            domains = intent.get("domains", [])

            if not domains:
                print("\nRouter could not confidently determine a domain.\n")
                continue

            print(f"\nRouter → {', '.join(domains)}")

            # Collect hits and clarifications across all domains first
            all_hits = {}
            all_clarifications = {}

            for domain in domains:

                if domain == DOMAIN_CALENDAR:
                    hits = calendar_route_query(query)

                elif domain == DOMAIN_CONTACTS:
                    hits = contacts_rrf_search(query)

                elif domain == DOMAIN_TUITION:
                    hits = tuition_rrf_search(query)

                elif domain == DOMAIN_DOCUMENTS:
                    hits = documents_rrf_search(query)

                else:
                    continue

                if isinstance(hits, dict) and hits.get("needs_clarification"):
                    all_clarifications[domain] = hits
                elif hits:
                    all_hits[domain] = hits

            # Only show clarification if NO domain retrieved anything
            if not all_hits:
                if all_clarifications:
                    # Show only the first clarification message
                    first = next(iter(all_clarifications.values()))
                    print("\nClarification needed:")
                    print(first["message"])
                    if first.get("options"):
                        print("Options: " + ", ".join(first["options"][:10]))
                else:
                    print("\nNo results found.\n")
                continue

            # Print results for each domain that retrieved
            for domain, hits in all_hits.items():
                if domain == DOMAIN_CALENDAR:
                    print_calendar_hits(hits)
                elif domain == DOMAIN_CONTACTS:
                    print_contacts_hits(hits)
                elif domain == DOMAIN_TUITION:
                    print_tuition_hits(hits)
                elif domain == DOMAIN_DOCUMENTS:
                    print_documents_hits(hits)

        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()