import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_root))

from search.contacts_search import contacts_rrf_search

def main():
    print("Contacts Search CLI\n")
    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query or query.lower() == "quit":
            break

        try:
            hits = contacts_rrf_search(query)
            # `contacts_rrf_search()` can return either:
            # - a clarification dict: {"needs_clarification": True, "message": ..., "options": [...]}
            # - a list of ES hits: [{"_score": ..., "_source": {...}}, ...]
            if isinstance(hits, dict) and hits.get("needs_clarification"):
                msg = hits.get("message") or "Clarification needed."
                opts = hits.get("options") or []
                print(f"\n[CLARIFICATION NEEDED]\n{msg}\n")
                if opts:
                    print("Options:")
                    for o in opts:
                        print(f"  - {o}")
                print()
                continue

            if not isinstance(hits, list):
                print(f"Error: unexpected return type from contacts_rrf_search(): {type(hits)}\n")
                continue

            score = max(
                (
                    (h.get("_score") or 0.0) if isinstance(h, dict) else 0.0
                    for h in hits
                ),
                default=0.0,
            )
            print(f"\nTop rerank score: {score:.4f} | Hits: {len(hits)}\n")

            sources = []
            for h in hits:
                if not isinstance(h, dict) or "_source" not in h:
                    continue
                s = h["_source"]

                print(f"[{s['category'].upper()}] {s['name']}")
                print(f"Department: {s['department']}")

                if s['description']:
                    print(f"Description: {s['description']}")
                if s['phone']:
                    print(f"Phone: {s['phone']}")
                if s['fax']:
                    print(f"Fax: {s['fax']}")
                if s['email']:
                    print(f"Email: {s['email']}")
                if s['building']:
                    print(f"Building: {s['building']}")
                if s['address']:
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
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()