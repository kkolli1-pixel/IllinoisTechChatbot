import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_root))

from router.calendar_router import route_query

def main():
    print("Calendar Search CLI\n")
    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break


        if not query or query.lower() == "quit":
            break

        try:
            hits = route_query(query)
            score = max((h.get("_score") or 0.0 for h in hits), default=0.0)
            print(f"\nTop rerank score: {score:.4f} | Hits: {len(hits)}\n")

            sources = []
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
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
