import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_root))

from search.documents_search import documents_rrf_search
from common.embedding_model import model_large

def main():
    print("Documents Search CLI\n")
    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query or query.lower() == "quit":
            break

        try:
            hits = documents_rrf_search(query)
            score = max((h.get("_rerank_score") or 0.0 for h in hits), default=0.0)
            print(f"\nTop rerank score: {score:.4f} | Hits: {len(hits)}\n")

            sources = []
            for h in hits:
                s = h["_source"]

                print(f"[{s['doc_type'].upper()}] {s['doc_name']}")
                print(f"Topic: {s['topic']}")
                print(f"Page {s['page_start']}-{s['page_end']}")

                if s['content']:
                    print(f"Content: {s['content']}")

                url = s.get("source_url")
                if url and url not in sources:
                    sources.append(url)

            if sources:
                print("\nSources:")
                for u in sources:
                    print(f"  - {u}")
            
            print()
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()