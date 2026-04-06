import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_root))

from search.tuition_search import tuition_rrf_search

def main():
    print("Tuition Search CLI\n")
    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query or query.lower() == "quit":
            break

        try:
            hits = tuition_rrf_search(query)
            score = max((h.get("_rerank_score") or h.get("_score") or 0.0 for h in hits), default=0.0)
            print(f"\nTop rerank score: {score:.4f} | Hits: {len(hits)}\n")

            sources = []
            for h in hits:
                s = h.get("_source", {})

                school = s.get("school") or "N/A"
                level = s.get("level") or "N/A"
                section = s.get("section") or ""
                fee_name = s.get("fee_name") or ""
                academic_year = s.get("academic_year") or ""
                term = s.get("term") or ""
                enrollment = s.get("enrollment") or ""
                program = s.get("program") or ""
                unit = s.get("unit") or ""
                amount = s.get("amount_value")

                header = f"[{school} | {level}]"
                print(header)

                if section:
                    print(f"Section: {section}")
                if fee_name:
                    print(f"Fee: {fee_name}")
                if academic_year:
                    print(f"Year: {academic_year}")
                if term:
                    print(f"Term: {term}")
                if enrollment:
                    print(f"Enrollment: {enrollment}")
                if program:
                    print(f"Program: {program}")
                if unit or amount is not None:
                    amount_str = f"{amount:.2f}" if isinstance(amount, (int, float)) else str(amount)
                    print(f"Amount: {amount_str} {unit}".strip())

                content = s.get("content")
                if content:
                    print(f"Content: {content}")

                url = s.get("source_url")
                if url and url not in sources:
                    sources.append(url)

                print()

            if sources:
                print("Sources:")
                for u in sources:
                    print(f"  - {u}")

            print()
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()

