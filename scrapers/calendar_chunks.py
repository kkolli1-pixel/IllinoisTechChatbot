"""Aggregate multi-day calendar events into single logical events (spans).

Reads data/calendar_events.json, groups by (term, event_name), merges
consecutive dates into spans, prints the first 5, and writes the full
result to data/calendar_chunks.json.

Run: python -m scrapers.calendar_chunks
"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pprint

# Paths (relative to project root, same pattern as calendar_scraper)
_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
CALENDAR_JSON = DATA_DIR / "calendar_events.json"
AGGREGATED_JSON = DATA_DIR / "calendar_chunks.json"


def merge_consecutive_dates(sorted_date_url_pairs):
    """
    Given a sorted list of (date, source_url), yield spans (start_date, end_date, source_urls).
    Same day: add url to current span. Consecutive day: extend span. Gap: close span, start new.
    source_urls are unique and preserve insertion order.
    """
    if not sorted_date_url_pairs:
        return

    start = end = sorted_date_url_pairs[0][0]
    urls = [sorted_date_url_pairs[0][1]]

    for i in range(1, len(sorted_date_url_pairs)):
        current_date, source_url = sorted_date_url_pairs[i]
        prev_date = sorted_date_url_pairs[i - 1][0]

        if current_date == prev_date:
            urls.append(source_url)
        elif (current_date - prev_date).days == 1:
            end = current_date
            urls.append(source_url)
        else:
            yield start, end, list(dict.fromkeys(urls))
            start = end = current_date
            urls = [source_url]

    yield start, end, list(dict.fromkeys(urls))


def main():
    with open(CALENDAR_JSON, encoding="utf-8") as f:
        rows = json.load(f)

    # Group by (term, event_name)
    groups = defaultdict(list)
    for row in rows:
        key = (row["term"], row["event_name"])
        groups[key].append((row["event_date"], row["source_url"]))

    # Sort each group by date and merge consecutive dates
    aggregated = []
    for (term, event_name), date_url_pairs in groups.items():
        parsed = [
            (datetime.strptime(d, "%Y-%m-%d").date(), url)
            for d, url in date_url_pairs
        ]
        parsed.sort(key=lambda x: x[0])

        for start_date, end_date, source_urls in merge_consecutive_dates(parsed):
            aggregated.append({
                "term": term,
                "event_name": event_name,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "source_urls": source_urls,
            })

    # Sort aggregated for stable output
    aggregated.sort(key=lambda r: (r["start_date"], r["term"], r["event_name"]))

    # Write full result to data/calendar_events_aggregated.json
    with open(AGGREGATED_JSON, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    print("First 5 aggregated events:\n")
    pprint(aggregated[:5])
    print("\nAggregated JSON written to data/calendar_events_aggregated.json")


if __name__ == "__main__":
    main()
