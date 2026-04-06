"""IIT calendar scraper. Run: python3 -m scrapers.calendar_scraper

Scrapes IIT academic calendar pages to structured JSON and Excel.
Outputs: JSON (term, event_date, event_name, event_type, source_url) and Excel.
"""
import json
import os
import re
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Paths (relative to project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
EXCEL_FILE = os.path.normpath(os.path.join(DATA_DIR, "university_calendar_data.xlsx"))
CALENDAR_JSON_FILE = os.path.normpath(os.path.join(DATA_DIR, "calendar_events.json"))

TERM_YEAR_PATTERN = re.compile(r"20\d{2}")

URLS_TO_SCRAPE = [
    {"name": "Academic", "url": "https://www.iit.edu/registrar/academic-calendar", "targets": ["Spring 2026", "Summer 2026"]},
    {"name": "Holidays", "url": "https://www.iit.edu/hr/employee-resources/paid-time#h27", "targets": ["2026-27 Calendar Year"], "max_tables": 1},
    {"name": "Academic_Future", "url": "https://www.iit.edu/registrar/academic-calendar/subsequent-academic-years", "targets": ["Fall 2026"]},
    {"name": "Coursera", "url": "https://www.iit.edu/coursera/coursera-academic-calendar", "targets": ["Coursera A Term", "Coursera B Term", "Coursera Summer 2026"]},
]

TERM_LABELS = {
    "Coursera A Term": "Coursera Spring 2026 (Term A)",
    "Coursera B Term": "Coursera Spring 2026 (Term B)",
}


def clean_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ").strip())


def parse_date(date_str: str, default_year: int) -> list[str]:
    if not date_str or not clean_text(date_str):
        return []
    date_str = clean_text(date_str)
    if re.search(r"\bTBA\b", date_str, re.I):
        return []
    months = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}
    
    # Check for full date with year first (e.g., "January 12, 2026")
    full_match = re.match(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", date_str, re.I)
    if full_match:
        mo, day, yr = full_match.groups()
        mo_num = months.get(mo.lower())
        if mo_num:
            try:
                return [datetime(int(yr), mo_num, int(day)).strftime("%Y-%m-%d")]
            except ValueError:
                pass
    
    # Check for date ranges WITH year (e.g., "March 16-21, 2026" or "March 16–21, 2026")
    range_match = re.match(r"([A-Za-z]+)\s+(\d{1,2})\s*[–\-]\s*(\d{1,2}),?\s*(\d{4})?", date_str, re.I)
    if range_match:
        mo, d1, d2, yr = range_match.groups()
        yr = int(yr) if yr else default_year
        mo_num = months.get(mo.lower())
        if mo_num:
            try:
                start, end = int(d1), int(d2)
                return [datetime(yr, mo_num, d).strftime("%Y-%m-%d") for d in range(min(start, end), max(start, end) + 1)]
            except ValueError:
                pass
    
    # Check for single date without year (e.g., "January 12")
    single_match = re.match(r"([A-Za-z]+)\s+(\d{1,2})\b", date_str, re.I)
    if single_match:
        mo, day = single_match.groups()
        mo_num = months.get(mo.lower())
        if mo_num:
            try:
                return [datetime(default_year, mo_num, int(day)).strftime("%Y-%m-%d")]
            except ValueError:
                pass
    
    return []

def _extract_events_from_cell(cell) -> list[str]:
    events = []
    for li in cell.find_all("li"):
        t = clean_text(li.get_text())
        if t:
            events.append(t)
    if not events:
        for line in cell.get_text(separator="\n").split("\n"):
            t = clean_text(line)
            if t:
                events.append(t)
    return events


def _term_year_from_target(target: str) -> int:
    m = TERM_YEAR_PATTERN.search(target)
    return int(m.group()) if m else datetime.now().year


def find_tables_for_target(soup, target, max_tables=None):
    pattern = re.compile(re.escape(target), re.IGNORECASE)
    for el in soup.find_all(["h2", "h3", "h4", "h5", "strong", "b", "p"]):
        if not pattern.search(el.get_text() or ""):
            continue
        curr = el if el.name not in ("strong", "b") else el.parent
        table = None
        for _ in range(5):
            table = curr.find_next_sibling("table")
            if table:
                break
            curr = curr.next_sibling
            if not curr:
                break
        if not table:
            table = el.find_next("table")
        if not table:
            continue
        tables = [table]
        if max_tables == 1:
            return tables
        nxt = table.find_next("table")
        if nxt and nxt.find_all("tr"):
            first_cells = nxt.find_all("tr")[0].find_all(["td", "th"])
            if len(first_cells) == 1:
                tables.append(nxt)
        return tables
    return []


def extract_calendar_section(soup, target, source_url, term=None, default_year=None, max_tables=None) -> list[dict]:
    term = term or target
    default_year = default_year or _term_year_from_target(term)
    term = TERM_LABELS.get(target, term)
    tables = find_tables_for_target(soup, target, max_tables=max_tables)
    if not tables:
        return []
    seen = set()
    records = []
    for table in tables:
        rows = table.find_all("tr")
        thead = table.find("thead")
        if thead:
            rows = [r for r in rows if r not in thead.find_all("tr")]
        use_session = len(tables) > 1
        for row in rows:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            if use_session and len(cells) == 1:
                continue
            date_str = clean_text(cells[0].get_text()) if cells else ""
            event_cell = cells[1] if len(cells) > 1 else None
            if not event_cell:
                continue
            events = _extract_events_from_cell(event_cell)
            if not events:
                continue
            for iso_date in parse_date(date_str, default_year):
                for event_name in events:
                    event_name = clean_text(event_name)
                    if not event_name:
                        continue
                    key = (term, iso_date, event_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append({"term": term, "event_date": iso_date, "event_name": event_name, "source_url": source_url})
    return records


def scrape_url(session, config) -> tuple[list[dict], dict]:
    url, targets, base_name = config["url"], config["targets"], config.get("name", "Calendar")
    print(f"Scraping: {url}")
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return [], {}
    soup = BeautifulSoup(r.content, "html.parser")
    json_records = []
    excel_sheets = {}
    for target in targets:
        print(f"  Looking for section: {target}")
        records = extract_calendar_section(soup, target, url, max_tables=config.get("max_tables"))
        if not records:
            print(f"    -> WARNING: Could not find section or table for '{target}'")
            continue
        print(f"    -> Found {len(records)} events for '{target}'")
        json_records.extend(records)
        sheet_data = [{"Source Section": target, "Source URL": url, "Date": r["event_date"], "Event": r["event_name"]} for r in records]
        name = re.sub(r'[:\\/?*\[\]]', '', f"{base_name} {target.replace('Calendar Year', '').strip()}"[:31].strip())
        excel_sheets[name] = sheet_data
    return json_records, excel_sheets


def main():
    session = requests.Session()
    all_records = []
    all_sheets = {}
    for config in URLS_TO_SCRAPE:
        records, sheets = scrape_url(session, config)
        all_records.extend(records)
        all_sheets.update(sheets)
    seen = set()
    unique_records = []
    for r in all_records:
        key = (r["term"], r["event_date"], r["event_name"])
        if key not in seen:
            seen.add(key)
            unique_records.append(r)
    unique_records.sort(key=lambda x: (x["event_date"], x["term"], x["event_name"]))
    if unique_records or all_sheets:
        os.makedirs(DATA_DIR, exist_ok=True)
        if unique_records:
            print(f"Writing {len(unique_records)} events to {CALENDAR_JSON_FILE}...")
            with open(CALENDAR_JSON_FILE, "w", encoding="utf-8") as f:
                json.dump(unique_records, f, indent=2, ensure_ascii=False)
        if all_sheets:
            print(f"Writing {len(all_sheets)} sheets to {EXCEL_FILE}...")
            with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:
                for name, data in all_sheets.items():
                    if data:
                        df = pd.DataFrame(data)
                        print(f"  Writing sheet '{name}' with {len(df)} rows.")
                        df.to_excel(writer, sheet_name=name, index=False)
        print("Done.")
    else:
        print("No data extracted.")


def excel_to_json():
    """Read Excel, apply classification, write JSON only. Use after manual Excel edits."""
    xls = pd.read_excel(EXCEL_FILE, sheet_name=None)
    all_records = []
    for sheet_name, df in xls.items():
        df = df.fillna("")
        for _, row in df.iterrows():
            row_clean = row.dropna().to_dict()
            event_name = str(row_clean.get("Event", "") or row_clean.get("Holiday", "")).strip()
            date_val = str(row_clean.get("Date", "") or "").strip()
            source_url = str(row_clean.get("Source URL", "") or "").strip()
            source_section = str(row_clean.get("Source Section", sheet_name))
            if not event_name:
                continue
            all_records.append({"term": source_section or sheet_name, "event_date": date_val, "event_name": event_name, "source_url": source_url})
    seen = set()
    unique = []
    for r in all_records:
        key = (r["term"], r["event_date"], r["event_name"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    unique.sort(key=lambda x: (x["event_date"], x["term"], x["event_name"]))
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CALENDAR_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(unique)} events to {CALENDAR_JSON_FILE}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--from-excel":
        excel_to_json()
    else:
        main()
