# IIT Calendar Scraper

Scrape IIT academic calendar and related pages to structured JSON and Excel.

## Prerequisites

- Python 3.8+

## Setup

```bash
cd Prototype
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

| Command | Description |
|---------|-------------|
| `python3 -m scrapers.calendar_scraper` | Scrape from web → JSON + Excel |
| `python3 -m scrapers.calendar_scraper --from-excel` | Regenerate JSON from Excel only (after manual edits) |

## Output

- `data/calendar_events.json` — Structured events (term, event_date, event_name, event_type, source_url)
- `data/university_calendar_data.xlsx` — Excel sheets by section

## Project layout

```
Prototype/
├── scrapers/
│   └── calendar_scraper.py
├── data/
│   ├── calendar_events.json
│   └── university_calendar_data.xlsx
├── requirements.txt
└── README.md
```

## License

MIT. See [LICENSE](LICENSE).
