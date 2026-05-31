"""Scrape upcoming US macro events from Trading Economics' public calendar.

Writes to config/macro_events.csv in the format the gamma_workflow pipeline
expects:
    date,event
    YYYY-MM-DD,FOMC
    ...

Resolution rules:
  - Only US macro releases are kept (event-name allowlist below).
  - Dates are normalized to ISO 8601 (YYYY-MM-DD).
  - Column headers are lowercase ('date', 'event') to match the consumer
    at gamma_workflow/stage1_calendar.py::_load_macro.

Safety guards:
  - If the scrape returns fewer than MIN_EXPECTED_EVENTS rows, the script
    raises a RuntimeError WITHOUT writing the file. The workflow job goes
    red, you get an email, and the existing config/macro_events.csv stays
    intact rather than being overwritten by an empty or near-empty result.
  - Atomic write: builds the new CSV in a temp file first, then renames into
    place once we know it passed the sanity check. No partial files.

Run from the repo root:
    python scripts/macroevents_webscrape.py
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup


# ---- Configuration ---------------------------------------------------------

URL = "https://tradingeconomics.com/calendar"

# Where to write. Resolved relative to the repo root regardless of where the
# script is invoked from, so it works the same locally and in CI.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_PATH = REPO_ROOT / "config" / "macro_events.csv"

# Sanity check — if we get fewer than this many events, something is wrong
# (TE redesigned the page, Cloudflare blocked us, etc.). Refuse to overwrite.
# Tune up over time once you've seen what a healthy scrape returns.
MIN_EXPECTED_EVENTS = 5

# US macro releases that move SPY. Curated; expand as needed by inspecting
# what Trading Economics produces for each week.
# Match is case-insensitive on a normalized event-name string.
MACRO_ALLOWLIST = {
    # Fed
    "fed interest rate decision",
    "fomc press conference",
    "fomc minutes",
    "fed chair powell speech",
    "fed chair powell testimony",
    "fed press conference",
    # Inflation
    "inflation rate yoy",
    "inflation rate mom",
    "core inflation rate yoy",
    "core inflation rate mom",
    "cpi",
    "ppi",
    "producer price index",
    "core ppi yoy",
    "core ppi mom",
    "pce price index yoy",
    "core pce price index yoy",
    # Jobs
    "non farm payrolls",
    "nonfarm payrolls",
    "unemployment rate",
    "initial jobless claims",
    "average hourly earnings yoy",
    "average hourly earnings mom",
    # Growth & activity
    "gdp growth rate qoq",
    "gdp growth rate qoq adv",
    "gdp growth rate yoy",
    "retail sales mom",
    "retail sales yoy",
    "ism manufacturing pmi",
    "ism services pmi",
    "ism non-manufacturing pmi",
    "industrial production yoy",
    "industrial production mom",
    # Sentiment
    "consumer confidence",
    "michigan consumer sentiment",
    "michigan consumer sentiment prel",
}

# Trading Economics is global. We only want US events. The country column is
# detectable via a flag cell in the row (TE uses CSS classes like "calendar-iso us").
US_COUNTRY_TOKENS = {"us", "u.s.", "united states", "usa"}


# ---- Helpers ---------------------------------------------------------------

def _normalize_event_name(name: str) -> str:
    """Lowercase + collapse whitespace for allowlist matching."""
    return " ".join(name.lower().split())


def _parse_iso_date(raw: str) -> str | None:
    """Parse Trading Economics' long-form date header text (e.g.
    'Monday May 27 2026') to ISO 8601. Returns None if unparseable.

    Uses pandas because it handles a wide range of natural-language date
    formats more leniently than datetime.strptime."""
    try:
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date().isoformat()
    except Exception:
        return None


def _row_is_us(row) -> bool:
    """Inspect a calendar table row and decide whether it's a US event.

    Trading Economics marks country with a flag span like
    <span class="calendar-iso">US</span>. Some layouts put the ISO code in
    a class attribute, others as text content. We check both."""
    # Look at class attributes that mention 'us' or country tokens
    classes = " ".join(row.get("class", []) or [])
    if any(tok in classes.lower() for tok in US_COUNTRY_TOKENS):
        return True
    # Look at any element inside the row whose text is an ISO code
    iso_elem = row.find(class_="calendar-iso")
    if iso_elem:
        txt = iso_elem.get_text().strip().lower()
        if txt in US_COUNTRY_TOKENS:
            return True
    # Last resort: scan the row text for a 'US' token between tags
    # (kept conservative — we don't want to match strings like 'bus')
    raw_text = row.get_text(separator=" ").lower()
    for token in US_COUNTRY_TOKENS:
        if f" {token} " in f" {raw_text} ":
            return True
    return False


# ---- Main scrape -----------------------------------------------------------

def scrape_live_calendar() -> int:
    """Scrape, filter, and write. Returns the number of events written.

    Raises RuntimeError if the scrape returns fewer than MIN_EXPECTED_EVENTS
    rows — the existing CSV is left untouched in that case."""
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "desktop": True}
    )

    print(f"Connecting to {URL}...")
    response = scraper.get(URL, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to access {URL}. Status code: {response.status_code}"
        )

    soup = BeautifulSoup(response.text, "html.parser")

    calendar_data: list[dict] = []
    current_date: str | None = None   # ISO date string or None if unparseable
    seen = set()                       # for dedupe; TE sometimes lists revisions

    rows = soup.find_all("tr")
    print(f"Examining {len(rows)} table rows...")

    for row in rows:
        # --- Date header rows -------------------------------------------
        date_header = row.find("td", class_="table-header") or row.find(
            "th", class_="table-header"
        )
        if date_header:
            current_date = _parse_iso_date(date_header.get_text().strip())
            continue

        # Fallback: rows that lead with a day-of-week name
        first_text = row.get_text().strip()
        weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday")
        if first_text.startswith(weekdays):
            first_line = first_text.split("\n")[0].strip()
            current_date = _parse_iso_date(first_line)
            continue

        # --- Event rows -------------------------------------------------
        if current_date is None:
            # We haven't seen a parseable date header yet; skip orphan events
            continue

        event_link = row.find("a", href=True)
        if not event_link or "calendar" in event_link["href"]:
            continue

        event_name = event_link.get_text().strip()
        if not event_name:
            continue
        # Skip the column-header-y junk that sometimes appears in cell text
        if event_name.startswith(("Actual", "Previous", "Consensus", "Forecast")):
            continue

        # US filter
        if not _row_is_us(row):
            continue

        # Allowlist filter — case-insensitive, whitespace-tolerant match
        normalized = _normalize_event_name(event_name)
        if normalized not in MACRO_ALLOWLIST:
            continue

        # Dedupe on (date, normalized name)
        key = (current_date, normalized)
        if key in seen:
            continue
        seen.add(key)

        calendar_data.append({"date": current_date, "event": event_name})

    # --- Sanity check -----------------------------------------------------
    print(f"Scraped {len(calendar_data)} US macro events after filtering.")
    if len(calendar_data) < MIN_EXPECTED_EVENTS:
        raise RuntimeError(
            f"Only {len(calendar_data)} events scraped (expected >= "
            f"{MIN_EXPECTED_EVENTS}). Trading Economics format may have "
            f"changed, or the request was blocked. Refusing to overwrite "
            f"{OUTPUT_PATH} — existing file left intact."
        )

    # --- Atomic write -----------------------------------------------------
    # Write to a temp file alongside the target, then rename. If anything
    # goes wrong mid-write, the existing CSV is not corrupted.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=OUTPUT_PATH.parent, prefix=".macro_events_", suffix=".csv.tmp"
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "event"])
            writer.writeheader()
            # Sort by date for stable diffs in git
            for row in sorted(calendar_data, key=lambda r: (r["date"], r["event"])):
                writer.writerow(row)
        os.replace(tmp_path, OUTPUT_PATH)
    except Exception:
        # Clean up the temp file if we crashed before rename
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    print(f"Wrote {len(calendar_data)} events to {OUTPUT_PATH}.")
    return len(calendar_data)


if __name__ == "__main__":
    try:
        scrape_live_calendar()
    except Exception as e:
        # Re-raise as a non-zero exit so the GitHub Action job fails red,
        # rather than silently logging the error.
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
