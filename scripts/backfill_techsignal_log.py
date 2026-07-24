"""Backfill the growing tech-signal parquet log (Mode A) from existing tickets.

Past EOD runs already wrote one technical-signal ticket per ticker per day to
``data/chains/ticker=<T>/tickets/<date>_<T>_techsignal.md``. Those tickets carry
the real, options-implied IV that was live at the time — so we can reconstruct
the growing ``data/techsignal_log.parquet`` from them directly, with no network
and no recomputation. This is "Mode A": gather the existing tickets and pack them
into the same log ``run_eod`` appends to daily.

Each row captures date, ticker, recommendation, score and the full ticket
markdown; the ``watchlist`` direction (``upward``/``downward``/``None``) is
recovered from whether the same ticket was routed into the Upward/Downward
watchlist folders. Writing is idempotent per date (re-running replaces a date's
rows rather than duplicating), via :func:`gamma_workflow.storage.append_techsignal_log`.

Usage:
    python scripts/backfill_techsignal_log.py                 # from 2026-06-04
    python scripts/backfill_techsignal_log.py --since 2026-06-10
    python scripts/backfill_techsignal_log.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import date as date_cls
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gamma_workflow import storage  # noqa: E402  (after sys.path setup)

DEFAULT_SINCE = date_cls(2026, 6, 4)

# <date>_<TICKER>_techsignal.md
_FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_([A-Z0-9.\-]+)_techsignal\.md$")
# **Recommendation**: HOLD  |  **Score**: +0.0   (header is always present)
_HEADER_RE = re.compile(
    r"\*\*Recommendation\*\*:\s*([A-Z_]+)\s*\|\s*\*\*Score\*\*:\s*([+-][\d.]+)"
)


def _watchlist_for(ticker: str, d: date_cls) -> str | None:
    """Return 'upward'/'downward'/None by checking which watchlist folder, if any,
    the ticket for (ticker, date) was routed into by the live run."""
    fname = f"{d.isoformat()}_{ticker}_techsignal.md"
    if (storage.UPWARD_WATCHLIST_DIR / d.isoformat() / fname).exists():
        return "upward"
    if (storage.DOWNWARD_WATCHLIST_DIR / d.isoformat() / fname).exists():
        return "downward"
    return None


def collect_records(chains_dir: Path, since: date_cls) -> dict[date_cls, list[dict]]:
    """Parse every techsignal ticket at/after ``since`` into per-date record lists."""
    by_date: dict[date_cls, list[dict]] = defaultdict(list)
    tickets = sorted(chains_dir.glob("ticker=*/tickets/*_techsignal.md"))
    for path in tickets:
        m = _FILENAME_RE.match(path.name)
        if not m:
            continue
        d = date_cls.fromisoformat(m.group(1))
        if d < since:
            continue
        ticker = m.group(2)
        content = path.read_text(encoding="utf-8")
        header = _HEADER_RE.search(content)
        if not header:
            raise ValueError(f"Missing Recommendation/Score header in {path}")
        by_date[d].append({
            "date": d.isoformat(),
            "ticker": ticker,
            "recommendation": header.group(1),
            "score": float(header.group(2)),
            "watchlist": _watchlist_for(ticker, d),
            "content": content,
        })
    return by_date


def backfill(since: date_cls = DEFAULT_SINCE, *, dry_run: bool = False) -> dict:
    """Backfill the tech-signal log from existing tickets. Returns a summary dict."""
    by_date = collect_records(storage.CHAINS_DIR, since)
    dates = sorted(by_date)
    total = sum(len(v) for v in by_date.values())

    if not dry_run:
        # Append chronologically so the parquet grows the same way daily runs build it.
        for d in dates:
            storage.append_techsignal_log(by_date[d], d)

    rec_counts: dict[str, int] = defaultdict(int)
    wl_counts: dict[str, int] = defaultdict(int)
    for records in by_date.values():
        for r in records:
            rec_counts[r["recommendation"]] += 1
            wl_counts[str(r["watchlist"])] += 1

    return {
        "dates": len(dates),
        "date_range": (dates[0].isoformat(), dates[-1].isoformat()) if dates else None,
        "rows": total,
        "recommendations": dict(sorted(rec_counts.items())),
        "watchlist": dict(sorted(wl_counts.items())),
        "dry_run": dry_run,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        type=date_cls.fromisoformat,
        default=DEFAULT_SINCE,
        help=f"Earliest ticket date to include (YYYY-MM-DD, default {DEFAULT_SINCE.isoformat()}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and summarize without writing the parquet log.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = backfill(args.since, dry_run=args.dry_run)
    verb = "Would write" if args.dry_run else "Wrote"
    print(f"{verb} {summary['rows']} ticket rows across {summary['dates']} dates "
          f"({summary['date_range'][0]} → {summary['date_range'][1]})"
          if summary["date_range"] else "No tickets found.")
    print(f"By recommendation: {summary['recommendations']}")
    print(f"By watchlist routing: {summary['watchlist']}")
    if not args.dry_run and summary["rows"]:
        print(f"\nLog: {storage.techsignal_log_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
