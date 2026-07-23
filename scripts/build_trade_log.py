"""Build a trade log from the Downward and Upward technical-signal watchlists.

Each watchlist lives under ``data/<direction>-Watchlist/<YYYY-MM-DD>/`` and holds
one ``<date>_<TICKER>_techsignal.md`` file per name per day. This module walks
both trees, extracts the recommendation, score and confidence from every signal
file, and returns them as a tidy :class:`pandas.DataFrame` that a downstream step
can use to analyse the quality of the recommendations.

Public API
----------
``build_trade_log(down_dir=..., up_dir=...) -> pandas.DataFrame``
    One row per signal file with columns:

    ==================  ==========================================================
    column              meaning
    ==================  ==========================================================
    ``ticker``          symbol parsed from the filename (e.g. ``ADP``)
    ``date``            signal date (``datetime64[ns]``)
    ``watchlist``       ``"down"`` or ``"up"`` (which watchlist the file came from)
    ``recommendation``  ``BUY`` / ``STRONG_BUY`` / ``SELL`` / ``STRONG_SELL``
    ``score_capped``    displayed score, clamped to the [-10, +10] band
    ``score_raw``       uncapped score before clamping (higher resolution)
    ``confidence_pct``  confidence as an integer percentage (0-100)
    ==================  ==========================================================

Run as a script (``python scripts/build_trade_log.py``) to print a short summary
and preview; no file is written.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWN_DIR = REPO_ROOT / "data" / "Downward-Watchlist"
DEFAULT_UP_DIR = REPO_ROOT / "data" / "Upward-Watchlist"

# 2026-07-22_ADP_techsignal.md -> ("2026-07-22", "ADP")
_FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_([A-Z0-9.\-]+)_techsignal\.md$")

# **Recommendation**: STRONG_BUY  |  **Score**: +10.0
_HEADER_RE = re.compile(
    r"\*\*Recommendation\*\*:\s*([A-Z_]+)\s*\|\s*\*\*Score\*\*:\s*([+-][\d.]+)"
)

# ### Technical Signal: BUY (+4.1 / confidence 41%)
_CONFIDENCE_RE = re.compile(
    r"^###\s*Technical Signal:\s*[A-Z_]+\s*\(\s*[+-][\d.]+\s*/\s*confidence\s*(\d+)%\s*\)",
    re.MULTILINE,
)

# **Score**: +11.56 + +0.00 + +1.00 = +12.56 → **+10.00**
#   group 1 = raw (uncapped) sum, group 2 = capped (clamped) final
_FINAL_SCORE_RE = re.compile(r"=\s*([+-][\d.]+)\s*→\s*\*\*([+-][\d.]+)\*\*")

COLUMNS = [
    "ticker",
    "date",
    "watchlist",
    "recommendation",
    "score_capped",
    "score_raw",
    "confidence_pct",
]


def parse_signal_file(path: Path, watchlist: str) -> dict:
    """Extract the trade-log fields from a single ``*_techsignal.md`` file.

    ``watchlist`` is the direction label (``"down"`` / ``"up"``) to record for
    this file. Raises ``ValueError`` if the filename or any required field cannot
    be parsed, so malformed files surface loudly rather than silently vanishing.
    """
    name_match = _FILENAME_RE.match(path.name)
    if not name_match:
        raise ValueError(f"Unexpected signal filename: {path.name}")
    date_str, ticker = name_match.group(1), name_match.group(2)

    text = path.read_text(encoding="utf-8")

    header = _HEADER_RE.search(text)
    if not header:
        raise ValueError(f"Missing Recommendation/Score header in {path}")
    recommendation = header.group(1)
    score_capped = float(header.group(2))

    confidence = _CONFIDENCE_RE.search(text)
    if not confidence:
        raise ValueError(f"Missing confidence line in {path}")
    confidence_pct = int(confidence.group(1))

    final = _FINAL_SCORE_RE.search(text)
    if not final:
        raise ValueError(f"Missing final score breakdown in {path}")
    score_raw = float(final.group(1))

    return {
        "ticker": ticker,
        "date": date_str,
        "watchlist": watchlist,
        "recommendation": recommendation,
        "score_capped": score_capped,
        "score_raw": score_raw,
        "confidence_pct": confidence_pct,
    }


def _collect(directory: Path, watchlist: str) -> list[dict]:
    if not directory.exists():
        raise FileNotFoundError(f"Watchlist directory not found: {directory}")
    rows = [
        parse_signal_file(path, watchlist)
        for path in sorted(directory.rglob("*_techsignal.md"))
    ]
    return rows


def build_trade_log(
    down_dir: Path = DEFAULT_DOWN_DIR,
    up_dir: Path = DEFAULT_UP_DIR,
) -> pd.DataFrame:
    """Read both watchlists and return the trade log as a DataFrame.

    One row per signal file, sorted by date, then watchlist, then ticker. The
    ``date`` column is parsed to ``datetime64[ns]``; all other columns keep the
    dtypes described in the module docstring.
    """
    rows = _collect(down_dir, "down") + _collect(up_dir, "up")
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d")
    frame = frame.sort_values(["date", "watchlist", "ticker"]).reset_index(drop=True)
    return frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--down-dir", type=Path, default=DEFAULT_DOWN_DIR)
    parser.add_argument("--up-dir", type=Path, default=DEFAULT_UP_DIR)
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="How many preview rows to print (default: 10).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    frame = build_trade_log(args.down_dir, args.up_dir)

    print(f"Trade log: {len(frame)} signals "
          f"({(frame['watchlist'] == 'down').sum()} down, "
          f"{(frame['watchlist'] == 'up').sum()} up)")
    if not frame.empty:
        print(f"Date range: {frame['date'].min().date()} → {frame['date'].max().date()}")
        print("\nBy recommendation × watchlist:")
        print(frame.groupby(["watchlist", "recommendation"]).size().to_string())
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(f"\nPreview (first {args.rows} rows):")
            print(frame.head(args.rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
