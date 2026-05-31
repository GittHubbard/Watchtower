from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yfinance as yf
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYMBOLS_FILE = REPO_ROOT / "config" / "more_stocks.txt"
FALLBACK_SYMBOLS_FILE = REPO_ROOT / "config" / "stocks.txt"
DEFAULT_WATCHLIST_FILE = REPO_ROOT / "config" / "watchlist.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh market_cap_b values in watchlist.yaml from yfinance using the "
            "S&P 500 symbol file."
        )
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help=(
            "Path to the symbol list. Defaults to config/sp500_list.txt and falls back "
            "to config/sp500.txt if needed."
        ),
    )
    parser.add_argument(
        "--watchlist-file",
        type=Path,
        default=DEFAULT_WATCHLIST_FILE,
        help="Path to the YAML watchlist to update.",
    )
    parser.add_argument(
        "--kind",
        default="single_name",
        help="Kind assigned to any newly added ticker rows.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Seconds to wait between yfinance calls to reduce rate-limit pressure.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned updates without writing changes.",
    )
    return parser.parse_args()


def resolve_symbols_path(path: Path | None) -> Path:
    if path:
        if not path.exists():
            raise FileNotFoundError(f"Symbols file not found: {path}")
        return path

    if DEFAULT_SYMBOLS_FILE.exists():
        return DEFAULT_SYMBOLS_FILE
    if FALLBACK_SYMBOLS_FILE.exists():
        return FALLBACK_SYMBOLS_FILE
    raise FileNotFoundError(
        "No symbol list found. Expected config/sp500_list.txt or config/sp500.txt."
    )


def load_symbols(path: Path) -> list[str]:
    symbols: list[str] = []
    with path.open() as handle:
        for line in handle:
            symbol = line.strip()
            if not symbol or symbol.startswith("#"):
                continue
            symbols.append(symbol.split("#", 1)[0].strip())
    return symbols


def load_watchlist(path: Path) -> dict:
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Watchlist YAML must contain a mapping, got {type(data).__name__}")
    if "tickers" not in data:
        data["tickers"] = []
    return data


def fetch_market_cap_b(symbol: str) -> float:
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    market_cap = info.get("marketCap")
    if market_cap is None:
        raise KeyError(f"No marketCap returned for {symbol}")
    return round(float(market_cap) / 1_000_000_000, 6)


def main() -> int:
    args = parse_args()

    symbols_path = resolve_symbols_path(args.symbols_file)
    watchlist_path = args.watchlist_file

    symbols = load_symbols(symbols_path)
    watchlist = load_watchlist(watchlist_path)

    existing_rows = watchlist["tickers"]
    ticker_index = {}
    for index, row in enumerate(existing_rows):
        if isinstance(row, dict) and row.get("symbol"):
            ticker_index[row["symbol"]] = index

    updated = []
    failed = []

    for symbol in symbols:
        try:
            market_cap_b = fetch_market_cap_b(symbol)
        except Exception as exc:  # noqa: BLE001
            failed.append((symbol, str(exc)))
            continue

        if symbol in ticker_index:
            row = existing_rows[ticker_index[symbol]]
            if row.get("kind") is None:
                row["kind"] = args.kind
            old_market_cap = row.get("market_cap_b")
            row["market_cap_b"] = market_cap_b
            if old_market_cap != market_cap_b:
                updated.append((symbol, old_market_cap, market_cap_b))
        else:
            existing_rows.append({
                "symbol": symbol,
                "kind": args.kind,
                "market_cap_b": market_cap_b,
            })
            updated.append((symbol, None, market_cap_b))

        time.sleep(args.delay)

    if args.dry_run:
        print(f"Dry run for {watchlist_path}")
        print(f"Symbols processed: {len(symbols)}")
        print(f"Rows updated/added: {len(updated)}")
        if updated:
            print("Updates:")
            for symbol, old_value, new_value in updated:
                if old_value is None:
                    print(f"  + {symbol}: market_cap_b={new_value}")
                else:
                    print(f"  * {symbol}: {old_value} -> {new_value}")
        if failed:
            print("Failures:")
            for symbol, message in failed:
                print(f"  - {symbol}: {message}")
        return 0

    watchlist_path.parent.mkdir(parents=True, exist_ok=True)
    with watchlist_path.open("w") as handle:
        yaml.safe_dump(watchlist, handle, sort_keys=False)

    print(f"Updated {watchlist_path}")
    print(f"Symbols processed: {len(symbols)}")
    print(f"Rows updated/added: {len(updated)}")
    if failed:
        print(f"Failures: {len(failed)}")
        for symbol, message in failed:
            print(f"  - {symbol}: {message}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
