"""Stage 7 — Sizing, stops, targets, and the trade ticket.

Turns a fired Stage 6 signal into a fully-specified, human-reviewable trade
ticket. No auto-execution — the ticket is a markdown file (date-stamped, per PM)
that the PM reads and acts on manually.

Sizing math (from the workflow spec):
    risk_capital   = NAV * (risk_per_trade_bps / 10_000)
    risk_per_share = entry - stop           (long; entry above stop)
    base_shares    = risk_capital / risk_per_share
    shares         = base_shares * suggested_size_pct   (the playbook's tranche)
    target_1       = nearest call wall above entry
    target_2       = secondary call wall, else entry + 2 * risk_per_share

Entry is taken as the current spot (we stage the order at market on the signal
day; the PM decides actual fill). For defined-risk call structures (Playbook C)
we still report the share-equivalent risk sizing so the PM can translate into a
spread of comparable risk; the ticket flags the instrument type explicitly.

Public API:
    build_ticket(signal, features, ticker_config, asof, thresholds) -> (dict, str)
        returns (ticket_dict, markdown_str)
    render_markdown(ticket_dict) -> str
"""
from __future__ import annotations
from datetime import date as date_cls
import logging

import numpy as np

from .config import load_thresholds

logger = logging.getLogger(__name__)


def _nearest_call_wall_above(entry: float, call_wall: float, call_wall_secondary: float) -> float:
    """First target = nearest call wall strictly above entry. Prefer primary,
    fall back to secondary, NaN if neither is above entry."""
    candidates = [w for w in (call_wall, call_wall_secondary)
                  if np.isfinite(w) and w > entry]
    return min(candidates) if candidates else float("nan")


def compute_sizing(
    signal: dict,
    features: dict,
    ticker_config: dict,
    thresholds: dict,
) -> dict:
    """Compute entry/stop/targets/shares. Returns a dict; share count may be NaN
    if the stop is invalid (non-positive risk per share)."""
    sizing_cfg = thresholds["sizing"]
    nav = float(sizing_cfg["nav_usd"])
    bps = float(ticker_config.get("risk_per_trade_bps", sizing_cfg["default_risk_per_trade_bps"]))

    entry = float(signal["key_levels"]["spot"])
    stop = float(signal["invalidation_level"])
    size_pct = float(signal["suggested_size_pct"])

    risk_capital = nav * (bps / 10_000.0)
    risk_per_share = entry - stop  # long: positive when entry above stop

    if not np.isfinite(risk_per_share) or risk_per_share <= 0:
        logger.warning(
            "Invalid risk_per_share for %s playbook %s: entry=%.4f stop=%.4f",
            features.get("ticker"), signal["playbook_id"], entry, stop,
        )
        shares = float("nan")
        base_shares = float("nan")
    else:
        base_shares = risk_capital / risk_per_share
        shares = base_shares * size_pct

    call_wall = float(signal["key_levels"].get("call_wall", float("nan")))
    call_wall_sec = float(features.get("call_wall_secondary", float("nan")))
    target_1 = _nearest_call_wall_above(entry, call_wall, call_wall_sec)

    # Target 2: secondary call wall (if above target_1) else 2R
    two_r = entry + 2.0 * risk_per_share if np.isfinite(risk_per_share) and risk_per_share > 0 else float("nan")
    if np.isfinite(call_wall_sec) and call_wall_sec > target_1:
        target_2 = call_wall_sec
    else:
        target_2 = two_r

    return {
        "entry": entry,
        "stop": stop,
        "risk_capital": risk_capital,
        "risk_per_share": risk_per_share,
        "base_shares": base_shares,
        "shares": shares,
        "size_pct": size_pct,
        "target_1": target_1,
        "target_2": target_2,
        "risk_per_trade_bps": bps,
        "nav_usd": nav,
    }


def build_ticket(
    signal: dict,
    features: dict,
    ticker_config: dict,
    asof: date_cls,
    thresholds: dict | None = None,
) -> tuple[dict, str]:
    """Assemble the complete trade ticket dict and its markdown rendering."""
    thresholds = thresholds or load_thresholds()
    sizing = compute_sizing(signal, features, ticker_config, thresholds)

    ticket = {
        "asof": asof.isoformat(),
        "ticker": features.get("ticker"),
        "playbook_id": signal["playbook_id"],
        "playbook_name": signal["playbook_name"],
        "instrument": signal["instrument"],
        "signal_strength": signal["signal_strength"],
        "suggested_size_pct": signal["suggested_size_pct"],
        **sizing,
        "key_levels": signal["key_levels"],
        "gates_passed": signal["gates_passed"],
        "regime": features.get("regime"),
        "regime_transition": features.get("regime_transition"),
        # Per-signal notes (e.g. IV/HV strength reducer, Feature 1).
        "notes": list(signal.get("notes", [])),
        # Stage 5.5 commentary — transient (not in the parquet schema); rendered if present.
        "technical_commentary": features.get("technical_commentary"),
        "analyst_commentary": features.get("analyst_commentary"),
    }
    return ticket, render_markdown(ticket)


def _fmt(v, pct=False, money=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if pct:
        return f"{v * 100:.2f}%"
    if money:
        return f"${v:,.0f}"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


def render_markdown(ticket: dict) -> str:
    """Render the ticket as a markdown document for PM review."""
    kl = ticket["key_levels"]
    gates = ticket["gates_passed"]
    gates_lines = "\n".join(f"- **{k}**: {_fmt(v) if isinstance(v, (int, float)) else v}"
                            for k, v in gates.items() if k != "regime")

    instrument_note = (
        "Defined-risk call structure (e.g. call debit spread). Share count below "
        "is the risk-equivalent sizing — translate into a spread of comparable risk."
        if ticket["instrument"] == "defined_risk_calls"
        else "Common shares."
    )

    transition_line = (
        f"\n- **Regime transition**: {ticket['regime_transition']}"
        if ticket.get("regime_transition") else ""
    )

    # Per-signal notes (e.g. the IV/HV strength reducer). Rendered as a callout.
    notes = [n for n in ticket.get("notes", []) if n]
    notes_section = ""
    if notes:
        notes_section = "\n## Notes\n\n" + "\n".join(f"> ⚠️ {n}" for n in notes) + "\n"

    # Stage 5.5 commentary (technicals + analyst evolution). Optional free text.
    commentary_parts = []
    if ticket.get("technical_commentary"):
        commentary_parts.append(f"**Technicals**: {ticket['technical_commentary']}")
    if ticket.get("analyst_commentary"):
        commentary_parts.append(f"**Analyst view**: {ticket['analyst_commentary']}")
    commentary_section = ""
    if commentary_parts:
        commentary_section = "\n## Commentary\n\n" + "\n\n".join(commentary_parts) + "\n"

    md = f"""# Trade Ticket — {ticket['ticker']} — Playbook {ticket['playbook_id']}

**Date**: {ticket['asof']}
**Setup**: {ticket['playbook_name']}
**Signal strength**: {ticket['signal_strength']:.2f} / 1.00
**Instrument**: {instrument_note}

> ⚠️ For human review. The system proposes; the PM disposes. No order has been placed.

## Sizing

| Field | Value |
|---|---|
| NAV | {_fmt(ticket['nav_usd'], money=True)} |
| Risk budget | {_fmt(ticket['risk_per_trade_bps'])} bps = {_fmt(ticket['risk_capital'], money=True)} |
| Tranche | {_fmt(ticket['suggested_size_pct'], pct=True)} of full position |
| Entry (spot) | {_fmt(ticket['entry'])} |
| Stop (invalidation) | {_fmt(ticket['stop'])} |
| Risk / share | {_fmt(ticket['risk_per_share'])} |
| Suggested shares | {_fmt(ticket['shares'])} |
| Target 1 | {_fmt(ticket['target_1'])} |
| Target 2 | {_fmt(ticket['target_2'])} |

## Key levels

- **Spot**: {_fmt(kl.get('spot'))}
- **Put wall**: {_fmt(kl.get('put_wall'))}
- **Call wall**: {_fmt(kl.get('call_wall'))}
- **Gamma flip**: {_fmt(kl.get('flip'))}

## Why it fired

- **Regime**: {ticket['regime']}{transition_line}
{gates_lines}
{notes_section}{commentary_section}
## Management notes

- Stop is the invalidation level above; re-evaluate as walls migrate (Stage 8).
- First scale at Target 1 (nearest call wall). Trail the remainder toward Target 2.
"""
    return md
