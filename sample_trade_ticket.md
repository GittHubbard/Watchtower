# Trade Ticket — SPY — Playbook A

**Date**: 2026-05-27
**Setup**: Washout at Put Wall in Negative Gamma
**Signal strength**: 0.78 / 1.00
**Instrument**: Common shares.

> ⚠️ For human review. The system proposes; the PM disposes. No order has been placed.

## Sizing

| Field | Value |
|---|---|
| NAV | $100,000,000 |
| Risk budget | 50.00 bps = $500,000 |
| Tranche | 50.00% of full position |
| Entry (spot) | 500.00 |
| Stop (invalidation) | 495.00 |
| Risk / share | 5.00 |
| Suggested shares | 50,000.00 |
| Target 1 | 520.00 |
| Target 2 | 530.00 |

## Key levels

- **Spot**: 500.00
- **Put wall**: 495.00
- **Call wall**: 520.00
- **Gamma flip**: 510.00

## Why it fired

- **Regime**: deep_negative
- **pcr_volume_z**: 2.10
- **skew_25d_z**: 1.60
- **volume_z**: 2.30
- **vix_5d_change_pct**: 0.22

## Management notes

- Stop is the invalidation level above; re-evaluate as walls migrate (Stage 8).
- First scale at Target 1 (nearest call wall). Trail the remainder toward Target 2.
