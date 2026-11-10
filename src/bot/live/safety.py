#!/usr/bin/env python3
"""Safety gates for the live pipeline. Every gate fails CLOSED (no orders).

Gates (all must pass; first failure aborts the whole batch):
  1. DRY_RUN unless LIVE_TRADING=1 is set explicitly. Default: print, don't send.
  2. CNC-only: any order with productType != CNC is rejected before send.
  3. Whitelist: every order symbol must be in tonight's engine decision set
     (entries staged + exits due). Nothing else can be ordered, ever.
  4. Value cap: order notional <= max_notional (slot size + rounding headroom).
  5. Kill-freeze: if the paper state carries frozen/kill flags, abort.
  6. Duplicate guard: stamp (session date) already executed -> zero orders.
  7. SELL coverage: every SELL must match a held position qty (no naked CNC
     short — which the broker would reject anyway, but we check first).
"""
import os

LIVE = os.environ.get("LIVE_TRADING") == "1"


class GateFail(RuntimeError):
    pass


def check(orders, *, decisions, positions, frozen, already_done, max_notional):
    """orders: list of dicts {side,symbol,security_id,qty,px}.
    decisions: {entries: {sym: {...}}, exits: {sym: {...}}} from tonight's engine.
    positions: {sym: qty_held}. frozen/already_done: bools. Returns orders."""
    if not LIVE:
        raise GateFail("DRY_RUN (set LIVE_TRADING=1 to arm)")
    allowed = set(decisions.get("entries", {})) | set(decisions.get("exits", {}))
    if frozen:
        raise GateFail("book frozen (kill-freeze set) — no orders")
    if already_done:
        raise GateFail("session already executed — duplicate guard")
    if not orders:
        return []  # NO-SIGNAL night: clean zero-order session, still valid
    held = dict(positions or {})
    for o in orders:
        if o.get("product") != "CNC":
            raise GateFail(f"non-CNC order rejected: {o}")
        if o["symbol"] not in allowed:
            raise GateFail(f"symbol not in tonight's decisions: {o['symbol']}")
        if not isinstance(o["qty"], int) or o["qty"] <= 0:
            raise GateFail(f"bad qty: {o}")
        notion = o["qty"] * float(o.get("px") or 0)
        if notion <= 0 or notion > max_notional:
            raise GateFail(f"notional {notion:.0f} over cap {max_notional:.0f}: {o['symbol']}")
        if o["side"] == "SELL" and held.get(o["symbol"], 0) < o["qty"]:
            raise GateFail(f"SELL without holding: {o['symbol']}")
        if o["side"] not in ("BUY", "SELL"):
            raise GateFail(f"bad side: {o}")
    return orders
