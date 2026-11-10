#!/usr/bin/env python3
"""Live execution adapter — paper states in, broker AMO orders out. DRY-RUN by default.

Evening pipeline (one job, no morning cron, no always-on server):
  1. paper job finishes (11:30 UTC / 17:00 IST): states committed.
  2. THIS job: decrypt states -> diff tonight's decisions -> safety gates ->
     place CNC afterMarketOrder (amoTime OPEN) for NEXT open -> receipts log.
  3. Next morning: exchange opens the AMO orders at/near the open — the same
     fill the backtest models (t+1 open). No wake-up job required.

State contract (paper_state_{cfg}.dec.json): positions [{symbol, qty?, ...}],
staged [{symbol, sig_close}] (= tonight's BUY list), pending_exits [{symbol,
qty?}] (= tonight's SELL list), last_session, frozen, killed.
Qty booked: BUY = floor(slot_notional / sig_close) whole shares (0 -> skip with
log); SELL = held qty. slot_notional = nav / slots.

Idempotency: correlationId = q10-{cfg}-{date}-{SYM} (deterministic). Before
placing, the live path lists today's broker orders and skips IDs already
present — safe same-evening re-runs, never double-orders.

Secrets (CI only, never git): DHAN_CLIENT_ID + one of
  (a) DHAN_ACCESS_TOKEN (pasted daily), or
  (b) DHAN_API_KEY + DHAN_API_SECRET + DHAN_PIN + DHAN_TOTP_SECRET with
      --auto-token (pyotp generates the current TOTP in-job; 24h token).
Receipts (orderIds, statuses) print to stdout / file — no secrets in them.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dhan import DhanClient, generate_token_totp
from safety import check, GateFail

CFGS = ("r1", "r2", "t18", "s7y")
SLOTS = {"r1": 10, "r2": 10, "t18": 10, "s7y": 7}

# Scramble-quiet stdout (MISTAKES #6): the public Actions log carries COUNTS +
# gate verdicts + a sha of the intent ONLY. Symbols, qtys, prices, notionals
# live in the receipts file, which is AES-GCM-encrypted before commit — never
# in stdout. Same rule as tick.mjs's scrambled quiet branch.
import hashlib
QUIET = os.environ.get("LIVE_QUIET", "1") == "1"


def _gate_name(msg):
    m = str(msg)
    for prefix, name in [
        ("DRY_RUN", "DRY_RUN"),
        ("book frozen", "FROZEN"),
        ("session already executed", "DUPLICATE"),
        ("non-CNC", "NONCNC"),
        ("symbol not in tonight's decisions", "WHITELIST"),
        ("bad qty", "QTY"),
        ("notional", "NOTIONAL"),
        ("SELL without holding", "COVERAGE"),
        ("bad side", "SIDE"),
    ]:
        if m.startswith(prefix):
            return name
    return "GATE"


def _intent_sha(receipts):
    return hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest()[:16]


def load_state(dec_dir, cfg):
    with open(os.path.join(dec_dir, f"paper_state_{cfg}.dec.json")) as f:
        return json.load(f)


def build_orders(st, cfg, ids, max_notional):
    """Returns (decisions, orders). decisions feed the whitelist gate."""
    nav = float(st.get("nav", 100000.0))
    slots = SLOTS[cfg]
    slot_cash = nav / slots
    staged = st.get("staged", []) or []
    exits = st.get("pending_exits", []) or []
    held = {p["symbol"]: int(p.get("qty", 0)) for p in (st.get("positions", []) or [])}
    decisions = {"entries": {s["symbol"]: s for s in staged},
                 "exits": {e["symbol"] if isinstance(e, dict) else e: e for e in exits}}
    orders, skipped = [], []
    for s in staged:
        sym = s["symbol"]
        if sym not in ids:
            skipped.append((sym, "no-securityId-mapping"))
            continue
        qty = int(slot_cash // float(s["sig_close"]))
        if qty <= 0:
            skipped.append((sym, "whole-share-stranding"))
            continue
        orders.append({"side": "BUY", "symbol": sym, "security_id": ids[sym],
                       "qty": qty, "px": float(s["sig_close"]), "product": "CNC"})
    for e in exits:
        sym = e["symbol"] if isinstance(e, dict) else e
        if sym not in ids:
            skipped.append((sym, "no-securityId-mapping"))
            continue
        q = held.get(sym, 0)
        if q <= 0:
            skipped.append((sym, "nothing-held"))
            continue
        px = float((e.get("px") if isinstance(e, dict) else 0) or 0)
        orders.append({"side": "SELL", "symbol": sym, "security_id": ids[sym],
                       "qty": q, "px": px, "product": "CNC"})
    return decisions, orders, skipped, held


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dec-dir", default=".")
    ap.add_argument("--cfg", default="all")
    ap.add_argument("--ids", default="instrument_ids.json",
                    help="JSON {NSE symbol: Dhan securityId} (from Dhan instrument list)")
    ap.add_argument("--amo-time", default="OPEN")
    ap.add_argument("--auto-token", action="store_true")
    ap.add_argument("--receipts", default="live_receipts.json")
    a = ap.parse_args()

    cfgs = CFGS if a.cfg == "all" else (a.cfg,)
    try:
        with open(a.ids) as f:
            ids = json.load(f)
    except FileNotFoundError:
        ids = {}
    live = os.environ.get("LIVE_TRADING") == "1"

    client = None
    if live:
        cid = os.environ["DHAN_CLIENT_ID"]
        if a.auto_token:
            import pyotp
            totp = pyotp.TOTP(os.environ["DHAN_TOTP_SECRET"]).now()
            token = generate_token_totp(os.environ["DHAN_API_KEY"],
                                        os.environ["DHAN_API_SECRET"],
                                        cid, os.environ["DHAN_PIN"], totp)
        else:
            token = os.environ["DHAN_ACCESS_TOKEN"]
        client = DhanClient(cid, token, dry_run=False)
        seen = set()
        try:  # idempotency: never double-place a correlationId
            book = client._get("/orders")
            rows = book if isinstance(book, list) else book.get("data", book.get("orders", []))
            for o in rows if isinstance(rows, list) else []:
                c = o.get("correlationId") or o.get("correlation_id", "")
                if c:
                    seen.add(c)
        except Exception as e:
            print(f"WARN order-book pre-check failed ({e}); aborting (fail closed)")
            return 2
    else:
        client = DhanClient("DRYRUN", "none", dry_run=True)
        seen = set()

    all_receipts, rc = [], 0
    for cfg in cfgs:
        try:
            st = load_state(a.dec_dir, cfg)
        except FileNotFoundError:
            print(f"{cfg}: no state file, skip")
            continue
        date = st.get("last_session", "?")
        decisions, orders, skipped, held = build_orders(st, cfg, ids, max_notional=1.5 * st.get("nav", 100000.0) / SLOTS[cfg])
        if not QUIET:
            for s, why in skipped:
                print(f"{cfg} SKIP {s}: {why}")
        else:
            skip_reasons = {}
            for _, why in skipped:
                skip_reasons[why] = skip_reasons.get(why, 0) + 1
        try:
            check(orders, decisions=decisions, positions=held,
                  frozen=bool(st.get("frozen") or st.get("killed")),
                  already_done=False, max_notional=1.5 * st.get("nav", 100000.0) / SLOTS[cfg])
            gated = True
        except GateFail as g:
            if QUIET:
                print(f"{cfg} {date}: {_gate_name(g)} (orders={len(orders)} skipped={len(skipped)})")
            else:
                print(f"{cfg} {date}: GATE — {g}")
            if not live:
                if QUIET:
                    n_buy = sum(1 for o in orders if o["side"] == "BUY")
                    n_sell = len(orders) - n_buy
                    print(f"  {cfg} dry-run intent: buy={n_buy} sell={n_sell} (detail in encrypted receipts)")
                else:
                    for o in orders:  # dry-run: show intent even under DRY_RUN gate
                        print(f"  would {o['side']} {o['qty']}x {o['symbol']} CNC AMO-{a.amo_time}")
            continue
        for o in orders:
            cid_tag = f"q10-{cfg}-{date}-{o['symbol']}"[:30]
            if cid_tag in seen:
                print(f"{cfg} SKIP-PLACED {o['symbol']} (already in broker book)")
                continue
            r = client.place_cnc_amo(o["side"], o["security_id"], o["qty"],
                                     amo_time=a.amo_time, correlation_id=cid_tag)
            if QUIET:
                print(f"{cfg} {o['side']} ok ({len(all_receipts) + 1}/{len(orders)})")
            else:
                print(f"{cfg} {o['side']} {o['qty']}x {o['symbol']}: {r}")
            all_receipts.append({"cfg": cfg, "date": date, **o, "receipt": r})
    with open(a.receipts, "w") as f:
        json.dump(all_receipts, f, indent=1)
    print(f"receipts: {len(all_receipts)} -> {a.receipts} sha={_intent_sha(all_receipts)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
