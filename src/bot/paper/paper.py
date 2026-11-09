#!/usr/bin/env python3
"""Paper-trade runner v2 — frozen R1/R2 forward pair, backtest-faithful.

Close-decision / next-open-fill (mirrors src/bot/backtest.py compute_mr_wants +
backtest_symbol order engine):
  - entry signal on bar t close (R_drop_days <= drop + cross-sectional dvol gate),
    filled at t+1 OPEN;
  - exit signal on close (take/stop/time), filled at NEXT OPEN;
  - winners are NOT capped intraday at +3%: the backtested edge lives in the
    overnight continuation after the signal close (past-data replay 2026-11-08:
    intraday-limit exits printed mean -0.08%/trade vs backtest +0.77% on the
    same bars; close/next-open replay recovers +0.75%).
Usage: python paper.py [r1|r2|t18]  (default r1). State: paper_state_{cfg}.dec.json.
Holiday guard: latest bar must be today-IST or the script logs NO_SESSION and
exits 0 without touching state. Kill-freeze: mean<+0.15% over >=20 closed OR
DD>15% -> book frozen, logged daily.
"""
import sys, json, hashlib, datetime as dt

CONFIGS = {
    "r1": {"drop": -6.5, "drop_days": 3, "take": 3.0, "stop": 8.0, "hold": 5,
           "slots": 10, "friction": 0.003, "universe": "nifty50-49", "variant": "sharp-r1"},
    "r2": {"drop": -7.0, "drop_days": 3, "take": 3.0, "stop": 8.0, "hold": 5,
           "slots": 10, "friction": 0.003, "universe": "nifty50-49", "variant": "sharp-r2"},
    "t18": {"drop": -6.5, "drop_days": 3, "take": 3.0, "stop": 10.0, "hold": 10,
            "slots": 10, "friction": 0.003, "universe": "nifty50-49", "variant": "sharp-t18"},
    # s7y: honest-book forward (Batch 13 sweet spot + Batch 15 honest yield).
    # Genesis Day 1 Monday; trio (r1/r2/t18, no yield) untouched beside it.
    "s7y": {"drop": -6.5, "drop_days": 3, "take": 3.0, "stop": 10.0, "hold": 10,
            "slots": 7, "friction": 0.003, "universe": "nifty50-49", "variant": "sharp-s7y",
            "yield": True},
}
UNIVERSE = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","ITC.NS",
    "SBIN.NS","BHARTIARTL.NS","LICI.NS","HINDUNILVR.NS",
    "KOTAKBANK.NS","AXISBANK.NS","LT.NS","ASIANPAINT.NS","MARUTI.NS",
    "SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","NTPC.NS","POWERGRID.NS",
    "ONGC.NS","TATASTEEL.NS","JSWSTEEL.NS","ADANIENT.NS","ADANIPORTS.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","HCLTECH.NS","TECHM.NS","WIPRO.NS",
    "NESTLEIND.NS","BRITANNIA.NS","DIVISLAB.NS","DRREDDY.NS","CIPLA.NS",
    "EICHERMOT.NS","HEROMOTOCO.NS","BAJAJ-AUTO.NS","GRASIM.NS","HINDALCO.NS",
    "COALINDIA.NS","BPCL.NS","TATACONSUM.NS","APOLLOHOSP.NS","TRENT.NS",
    "BEL.NS","SHRIRAMFIN.NS","SBILIFE.NS","HDFCLIFE.NS",
]

TRADE_CAP = 1.0  # same data-hygiene cap as the locked backtest engine

def sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()

def cap(gross):
    if gross > TRADE_CAP:
        return True, TRADE_CAP
    if gross < -0.999:
        return True, -0.999
    return False, gross

# Honest-book idle yield (Batch 15 proxy, s7y only): RBI Bank Rate minus
# 150bp liquid-fund drag. Same change-point table as backtest.py
# HIST_YIELD_POINTS; trio books accrue nothing (record semantics preserved).
S7Y_YIELD_POINTS = [
    ("2015-01", 0.0725), ("2015-03", 0.0700), ("2015-06", 0.0675),
    ("2015-10", 0.0625), ("2016-04", 0.0550), ("2016-10", 0.0525),
    ("2017-04", 0.0500), ("2017-08", 0.0475), ("2018-06", 0.0500),
    ("2018-08", 0.0525), ("2019-02", 0.0500), ("2019-04", 0.0475),
    ("2019-06", 0.0450), ("2019-08", 0.0415), ("2019-10", 0.0390),
    ("2020-03", 0.0315), ("2020-05", 0.0275), ("2022-05", 0.0315),
    ("2022-06", 0.0365), ("2022-08", 0.0415), ("2022-09", 0.0465),
    ("2022-12", 0.0500), ("2023-02", 0.0525), ("2025-02", 0.0500),
    ("2025-04", 0.0475), ("2025-06", 0.0425),
]

def s7y_annual_rate(ym):
    r = S7Y_YIELD_POINTS[0][1]
    for m, v in S7Y_YIELD_POINTS:
        if m <= ym:
            r = v
        else:
            break
    return r

def main():
    import yfinance as yf
    cfg = sys.argv[1] if len(sys.argv) > 1 else "r1"
    if cfg not in CONFIGS:
        print(f"unknown config {cfg} (want r1|r2|t18|s7y)")
        sys.exit(2)
    F = CONFIGS[cfg]
    sp, lp = f"paper_state_{cfg}.dec.json", f"paper_ledger_{cfg}.jsonl"
    st = {"cash": 100000.0, "positions": [], "staged": [], "pending_exits": [],
          "closed": [], "chain": "GENESIS", "frozen": None, "day": 0,
          "killed": None, "capped": 0, "nav": 100000.0, "nav_peak": 100000.0}
    try:
        with open(sp) as f:
            st = json.load(f)
    except FileNotFoundError:
        print(f"genesis {cfg}: day 1 (Rs 100000, empty book)")
    for k, v in (("pending_exits", []), ("capped", 0),
                 ("nav", 100000.0), ("nav_peak", 100000.0),
                 ("yield_total", 0.0), ("last_session", None)):
         if k not in st:
             st[k] = v
    fh = sha(F)
    if st.get("frozen") in (None, fh):
        st["frozen"] = fh
    else:
        print(f"CONFIG DRIFT {cfg}: record void — abort")
        sys.exit(3)
    if st.get("killed"):
        print(f"KILLED-FROZEN {cfg}: {st['killed']} — no new entries")
        return
    ist_today = (dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)).date().isoformat()
    data = yf.download(UNIVERSE, period="3mo", auto_adjust=True, progress=False, threads=True)
    closes, opens = data["Close"], data["Open"]
    vols = data["Volume"]
    if len(closes) < 30:
        print("fetch too short (need 30+ bars for the 20d dvol gate) — abort without touching state")
        sys.exit(2)
    last = closes.index[-1]
    d = last.date().isoformat()
    if d != ist_today:
        # No fresh bar (weekend/holiday or feed lag): do NOT treat stale close as new
        print(f"NO_SESSION {cfg}: latest bar {d} != today-ist {ist_today} — state untouched")
        return
    if st.get("last_session") == d:
        # Same-day re-dispatch (cron + manual, or retry): session already banked.
        print(f"DUPLICATE_SESSION {cfg}: {d} already banked as day {st.get('day')} — state untouched")
        return
    if F.get("yield"):
        # Honest-book accrual on idle cash at today's open (s7y only).
        r = s7y_annual_rate(d[:7])
        dyf = (1.0 + r) ** (1.0 / 252.0) - 1.0 if r and r > 0 else 0.0
        if dyf:
            yacc = st["cash"] * dyf
            st["cash"] *= (1.0 + dyf)
            st["yield_total"] = st.get("yield_total", 0.0) + yacc

    def pxat(frame, sym, loc):
        try:
            v = float(frame[sym].loc[loc])
            return v if v == v and v > 0 else None
        except Exception:
            return None

    # ---- (a) fills at today's OPEN: yesterday's exit signals + staged entries
    # Exit fills first (frees slots for same-open entries, like the backtest ledger).
    still_open = []
    for p in st["positions"]:
        hit = next((x for x in st["pending_exits"] if x["symbol"] == p["symbol"]
                    and x["entry_date"] == p["entry_date"]), None)
        if hit is None:
            still_open.append(p)
            continue
        px = pxat(opens, p["symbol"], last)
        if px is None:
            still_open.append(p)  # halted: retain, retry next session
            continue
        capped, gross = cap(px / p["entry_px"] - 1.0)
        net = (1 + gross) * (1 - F["friction"]) - 1.0
        st["cash"] *= 1 + net / F["slots"]
        st["closed"].append(net)
        st["capped"] += 1 if capped else 0
        sq = hit.get("sig_close")
        st.setdefault("ledger", []).append({"t": "exit", "d": d, "s": p["symbol"],
            "px": round(px, 4), "net": round(net, 5), "why": hit["why"],
            "capped": capped,
            "sig_px": sq, "gap": round(px / sq - 1.0, 5) if sq else None})
    st["positions"] = still_open
    st["pending_exits"] = [x for x in st["pending_exits"]
                           if any(p["symbol"] == x["symbol"] and p["entry_date"] == x["entry_date"]
                                  for p in still_open)]
    for o in st["staged"]:
        if any(p["symbol"] == o["symbol"] for p in st["positions"]):
            continue
        px = pxat(opens, o["symbol"], last)
        if px is None:
            continue  # halted: drop single-day stale stage (re-signal tomorrow if set up)
        notion = st["nav"] / F["slots"]
        st["positions"].append({"symbol": o["symbol"], "entry_date": d,
                                "entry_px": px, "age": 0, "notional": notion})
        sq = o.get("sig_close")
        st.setdefault("ledger", []).append({"t": "fill", "d": d, "s": o["symbol"],
            "px": round(px, 4),
            "sig_px": sq, "slip": round(px / sq - 1.0, 5) if sq else None})
    st["staged"] = []

    # ---- (b) exit SIGNALS on today's CLOSE (filled at NEXT open)
    for p in st["positions"]:
        cl = pxat(closes, p["symbol"], last)
        if cl is None:
            continue
        take_hit = cl >= p["entry_px"] * (1 + F["take"] / 100.0)
        stop_hit = cl <= p["entry_px"] * (1 - F["stop"] / 100.0)
        expired = p["age"] >= F["hold"]
        if take_hit or stop_hit or expired:
            st["pending_exits"].append({"symbol": p["symbol"],
                                        "entry_date": p["entry_date"], "date": d,
                                        "sig_close": round(cl, 4),
                                        "why": "stop" if stop_hit else ("take" if take_hit else "time")})

    # ---- (c) entry signals on today's CLOSE (dvol gate + depth rank, filled next open)
    # 20d dollar-volume per symbol over the trailing window, cross-sectional p10 gate.
    dv, valid = {}, []
    for s in closes.columns:
        try:
            hist = [(float(closes[s].iloc[k]), float(vols[s].iloc[k]))
                    for k in range(len(closes) - 20, len(closes))]
            # NaN-check ONLY, exactly like the backtest's (closes*vols).rolling(20)
            # (a zero-volume bar counts; requiring v>0 fail-shut whole windows —
            # proven by parity forensics 2026-11-08: replay missed 2025-04-08-style
            # panic entries the backtest took).
            hist = [(c, v) for c, v in hist if c == c and v == v]
            if len(hist) >= 20:
                m = sum(c * v for c, v in hist) / len(hist)
                dv[s] = m
                valid.append(m)
        except Exception:
            continue
    valid.sort()
    # 10th percentile (linear interpolation, pandas-quantile style)
    if valid:
        q = 0.10 * (len(valid) - 1)
        lo, hi = int(q), min(int(q) + 1, len(valid) - 1)
        thresh = valid[lo] + (valid[hi] - valid[lo]) * (q - lo)
    else:
        thresh = None
    sigs = []
    if thresh is not None and len(closes) > F["drop_days"]:
        for s in closes.columns:
            try:
                px = float(closes[s].iloc[-1])
                then = float(closes[s].iloc[-1 - F["drop_days"]])
            except Exception:
                continue
            if not (px == px and then == then and then > 0):
                continue
            ret = px / then - 1.0
            if ret <= F["drop"] / 100.0 and s in dv and dv[s] >= thresh:
                sigs.append((ret, s, px))
    sigs.sort(key=lambda x: x[1])  # symbol order = backtest allocate_nav's
    # (entry-date, symbol) tie-break, verbatim. NOT depth-ranked: R1 was
    # validated with unranked placement (Batch 7 R5 tested ranking separately
    # and starved). Depth-rank staging was a v2 deviation; parity replay
    # measured it at 83% key agreement, exact prices on matches.
    # #56-fix (pre-Session-2 slot-allocation correction): filter held BEFORE
    # the free-slot cap, like allocate_nav (which iterates ALL of a day's
    # cands and skips only when free_slots hits 0 — same-symbol overlap is
    # impossible there by the per-symbol pos gate). The old code sliced
    # sigs[:free] first: runner signals have no pos gate, so held symbols
    # re-signal while dropping and occupied slice slots that were then
    # skipped — systematic understaging concentrated on crowded down days
    # (K-free sim on banked prices: 632 wasted slots / 418 days; fix
    # recovers +67 fills, 1179 -> 1246 vs backtest-taken 1238).
    exiting = {x["symbol"] for x in st["pending_exits"]}
    held = {p["symbol"] for p in st["positions"]} | exiting
    # Slots freed by tomorrow's exit fills are reusable by tomorrow's entry fills.
    free = F["slots"] - (len(st["positions"]) - len([x for x in st["pending_exits"]
          if any(p["symbol"] == x["symbol"] and p["entry_date"] == x["entry_date"]
                 for p in st["positions"])]))
    staged = []
    for _, s, sc in sigs:
        if len(staged) >= max(free, 0):
            break
        if s not in held:
            staged.append({"symbol": s, "date": d, "sig_close": round(sc, 4)})
            held.add(s)
    st["staged"] = staged
    for p in st["positions"]:
        if not any(x["symbol"] == p["symbol"] and x["entry_date"] == p["entry_date"]
                   for x in st["pending_exits"]):
            p["age"] += 1
    st["day"] += 1
    st["last_session"] = d
    for row in st.get("ledger", []):
        if "chain" not in row:
            row["prev"] = st["chain"]
            st["chain"] = sha(row)
            row["chain"] = st["chain"]
    n = len(st["closed"])
    import statistics as S
    mu = sum(st["closed"]) / n if n else 0.0
    # Portfolio NAV at today's close (cash + open-position MTM), the same
    # ledger semantics as the backtest slot book. The kill bar is drawn on
    # THIS, never on the raw cumsum of per-trade nets (which runs ~slots x
    # hotter and false-triggers on healthy books — caught by parity replay).
    mtm = 0.0
    for p in st["positions"]:
        cl = pxat(closes, p["symbol"], last)
        if cl is None:
            continue
        mtm += (cl / p["entry_px"] - 1.0) * p.get("notional", st["nav"] / F["slots"])
    st["nav"] = st["cash"] + mtm
    st["nav_peak"] = max(st["nav_peak"], st["nav"])
    dd = (st["nav_peak"] - st["nav"]) / st["nav_peak"]
    kill = None
    if n >= 20 and mu < 0.0015:
        kill = f"expectancy {mu:+.4f} < +0.15% over {n}"
    if dd > 0.15:
        kill = f"drawdown {dd:.2%} > 15% (NAV {st['nav']:,.0f} vs peak {st['nav_peak']:,.0f})"
    if kill and not st.get("killed"):
        st["killed"] = kill + f" on {d}"
    # Single state write, AFTER every derived field is final (#43 fix:
    # nav/nav_peak/kill used to compute after the write and never persist).
    # Fills/signals/exits above are byte-identical to the frozen runner.
    with open(lp, "a") as f:
        for row in st.get("ledger", []):
            if not row.get("_flushed"):
                f.write(json.dumps(row, sort_keys=True) + "\n")
                row["_flushed"] = True
    with open(sp, "w") as f:
        json.dump(st, f, indent=1)
    pend = f" pend_exit={len(st['pending_exits'])}"
    print(f"paper-{cfg} day {st['day']} ({d}): pos={len(st['positions'])} staged={len(staged)} "
          f"closed={n} mean={mu:+.5f} dd={dd:.3f} nav={st['nav']:,.0f} capped={st['capped']} chain={st['chain'][:12]}"
          + (f" yld={st.get('yield_total', 0.0):,.0f}" if F.get("yield") else "")
          + pend + (f" KILL-FROZEN: {kill}" if kill else ""))

if __name__ == "__main__":
    main()
