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
Usage: python paper.py [r1|r2]  (default r1). State: paper_state_{cfg}.dec.json.
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

def main():
    import yfinance as yf
    cfg = sys.argv[1] if len(sys.argv) > 1 else "r1"
    if cfg not in CONFIGS:
        print(f"unknown config {cfg} (want r1|r2)")
        sys.exit(2)
    F = CONFIGS[cfg]
    sp, lp = f"paper_state_{cfg}.dec.json", f"paper_ledger_{cfg}.jsonl"
    st = {"cash": 100000.0, "positions": [], "staged": [], "pending_exits": [],
          "closed": [], "chain": "GENESIS", "frozen": None, "day": 0,
          "killed": None, "capped": 0}
    try:
        with open(sp) as f:
            st = json.load(f)
    except FileNotFoundError:
        print(f"genesis {cfg}: day 1 (Rs 100000, empty book)")
    for k in ("pending_exits", "capped"):
        if k not in st:
            st[k] = [] if k == "pending_exits" else 0
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
        st.setdefault("ledger", []).append({"t": "exit", "d": d, "s": p["symbol"],
            "px": round(px, 4), "net": round(net, 5), "why": hit["why"],
            "capped": capped})
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
        st["positions"].append({"symbol": o["symbol"], "entry_date": d,
                                "entry_px": px, "age": 0})
        st.setdefault("ledger", []).append({"t": "fill", "d": d, "s": o["symbol"],
                                            "px": round(px, 4)})
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
                                        "why": "stop" if stop_hit else ("take" if take_hit else "time")})

    # ---- (c) entry signals on today's CLOSE (dvol gate + depth rank, filled next open)
    # 20d dollar-volume per symbol over the trailing window, cross-sectional p10 gate.
    dv, valid = {}, []
    for s in closes.columns:
        try:
            hist = [(float(closes[s].iloc[k]), float(vols[s].iloc[k]))
                    for k in range(len(closes) - 20, len(closes))]
            hist = [(c, v) for c, v in hist if c == c and v == v and c > 0 and v > 0]
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
                sigs.append((ret, s))
    sigs.sort()
    exiting = {x["symbol"] for x in st["pending_exits"]}
    held = {p["symbol"] for p in st["positions"]} | exiting
    # Slots freed by tomorrow's exit fills are reusable by tomorrow's entry fills.
    free = F["slots"] - (len(st["positions"]) - len([x for x in st["pending_exits"]
          if any(p["symbol"] == x["symbol"] and p["entry_date"] == x["entry_date"]
                 for p in st["positions"])]))
    staged = []
    for _, s in sigs[:max(free, 0)]:
        if s not in held:
            staged.append({"symbol": s, "date": d})
            held.add(s)
    st["staged"] = staged
    for p in st["positions"]:
        if not any(x["symbol"] == p["symbol"] and x["entry_date"] == p["entry_date"]
                   for x in st["pending_exits"]):
            p["age"] += 1
    st["day"] += 1
    for row in st.get("ledger", []):
        if "chain" not in row:
            row["prev"] = st["chain"]
            st["chain"] = sha(row)
            row["chain"] = st["chain"]
    with open(sp, "w") as f:
        json.dump(st, f, indent=1)
    with open(lp, "a") as f:
        for row in st.get("ledger", []):
            if not row.get("_flushed"):
                f.write(json.dumps(row, sort_keys=True) + "\n")
                row["_flushed"] = True
    with open(sp, "w") as f:
        json.dump(st, f, indent=1)
    n = len(st["closed"])
    import statistics as S
    mu = sum(st["closed"]) / n if n else 0.0
    sd = S.pstdev(st["closed"]) if n > 1 else 0.0
    dd = 0.0
    peak, run = 0.0, 0.0
    for x in st["closed"]:
        run += x
        peak = max(peak, run)
        dd = max(dd, peak - run)
    kill = None
    if n >= 20 and mu < 0.0015:
        kill = f"expectancy {mu:+.4f} < +0.15% over {n}"
    if dd > 0.15:
        kill = f"drawdown {dd:.2%} > 15%"
    if kill and not st.get("killed"):
        st["killed"] = kill + f" on {d}"
        with open(sp, "w") as f:
            json.dump(st, f, indent=1)
    pend = f" pend_exit={len(st['pending_exits'])}"
    print(f"paper-{cfg} day {st['day']} ({d}): pos={len(st['positions'])} staged={len(staged)} "
          f"closed={n} mean={mu:+.5f} dd={dd:.3f} capped={st['capped']} chain={st['chain'][:12]}"
          + pend + (f" KILL-FROZEN: {kill}" if kill else ""))

if __name__ == "__main__":
    main()
