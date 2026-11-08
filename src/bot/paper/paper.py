#!/usr/bin/env python3
"""Paper-trade runner v1 — frozen R1/R2 forward pair (NOT YET LAUNCHED).
Usage: python paper.py [r1|r2]  (default r1). State: paper_state_{cfg}.dec.json.
Holiday guard: latest bar must be today-IST or the script logs NO_SESSION and
exits 0 without touching state. Fills at t+1 OPEN (auto_adjust = split-adjusted).
Same-day touch: STOP takes absolute precedence (conservative). Kill-freeze:
mean<+0.15% over >=20 closed OR DD>15% -> book frozen, logged daily.
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

def sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()

def main():
    import yfinance as yf
    cfg = sys.argv[1] if len(sys.argv) > 1 else "r1"
    if cfg not in CONFIGS:
        print(f"unknown config {cfg} (want r1|r2)")
        sys.exit(2)
    F = CONFIGS[cfg]
    sp, lp = f"paper_state_{cfg}.dec.json", f"paper_ledger_{cfg}.jsonl"
    st = {"cash": 100000.0, "positions": [], "staged": [], "closed": [],
          "chain": "GENESIS", "frozen": None, "day": 0, "killed": None}
    try:
        with open(sp) as f:
            st = json.load(f)
    except FileNotFoundError:
        print(f"genesis {cfg}: day 1 (Rs 100000, empty book)")
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
    closes = data["Close"].dropna(axis=1, how="all")
    opens, highs, lows = data["Open"], data["High"], data["Low"]
    if len(closes) < 10:
        print("fetch too short — abort without touching state")
        sys.exit(2)
    last = closes.index[-1]
    d = last.date().isoformat()
    if d != ist_today and last.date().weekday() >= 5 - 1 and False:
        pass
    if d != ist_today:
        # No fresh bar (weekend/holiday or feed lag): do NOT treat stale close as new
        print(f"NO_SESSION {cfg}: latest bar {d} != today-ist {ist_today} — state untouched")
        return
    for o in st["staged"]:
        try:
            px = float(opens[o["symbol"]].loc[last])
        except Exception:
            continue
        st["positions"].append({"symbol": o["symbol"], "entry_date": d,
                                "entry_px": px, "bars": 0,
                                "target": px * (1 + F["take"] / 100.0),
                                "stop": px * (1 - F["stop"] / 100.0)})
        st["ledger"] = st.get("ledger", [])
        st["ledger"].append({"t": "fill", "d": d, "s": o["symbol"], "px": px})
    st["staged"] = []
    keep = []
    for p in st["positions"]:
        try:
            hi = float(highs[p["symbol"]].loc[last])
            lo = float(lows[p["symbol"]].loc[last])
            cl = float(closes[p["symbol"]].loc[last])
        except Exception:
            keep.append(p)
            continue
        p["bars"] += 1
        stop_hit = lo <= p["stop"]
        take_hit = hi >= p["target"]
        expired = p["bars"] >= F["hold"]
        if stop_hit or take_hit or expired:
            px = p["stop"] if stop_hit else (p["target"] if take_hit else cl)
            gross = px / p["entry_px"] - 1.0
            net = (1 + gross) * (1 - F["friction"]) - 1.0
            st["cash"] *= 1 + net / F["slots"]
            st["closed"].append(net)
            st["ledger"].append({"t": "exit", "d": d, "s": p["symbol"], "px": px,
                                 "net": round(net, 5),
                                 "why": "stop" if stop_hit else ("take" if take_hit else "time")})
        else:
            keep.append(p)
    st["positions"] = keep
    sigs = []
    for s in closes.columns:
        try:
            px = float(closes[s].iloc[-1])
            then = float(closes[s].iloc[-1 - F["drop_days"]])
            if then and then == then and px / then - 1.0 <= F["drop"] / 100.0:
                sigs.append((px / then - 1.0, s))
        except Exception:
            continue
    sigs.sort()
    free = F["slots"] - len(st["positions"])
    held = {p["symbol"] for p in st["positions"]}
    staged = []
    for _, s in sigs[:max(free, 0)]:
        if s not in held:
            staged.append({"symbol": s, "date": d})
            held.add(s)
    st["staged"] = staged
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
    print(f"paper-{cfg} day {st['day']} ({d}): pos={len(st['positions'])} staged={len(staged)} "
          f"closed={n} mean={mu:+.5f} dd={dd:.3f} chain={st['chain'][:12]}"
          + (f" KILL-FROZEN: {kill}" if kill else ""))

if __name__ == "__main__":
    main()
