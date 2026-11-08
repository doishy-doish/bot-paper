#!/usr/bin/env python3
"""K-neutral parity replay: runs the ACTUAL paper.main() day-loop over history.

NOT a trial: reuses frozen CONFIGS/UNIVERSE/cap/sha from paper.py unmodified;
only the fetch window and wall clock are faked. Pre-registered rule (standing):
the outcome changes NOTHING about parameters. MATCH = runner is gold.
MISMATCH = runner bug to fix (specification fidelity). Never tuning input,
never forward evidence. K stays 56, trial freeze intact.

Usage: python replay_check.py --ref-trades /tmp/r1_trades.csv --ref-summary /tmp/r1.json --out /tmp/replay_summary.json
Compares replay ledger exits vs the reference backtest trade list on
(symbol, entry-date, exit-date) keys + price agreement, plus summary
(N / mean / win) within refetch tolerances.
"""
import os, sys, json, types, argparse, tempfile
import datetime as _dt

KEY_RATE_BAR = 0.98   # >=98% of trades share exact (symbol, entry, exit) keys
N_TOL = 5             # |replay N - ref N| <= 5 (slot-ordering micro-diffs)
MEAN_TOL = 0.0010     # |mean diff| <= 10bp
WIN_TOL = 0.02        # |win diff| <= 2pp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-trades", required=True)
    ap.add_argument("--ref-summary", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # Self-reporting: ANY uncaught exception lands in the artifact as
    # DRIVER-ERROR (log-zips are hard to fetch; the artifact is the channel).
    def _hook(t, v, tb):
        try:
            import traceback as _t
            json.dump({"verdict": "DRIVER-ERROR",
                       "traceback": "".join(_t.format_exception(t, v, tb)),
                       "pre_registered_rule": "no parameter changes either way"},
                      open(a.out, "w"), indent=1)
        except Exception:
            pass
        sys.__excepthook__(t, v, tb)
    sys.excepthook = _hook
    print("PRE-REGISTERED RULE: this replay changes no parameters. "
          "MATCH = runner gold; MISMATCH = runner bug. Never tuning input.")

    base = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base)
    import yfinance as yfmod
    import paper

    data = yfmod.download(paper.UNIVERSE, start="2021-01-01",
                          auto_adjust=True, progress=False, threads=True)
    closes_all, opens_all = data["Close"], data["Open"]
    vols_all = data["Volume"]
    dates = closes_all.index
    assert len(dates) > 60, f"fetch too short: {len(dates)}"
    print(f"replay window: {dates[0].date()}..{dates[-1].date()} ({len(dates)} bars)")

    real_download = yfmod.download
    fake = {}

    def fake_download(*args, **kwargs):
        i = fake["i"]
        return {"Close": closes_all.iloc[:i + 1],
                "Open": opens_all.iloc[:i + 1],
                "Volume": vols_all.iloc[:i + 1]}
    yfmod.download = fake_download

    class FakeDT(_dt.datetime):
        @classmethod
        def utcnow(cls):
            return fake["now"]
    paper.dt = types.SimpleNamespace(datetime=FakeDT, timedelta=_dt.timedelta)

    tmp = tempfile.mkdtemp(prefix="replay_")
    os.chdir(tmp)
    sys.argv = ["paper.py", "r1"]
    try:
        for i in range(30, len(dates)):
            fake["i"] = i
            fake["now"] = dates[i].to_pydatetime() - _dt.timedelta(hours=5, minutes=30)
            try:
                paper.main()
            except SystemExit as e:
                if e.code not in (0, None):
                    raise RuntimeError(f"paper.main exited {e.code} on bar {dates[i].date()}")
    finally:
        yfmod.download = real_download

    st = json.load(open("paper_state_r1.dec.json"))
    ledger = [json.loads(l) for l in open("paper_ledger_r1.jsonl")]
    # Reconstruct (entry_date, exit_date) per exit via per-symbol FIFO.
    # Exact: the runner holds at most one open position per symbol.
    fills, exits = {}, []
    for r in ledger:
        if r.get("t") == "fill":
            fills.setdefault(r["s"], []).append(r)
        elif r.get("t") == "exit":
            exits.append(r)
    rp_trades = []
    for e in exits:
        q = fills.get(e["s"], [])
        if not q:
            raise RuntimeError(f"exit without fill: {e['s']} {e['d']}")
        f = q.pop(0)
        rp_trades.append((e["s"], f["d"], e["d"], f["px"], e["px"], e["net"]))

    import csv
    ref = list(csv.DictReader(open(a.ref_trades)))
    ref_keys = {(r["symbol"], r["entry"][:10], r["exit"][:10]) for r in ref}
    rp_keys = {(s, fe, xe) for s, fe, xe, _, _, _ in rp_trades}
    both = ref_keys & rp_keys
    key_rate = len(both) / max(len(ref_keys), 1)

    # Price agreement on matched keys (ledger rounds px to 4dp; allow 5bp).
    ref_by_key = {(r["symbol"], r["entry"][:10], r["exit"][:10]): r for r in ref}
    rp_by_key = {(s, fe, xe): (fp, xp) for s, fe, xe, fp, xp, _ in rp_trades}
    px_bad = 0
    for k in both:
        r = ref_by_key[k]
        fp, xp = rp_by_key[k]
        if abs(float(r["entry_px"]) - fp) / max(float(r["entry_px"]), 1e-9) > 5e-4:
            px_bad += 1
        if abs(float(r["exit_px"]) - xp) / max(float(r["exit_px"]), 1e-9) > 5e-4:
            px_bad += 1

    # Mismatch forensics (only read if MISMATCH): which keys diverge, and why.
    # Self-reporting: failures land in the artifact, never kill the run.
    miss_in_rp, extra_in_rp = [], []
    forensics_err, halt_info, crowded = None, [], []
    try:
        _miss = sorted(ref_keys - rp_keys)[:25]
        _extra = sorted(rp_keys - ref_keys)[:25]
        pos_of = {}
        for _k, _d in enumerate(opens_all.index):
            pos_of.setdefault(_d.date().isoformat(), _k)
        _ocol = {s: opens_all[s] for s in opens_all.columns}
        for _sym, _fe, _xe in _miss:
            try:
                _kk = pos_of.get(_fe)
                _o = float(_ocol[_sym].iloc[_kk]) if _kk is not None else float("nan")
                _oo = "ok" if _o == _o else ("NaN-open" if _kk is not None else "no-bar")
            except Exception:
                _oo = "no-bar"
            halt_info.append({"s": _sym, "entry": _fe, "open": _oo})
        from collections import Counter
        _rpd = Counter(fe for _, fe, _ in ref_keys)
        crowded = sorted(((_d, _c) for _d, _c in _rpd.items() if _c >= 8),
                         key=lambda _x: -_x[1])[:10]
        miss_in_rp, extra_in_rp = _miss, _extra
    except Exception:
        import traceback as _tb
        forensics_err = _tb.format_exc()
        miss_in_rp, extra_in_rp = [], []
    # Entry crowding on missing-entry dates (slot overflow pattern?).
    from collections import Counter
    ref_per_day = Counter(fe for _, fe, _ in ref_keys)
    crowded = sorted(((d, c) for d, c in ref_per_day.items() if c >= 8],
                     key=lambda x: -x[1])[:10]
    nets = st["closed"]
    n = len(nets)
    mean = sum(nets) / max(n, 1)
    win = sum(1 for x in nets if x > 0) / max(n, 1)
    ref_sum = json.load(open(a.ref_summary))
    dn = abs(n - ref_sum["trades"])
    dm = abs(mean - ref_sum["mean_net"])
    dw = abs(win - ref_sum["winrate"])
    match = (key_rate >= KEY_RATE_BAR and dn <= N_TOL
             and dm <= MEAN_TOL and dw <= WIN_TOL and px_bad == 0)
    # Driver-health fields: discriminate "runner diverged" from "driver stalled".
    # st["day"] increments ONLY when the IST-today guard passes, so it counts
    # processed sessions directly.
    by_year = {}
    for s, fe, xe, _, _, _ in rp_trades:
        by_year[xe[:4]] = by_year.get(xe[:4], 0) + 1
    first_x = min((xe for _, _, xe, _, _, _ in rp_trades), default=None)
    last_x = max((xe for _, _, xe, _, _, _ in rp_trades), default=None)
    out = {"verdict": "MATCH" if match else "MISMATCH",
           "pre_registered_rule": "no parameter changes either way",
           "driver": {"sessions_processed": st.get("day"),
                      "bars_available": len(dates),
                      "open_positions": len(st.get("positions", [])),
                      "pending_exits": len(st.get("pending_exits", [])),
                      "killed": st.get("killed"), "capped": st.get("capped", 0),
                      "first_exit": first_x, "last_exit": last_x,
                      "exits_by_year": by_year},
           "replay": {"trades": n, "mean_net": round(mean, 5),
                       "winrate": round(win, 4), "key_count": len(rp_keys)},
           "ref": {"trades": ref_sum["trades"], "mean_net": ref_sum["mean_net"],
                   "winrate": ref_sum["winrate"], "key_count": len(ref_keys)},
           "key_match_rate": round(key_rate, 4),
           "price_mismatches": px_bad,
           "forensics": {"missing_in_replay_sample": [list(k) for k in miss_in_rp],
                          "extra_in_replay_sample": [list(k) for k in extra_in_rp],
                          "halt_info": halt_info,
                          "crowded_ref_days": crowded,
                          "forensics_error": forensics_err},
           "bars": len(dates)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"replay trades={n} mean={mean:+.5f} win={win:.4f} "
          f"ref trades={ref_sum['trades']} mean={ref_sum['mean_net']:+.5f} win={ref_sum['winrate']:.4f}")
    print(f"key_match={key_rate:.4f} (bar {KEY_RATE_BAR}) dn={dn} dm={dm:.5f} dw={dw:.4f} px_bad={px_bad}")
    print("VERDICT:", out["verdict"])

if __name__ == "__main__":
    main()
