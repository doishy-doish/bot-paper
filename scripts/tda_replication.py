# tda_replication.py -- X2 proper TDA landscape-norm replication (K-FREE, ZERO K).
#
# Pre-reg: BATCHES.md "X2". No engine change, no K trial. Standalone script:
# yfinance + numpy/pandas/scipy/ripser only. Lives in the PUBLIC repo
# (scripts/tda_replication.py, synced from src/bot/gh/tda_replication.py);
# outputs are scramble-encrypted by the tda.yml collect job.
#
# Four modes (one matrix leg each):
#   stocks22    N=49 NSE stocks, epochs tau=22 shift=5, RAW log-returns
#               (India paper 2311.17016 geometry: tau=22/Delta=5, N-stock clouds)
#   stocks22std same, but each stock STANDARDIZED within each epoch (mu=0, sd=1)
#               fixes F=1 -> pure geometry, kills the rung-2 variance confound
#   idx50       d=4 Indian indices (^NSEI ^NSEBANK ^NSEMDCP50 ^CNXIT),
#               w=50 step=1, RAW (Gidea-Katz 1703.04385 geometry, d=4 / w=50)
#   idx100      same d=4, w=100 step=5, RAW
#
# Per epoch: mean_corr, lam1_frac (RMT market-mode share), persistent entropy
# H0/H1/H-all, H1 landscape L1/L2 (Bubenik tents), forward-10-bar equal-weight
# basket DD (diagnostic target), trailing-20d basket vol (control).
# Spectral-slope + shuffle-null + vol-control comparisons happen in SCORING
# (workspace-side, tiny), not here.

import sys
import json
import numpy as np
import pandas as pd
from ripser import ripser

START = "2015-01-01"

# Same 49-name Nifty-50 universe as the production engine (backtest.py UNIVERSE).
UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LICI.NS", "HINDUNILVR.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "LT.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS",
    "ONGC.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "HCLTECH.NS", "TECHM.NS", "WIPRO.NS",
    "NESTLEIND.NS", "BRITANNIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS",
    "EICHERMOT.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "GRASIM.NS",
    "HINDALCO.NS", "COALINDIA.NS", "BPCL.NS", "TATACONSUM.NS", "APOLLOHOSP.NS",
    "TRENT.NS", "BEL.NS", "SHRIRAMFIN.NS", "SBILIFE.NS", "HDFCLIFE.NS",
]

IDX_SYMBOLS = ["^NSEI", "^NSEBANK", "^NSEMDCP50", "^CNXIT"]

MODES = {
    "stocks22": {"kind": "stocks", "w": 22, "step": 5, "std": False},
    "stocks22std": {"kind": "stocks", "w": 22, "step": 5, "std": True},
    "idx50": {"kind": "idx", "w": 50, "step": 1, "std": False},
    "idx100": {"kind": "idx", "w": 100, "step": 5, "std": False},
}

GRID_N = 2001


def parse_args(argv):
    mode = None
    out = None
    start = START
    end = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--mode" and i + 1 < len(argv):
            mode = argv[i + 1]
            i += 2
        elif a == "--out" and i + 1 < len(argv):
            out = argv[i + 1]
            i += 2
        elif a == "--start" and i + 1 < len(argv):
            start = argv[i + 1]
            i += 2
        elif a == "--end" and i + 1 < len(argv):
            end = argv[i + 1]
            i += 2
        else:
            i += 1
    return mode, out, start, end


def persistent_entropy(lifetimes):
    vals = [float(x) for x in lifetimes if np.isfinite(x) and x > 0]
    if len(vals) < 2:
        return 0.0
    tot = sum(vals)
    if tot <= 0:
        return 0.0
    return float(-sum((v / tot) * np.log(v / tot) for v in vals))


def landscape_norms_h1(dgm1):
    """Bubenik first-landscape L1/L2 on a fixed grid. dgm1: (birth, death) rows."""
    if dgm1 is None or len(dgm1) == 0:
        return 0.0, 0.0
    bd = np.asarray(dgm1, dtype=float)
    bd = bd[np.isfinite(bd).all(axis=1)]
    bd = bd[bd[:, 1] > bd[:, 0]]
    if len(bd) == 0:
        return 0.0, 0.0
    dmax = float(bd[:, 1].max())
    if dmax <= 0:
        return 0.0, 0.0
    grid = np.linspace(0.0, dmax, GRID_N)
    lam = np.zeros_like(grid)
    for b, d in bd:
        tent = np.maximum(0.0, np.minimum(grid - b, d - grid))
        lam = np.maximum(lam, tent)
    l1 = float(np.trapezoid(lam, grid))
    l2 = float(np.sqrt(np.trapezoid(lam * lam, grid)))
    return l1, l2


def epoch_measures(X):
    """X: (w, n) log-returns, NaN-free, non-degenerate. Returns dict."""
    w, n = X.shape
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 1.0)
    iu = np.triu_indices(n, k=1)
    mean_corr = float(np.mean(C[iu])) if n > 1 else 0.0
    eig = np.linalg.eigvalsh(C)
    lam1_frac = float(eig[-1] / max(n, 1))
    D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - C)))
    dg = ripser(D, distance_matrix=True, maxdim=1)["dgms"]
    d0 = np.asarray(dg[0], dtype=float) if len(dg) > 0 else np.zeros((0, 2))
    d1 = np.asarray(dg[1], dtype=float) if len(dg) > 1 else np.zeros((0, 2))
    l0 = d0[:, 1] - d0[:, 0] if len(d0) else np.zeros(0)
    l1 = d1[:, 1] - d1[:, 0] if len(d1) else np.zeros(0)
    both = np.concatenate([l0, l1]) if len(l0) + len(l1) else np.zeros(0)
    L1, L2 = landscape_norms_h1(d1)
    return {
        "mean_corr": mean_corr,
        "lam1_frac": lam1_frac,
        "pe_h0": persistent_entropy(l0),
        "pe_h1": persistent_entropy(l1),
        "pe_all": persistent_entropy(both),
        "l1": L1,
        "l2": L2,
    }


def main():
    import yfinance as yf
    mode, out, start, end = parse_args(sys.argv[1:])
    if mode not in MODES or not out:
        print("usage: tda_replication.py --mode %s --out FILE [--start D --end D]"
              % "|".join(sorted(MODES)))
        sys.exit(2)
    cfg = MODES[mode]
    syms = list(UNIVERSE) if cfg["kind"] == "stocks" else list(IDX_SYMBOLS)
    kw = {"start": start, "progress": False, "threads": True, "auto_adjust": True}
    if end:
        kw["end"] = end
    print("pulling %d symbols %s..%s [%s] ..." % (len(syms), start, end or "now", mode))
    data = yf.download(syms, **kw)
    closes = data["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame()
    closes = closes.dropna(axis=1, how="all")
    logret = np.log(closes / closes.shift(1))
    dates = logret.index
    R = logret.to_numpy(dtype=float)
    T, N = R.shape
    w = cfg["w"]
    step = cfg["step"]
    min_n = 20 if cfg["kind"] == "stocks" else 3
    basket = np.nanmean(R, axis=1)
    epochs = []
    skipped = 0
    for s in range(0, T - w - 10 + 1, step):
        blk = R[s:s + w, :]
        good = [j for j in range(N)
                if np.count_nonzero(~np.isnan(blk[:, j])) >= w - 2]
        if len(good) < min_n:
            skipped += 1
            continue
        X = blk[:, good].copy()
        sd = np.nanstd(X, axis=0)
        X = X[:, sd > 1e-12]
        if X.shape[1] < min_n:
            skipped += 1
            continue
        colmean = np.nanmean(X, axis=0)
        bad = np.where(np.isnan(X))
        X[bad] = np.take(colmean, bad[1])
        if cfg["std"]:
            mu = X.mean(axis=0)
            sg = X.std(axis=0)
            sg[sg < 1e-12] = 1.0
            X = (X - mu) / sg
        try:
            m = epoch_measures(X)
        except Exception:
            skipped += 1
            continue
        eidx = s + w - 1
        fwd = basket[s + w:s + w + 10]
        if np.count_nonzero(~np.isnan(fwd)) < 8:
            skipped += 1
            continue
        cum = np.exp(np.nancumsum(fwd)) - 1.0
        fwd10dd = float(np.nanmin(cum))
        tr = basket[max(0, s + w - 20):s + w]
        tr = tr[~np.isnan(tr)]
        vol20 = float(np.std(tr)) if len(tr) >= 15 else float("nan")
        epochs.append({
            "date": dates[eidx].strftime("%Y-%m-%d"),
            "n": int(X.shape[1]),
            "mean_corr": m["mean_corr"],
            "lam1_frac": m["lam1_frac"],
            "pe_h0": m["pe_h0"],
            "pe_h1": m["pe_h1"],
            "pe_all": m["pe_all"],
            "l1": m["l1"],
            "l2": m["l2"],
            "fwd10dd": fwd10dd,
            "vol20": vol20,
        })
    res = {
        "mode": mode,
        "kind": cfg["kind"],
        "w": w,
        "step": step,
        "standardized": cfg["std"],
        "symbols_requested": syms,
        "symbols_got": [str(c) for c in closes.columns],
        "bars": int(T),
        "epochs": epochs,
        "skipped": skipped,
    }
    with open(out, "w") as f:
        json.dump(res, f)
    print("mode=%s bars=%d epochs=%d skipped=%d -> %s"
          % (mode, T, len(epochs), skipped, out))


if __name__ == "__main__":
    main()
