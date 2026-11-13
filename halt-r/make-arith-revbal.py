# make-arith-revbal.py: reversed-output + balanced carry-chain corpus.
# Fixes, in order: (1) dataset shift — train sums 0-150 cross 100 so hundreds
# digit 0 AND 1 are both seen in train (old 0-49/50-99 made hundreds constant
# 0 in train, 1 in test: unlearnable by construction); (2) causality — sum
# digits emitted least-significant-first (reversed), so carry propagates
# causally (Lee et al. 2310.16028: reverse restores causal carry; forward has
# no hard-carry length generalization); (3) distribution — balanced
# carry-chain sampling: chain length sampled uniformly from {0,1,2}, then a
# question with exactly that length (Zhou et al. 2307.03381: uniform 0..n
# carry-ons, not just "more carry cases").
# Format: "%02d+%02d=%s\n" with 3 reversed sum digits (10 chars/line, V=13
# unchanged: digits+plus+equals+newline). 2-digit addends keep chain lengths
# 0/1/2: c1=(a0+b0>=10), c2=(a1+b1+c1>=10); L0:(0,0) L1:exactly-one L2:(1,1).
# Counts for exact 90/10 bench cut: train 5400 (1800/length) + test 600
# (200/length), all disjoint pairs, total 60000 chars, cut=54000=train len.
# Train pool sums<=150, test pool sums>=151 (hard-carry, hundreds 1 seen in
# train via 100-150). Deterministic LCG(1234) + Fisher-Yates, asserts fail
# loudly. Stdlib only.
import argparse, json, hashlib

def lcg(seed):
    s = seed & 0xFFFFFFFF
    while True:
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        yield s / 4294967296

def shuffled(lst, rr):
    a = list(lst)
    for i in range(len(a) - 1, 0, -1):
        j = int(next(rr) * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a

def chain(a, b):
    c1 = 1 if (a % 10 + b % 10) >= 10 else 0
    c2 = 1 if (a // 10 + b // 10 + c1) >= 10 else 0
    if c1 == 0 and c2 == 0:
        return 0
    if c1 == 1 and c2 == 1:
        return 2
    return 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--W', type=int, default=8)
    a = ap.parse_args()
    import os
    os.makedirs(a.out, exist_ok=True)
    rr = lcg(1234)
    pools = {0: [], 1: [], 2: []}
    for x in range(100):
        for y in range(100):
            if x + y > 150:
                continue
            pools[chain(x, y)].append((x, y))
    for L in (0, 1, 2):
        assert len(pools[L]) >= 1800, 'pool L%d=%d <1800' % (L, len(pools[L]))
    fmt = lambda p: '%02d+%02d=%s\n' % (p[0], p[1], ('%03d' % (p[0] + p[1]))[::-1])
    train_lines, test_lines = [], []
    for L in (0, 1, 2):
        q = shuffled(pools[L], rr)
        train_lines += [fmt(p) for p in q[:1620]]
        test_lines += [fmt(p) for p in q[1620:1800]]
    train_lines = shuffled(train_lines, rr)
    test_lines = shuffled(test_lines, rr)
    assert len(train_lines) == 4860 and len(test_lines) == 540
    assert all(len(l) == 10 for l in train_lines + test_lines)
    train_txt = ''.join(train_lines)
    full_txt = train_txt + ''.join(test_lines)
    assert len(full_txt) == 54000
    cut = (len(full_txt) * 9) // 10
    if cut != len(train_txt):
        raise SystemExit('FAIL boundary cut=%d train_len=%d' % (cut, len(train_txt)))
    vocab = sorted(set(train_txt))
    if any(c not in vocab for c in ''.join(test_lines)):
        raise SystemExit('FAIL test has chars outside train vocab')
    tw = set()
    for i in range(len(train_txt) - a.W + 1):
        tw.add(train_txt[i:i + a.W])
    test_sec = ''.join(test_lines)
    nov, tot = 0, 0
    for i in range(len(test_sec) - a.W + 1):
        tot += 1
        if test_sec[i:i + a.W] not in tw:
            nov += 1
    match_free = nov / tot
    print('train_chars=%d test_chars=%d cut=%d vocab=%r V=%d' % (
        len(train_txt), len(test_sec), cut, ''.join(vocab), len(vocab)))
    print('W=%d test_windows=%d match_free=%.4f' % (a.W, tot, match_free))
    if match_free < 0.99:
        raise SystemExit('FAIL match_free=%.4f < 0.99' % match_free)
    open(a.out + '/arith-train.txt', 'w').write(train_txt)
    open(a.out + '/arith-full.txt', 'w').write(full_txt)
    spec = {'task': 'arith-revbal-corpus', 'trainSums': '0-150', 'testSums': '0-150 disjoint unseen pairs',
            'reversed': True, 'balanced': [1620, 1620, 1620], 'testBalanced': [180, 180, 180],
            'lineFmt': '%02d+%02d=REV(%03d)\\n', 'lineLen': 10, 'trainLines': 4860,
            'testLines': 540, 'totalChars': 54000, 'cut': cut,
            'boundaryExact': True, 'vocab': vocab, 'V': len(vocab),
            'W': a.W, 'testWindows': tot, 'matchFree': match_free,
            'cites': ['2310.16028', '2307.03381'],
            'shaFull': hashlib.sha256(full_txt.encode()).hexdigest(),
            'shaTrain': hashlib.sha256(train_txt.encode()).hexdigest()}
    open(a.out + '/arith-spec.json', 'w').write(json.dumps(spec, indent=1))
    print('make-arith-revbal: done')

main()
