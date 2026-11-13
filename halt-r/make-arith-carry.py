# make-arith-carry.py: carry-rich arithmetic corpus (data shot, not size shot).
# Train range: a,b in [0,74]. Test range: a,b in [75,99]. Fixed-width lines
# "%02d+%02d=%03d\n" (10 chars). Train pairs 75*75=5625 x1 copy = 5625 lines
# (56250 chars); test 25*25=625 lines (6250 chars); total 62500 chars,
# cut = floor(0.9*62500) = 56250 = train length EXACTLY.
# Same LCG/shuffle/boundary-proof/matchFree bar as make-arith.py. Stdlib only.
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--W', type=int, default=8)
    a = ap.parse_args()
    import os
    os.makedirs(a.out, exist_ok=True)
    rr = lcg(1234)
    train_pairs = [(x, y) for x in range(75) for y in range(75)]
    test_pairs = [(x, y) for x in range(75, 100) for y in range(75, 100)]
    fmt = lambda p: '%02d+%02d=%03d\n' % (p[0], p[1], p[0] + p[1])
    test_lines = [fmt(p) for p in shuffled(test_pairs, rr)]
    train_copies = [shuffled(train_pairs, rr)]
    train_lines = [fmt(p) for copy in train_copies for p in copy]
    assert len(train_lines) == 5625 and len(test_lines) == 625
    assert all(len(l) == 10 for l in train_lines + test_lines)
    train_txt = ''.join(train_lines)
    full_txt = train_txt + ''.join(test_lines)
    assert len(full_txt) == 62500
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
        raise SystemExit('FAIL match_free=%.4f < 0.99 (test range leaks into train windows)' % match_free)
    open(a.out + '/arith-train.txt', 'w').write(''.join(fmt(p) for p in train_copies[0]))
    open(a.out + '/arith-full.txt', 'w').write(full_txt)
    spec = {'task': 'arith-carry-corpus', 'trainRange': [0, 74], 'testRange': [75, 99],
            'lineFmt': '%02d+%02d=%03d\\n', 'lineLen': 10, 'trainLines': 5625,
            'testLines': 625, 'totalChars': 62500, 'cut': cut,
            'boundaryExact': True, 'vocab': vocab, 'V': len(vocab),
            'W': a.W, 'testWindows': tot, 'matchFree': match_free,
            'shaFull': hashlib.sha256(full_txt.encode()).hexdigest(),
            'shaTrain': hashlib.sha256(
                ''.join(fmt(p) for p in train_copies[0]).encode()).hexdigest()}
    open(a.out + '/arith-spec.json', 'w').write(json.dumps(spec, indent=1))
    print('make-arith-carry: done')

main()
