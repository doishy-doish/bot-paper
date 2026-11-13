# make-arith.py: deterministic arithmetic corpus for the capability-ingestion (gluttony) test.
# Train range: a+b with a,b in [0,49]. Test range: a,b in [50,99]. Fixed-width lines
# "%02d+%02d=%03d\n" (10 chars each) so the bench's 90/10 char cut lands EXACTLY on the
# range boundary: test = 2500 lines (25000 chars), train = 9 copies (22500 lines,
# 225000 chars), total 250000 chars, cut = floor(0.9*250000) = 225000 = train length.
# Outputs: arith-train.txt (donor training, one shuffled copy, its own 90/10 val stays
# in-range), arith-full.txt (bench file), arith-spec.json (boundary proof + vocab +
# window-overlap rate). All randomness from one LCG (same constants as bench/trainer).
# Fails loudly on any violation. Stdlib only.
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
    train_pairs = [(x, y) for x in range(50) for y in range(50)]
    test_pairs = [(x, y) for x in range(50, 100) for y in range(50, 100)]
    fmt = lambda p: '%02d+%02d=%03d\n' % (p[0], p[1], p[0] + p[1])
    test_lines = [fmt(p) for p in shuffled(test_pairs, rr)]
    train_copies = [shuffled(train_pairs, rr) for _ in range(9)]
    train_lines = [fmt(p) for copy in train_copies for p in copy]
    assert len(train_lines) == 22500 and len(test_lines) == 2500
    assert all(len(l) == 10 for l in train_lines + test_lines)
    train_txt = ''.join(train_lines)
    full_txt = train_txt + ''.join(test_lines)
    assert len(full_txt) == 250000
    cut = (len(full_txt) * 9) // 10
    if cut != len(train_txt):
        raise SystemExit('FAIL boundary cut=%d train_len=%d' % (cut, len(train_txt)))
    vocab = sorted(set(train_txt))
    if any(c not in vocab for c in ''.join(test_lines)):
        raise SystemExit('FAIL test has chars outside train vocab')
    # window-overlap: fraction of test W-windows already present in train (want ~0)
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
    spec = {'task': 'arith-gluttony-corpus', 'trainRange': [0, 49], 'testRange': [50, 99],
            'lineFmt': '%02d+%02d=%03d\\n', 'lineLen': 10, 'trainLines': 22500,
            'testLines': 2500, 'totalChars': 250000, 'cut': cut,
            'boundaryExact': True, 'vocab': vocab, 'V': len(vocab),
            'W': a.W, 'testWindows': tot, 'matchFree': match_free,
            'shaFull': hashlib.sha256(full_txt.encode()).hexdigest(),
            'shaTrain': hashlib.sha256(
                ''.join(fmt(p) for p in train_copies[0]).encode()).hexdigest()}
    open(a.out + '/arith-spec.json', 'w').write(json.dumps(spec, indent=1))
    print('make-arith: done')

main()
