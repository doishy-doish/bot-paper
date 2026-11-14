# probe-raw.py: one-shot raw-completion dump (diagnostic — fires on user push order).
# Runs 10 queries with -p-only flags (same as Batch A scripts) and persists FULL raw
# stdout AND stderr per completion, plus a shape table graded on STDOUT ONLY.
# The table is the shape-namer: it discriminates the two live hypotheses for gen-4's
# uniform schema wall — (a) donor-side prose/restatement endings vs (b) harness-side
# mis-extraction (stderr last-line poisoning the old combined-stream grade).
# Shape buckets: strict (bare-int last line) / loose (int elsewhere, format-agnostic) /
# none (no int) / empty; truncated flag orthogonal (token limit hit — always voids
# downstream regardless of shape). Correctness is format-agnostic (loose pred == answer)
# AND strict, both recorded; the table ranks, it does NOT close (n=10, ~+-30%).
# Report `none` bucket first (the other-bucket rule). No verdict language here.
# Usage: python3 halt-r/probe-raw.py --bin ... --model ... --queries ... --out out/probe
import argparse, json, os, re, subprocess

N_TOK = 300

def classify(stdout, answer, n=N_TOK):
    toks = stdout.split()
    lines = [l.strip() for l in stdout.strip().split('\n') if l.strip()]
    strict = bool(lines) and re.fullmatch(r'-?\d{1,6}', lines[-1]) is not None
    ints = re.findall(r'(?<!\d)-?\d{1,9}(?!\d)', stdout)
    loose = int(ints[-1]) if ints else None
    trunc = len(toks) >= n - 5
    shape = 'strict' if strict else ('loose' if loose is not None else ('empty' if not lines else 'none'))
    return {'shape': shape, 'truncated': trunc,
            'correct_strict': bool(strict and int(lines[-1]) == answer),
            'correct_loose': bool(loose is not None and loose == answer)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--queries', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    qs = json.load(open(a.queries))[:10]
    cli = os.path.join(a.bin, 'llama-cli')
    assert os.path.isfile(cli), 'FAIL no llama-cli'
    shapes = []
    for q in qs:
        for temp, seed in ((0.7, 0), (0.0, 999)):
            p = subprocess.run(
                [cli, '-m', a.model, '-p', q['prompt'], '-n', str(N_TOK),
                 '-c', '2048', '--temp', str(temp), '--seed', str(seed),
                 '-t', '4', '--no-display-prompt'],
                capture_output=True, text=True, timeout=900)
            fn = '%s/raw-%s-t%s-s%d.txt' % (a.out, q['id'], temp, seed)
            with open(fn, 'w') as f:
                f.write('RC=%d\n---STDOUT---\n%s\n---STDERR---\n%s' % (p.returncode, p.stdout, p.stderr))
            c = classify(p.stdout or '', q['answer'])
            c.update({'id': q['id'], 'temp': temp, 'seed': seed,
                      'rc': p.returncode, 'stdout_bytes': len(p.stdout or '')})
            shapes.append(c)
            print('probe %s t=%s rc=%d shape=%s trunc=%s correct_loose=%s correct_strict=%s' % (
                q['id'], temp, p.returncode, c['shape'], c['truncated'],
                c['correct_loose'], c['correct_strict']), flush=True)
    by_shape = {}
    for c in shapes:
        by_shape[c['shape']] = by_shape.get(c['shape'], 0) + 1
    summary = {'n': len(shapes),
               'none_first': {k: by_shape.get(k, 0) for k in ('none', 'empty', 'loose', 'strict')},
               'truncated': sum(1 for c in shapes if c['truncated']),
               'correct_loose': sum(1 for c in shapes if c['correct_loose']),
               'correct_strict': sum(1 for c in shapes if c['correct_strict']),
               'rows': shapes}
    json.dump(summary, open('%s/probe-shapes.json' % a.out, 'w'), indent=1)
    print('SHAPES none=%d empty=%d loose=%d strict=%d truncated=%d correct_loose=%d correct_strict=%d' % (
        summary['none_first']['none'], summary['none_first']['empty'],
        summary['none_first']['loose'], summary['none_first']['strict'],
        summary['truncated'], summary['correct_loose'], summary['correct_strict']), flush=True)

main()
