# exp1-vote.py: matched-K sampling for verifier-vs-majority + headroom gate (stdlib only).
# HARNESS GATE (locked 2026-11-14): no scored completion may be truncated. Schema: the
# last non-empty line must be a bare integer (<=6 digits). Cap proxy: >=145 whitespace
# tokens with -n 150 means the cap likely cut the answer. Violations are EXCLUDED from
# votes, counted, and reported. Collect voids the run if excluded total > 0.
# Usage: python3 halt-r/exp1-vote.py --bin /tmp/llama.cpp/build/bin --model /tmp/qwen.gguf
#        --queries halt-r/exp1-queries.json --slice 0 --nslice 4 --k 8 --out out/exp1
import argparse, json, os, re, subprocess

N_CAP = 150

def run(cmd, timeout=900):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or '') + (p.stderr or '')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--queries', required=True)
    ap.add_argument('--slice', type=int, required=True)
    ap.add_argument('--nslice', type=int, required=True)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    qs = json.load(open(a.queries))
    mine = [q for i, q in enumerate(qs) if i % a.nslice == a.slice]
    assert len(mine) > 0, 'FAIL empty slice'
    cli = os.path.join(a.bin, 'llama-cli')
    assert os.path.isfile(cli), 'FAIL no llama-cli'
    rows = []
    for q in mine:
        samples = []
        for s in range(a.k):
            rc, out = run([cli, '-m', a.model, '-p', q['prompt'], '-n', str(N_CAP),
                           '-c', '2048', '--temp', '0.7', '--seed', str(s),
                           '-t', '4', '--no-display-prompt'])
            assert rc == 0, 'FAIL llama-cli rc=%d' % rc
            toks = out.split()
            lines = [l.strip() for l in out.strip().split('\n') if l.strip()]
            schema_ok = bool(lines) and re.fullmatch(r'-?\d{1,6}', lines[-1]) is not None
            cap_hit = len(toks) >= N_CAP - 5
            excluded = (not schema_ok) or cap_hit
            pred = int(lines[-1]) if schema_ok else None
            samples.append({'seed': s, 'pred': pred, 'excluded': excluded,
                            'why': ('schema' if not schema_ok else '') +
                                   ('+cap' if (not schema_ok and cap_hit) else ('cap' if cap_hit else ''))})
        valid = [h for h in samples if not h['excluded']]
        preds = [h['pred'] for h in valid]
        single = (preds[0] == q['answer']) if preds else False
        v4 = preds[:4]
        mv4 = (max(set(v4), key=v4.count) == q['answer']) if v4 else False
        mv8 = (max(set(preds), key=preds.count) == q['answer']) if preds else False
        oracle = any(p == q['answer'] for p in preds)
        rows.append({'id': q['id'], 'answer': q['answer'], 'single': single,
                     'mv4': mv4, 'mv8': mv8, 'oracle': oracle,
                     'k_valid': len(valid), 'excluded': len(samples) - len(valid),
                     'preds': [(h['pred'] if not h['excluded'] else 'X') for h in samples]})
        print('exp1 %s single=%s mv4=%s mv8=%s oracle=%s k_valid=%d/%d' % (
            q['id'], single, mv4, mv8, oracle, len(valid), len(samples)), flush=True)
    json.dump({'slice': a.slice, 'k': a.k, 'rows': rows},
              open('%s/exp1-slice-%d.json' % (a.out, a.slice), 'w'), indent=1)
    print('exp1 slice %d done (%d queries)' % (a.slice, len(rows)), flush=True)

main()
