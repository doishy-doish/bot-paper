# triage-eval.py: Tier G v2 triage signals, donor-side only (kill-row 2, no bench touch).
# Per query: K temp-0.7 samples + 1 greedy temp-0 sample. Harness gate mirrors
# exp1-vote.py (schema: last non-empty line bare int <=6 digits; cap proxy >=145
# whitespace tokens at -n 300). Excluded samples drop from votes, counted.
# Triage rule (pre-registered, no tuning): DIRECT iff margin>=3 AND greedy==MV8,
# else VERIFY-route. Metrics: directFrac, directAcc (MV8 on direct items), mv8.
# PASS iff directFrac>=0.5 AND directAcc>=mv8-0.05 (halve invokes, no accuracy loss).
# Usage: python3 halt-r/triage-eval.py --bin /tmp/llama.cpp/build --model /tmp/qwen.gguf
#        --queries halt-r/exp1-queries.json --slice 0 --nslice 4 --k 8 --out out/triage
import argparse, json, os, re, subprocess

N_CAP = 300

def run(cmd, timeout=900):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or '') + (p.stderr or '')

def sample(cli, model, prompt, temp, seed):
    rc, out = run([cli, '-m', model, '-p', prompt, '-n', str(N_CAP),
                   '-c', '2048', '--temp', str(temp), '--seed', str(seed),
                   '-t', '4', '--no-display-prompt'])
    assert rc == 0, 'FAIL llama-cli rc=%d' % rc
    toks = out.split()
    lines = [l.strip() for l in out.strip().split('\n') if l.strip()]
    schema_ok = bool(lines) and re.fullmatch(r'-?\d{1,6}', lines[-1]) is not None
    cap_hit = len(toks) >= N_CAP - 5
    pred = int(lines[-1]) if schema_ok else None
    return pred, (not schema_ok) or cap_hit

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
        preds, excl = [], 0
        for s in range(a.k):
            pred, bad = sample(cli, a.model, q['prompt'], 0.7, s)
            if bad:
                excl += 1
            else:
                preds.append(pred)
        gpred, gbad = sample(cli, a.model, q['prompt'], 0.0, 999)
        if gbad:
            excl += 1
        counts = {}
        for p in preds:
            counts[p] = counts.get(p, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])
        mv8 = (top[0][0] == q['answer']) if top else False
        margin = (top[0][1] - (top[1][1] if len(top) > 1 else 0)) if top else 0
        greedy_agrees = (gpred is not None) and top and (gpred == top[0][0])
        direct = (margin >= 3) and greedy_agrees
        rows.append({'id': q['id'], 'answer': q['answer'], 'mv8': mv8,
                     'margin': margin, 'greedy_agrees': greedy_agrees,
                     'direct': direct, 'k_valid': len(preds), 'excluded': excl})
        print('triage %s mv8=%s margin=%d greedy_agrees=%s direct=%s k_valid=%d/%d' % (
            q['id'], mv8, margin, greedy_agrees, direct, len(preds), a.k), flush=True)
    json.dump({'slice': a.slice, 'k': a.k, 'rows': rows},
              open('%s/triage-slice-%d.json' % (a.out, a.slice), 'w'), indent=1)

main()
