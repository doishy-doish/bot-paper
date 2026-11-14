# tier0-eval.py: Tier 0 draft-accept harness v0, donor-only (kill-row 4, no bench touch).
# Protocol per query (fixed cost, no tuning): 1 greedy draft (temp 0) + K=4 verify
# samples (temp 0.7). Accept draft iff draft==MV4. Answer = draft if accepted else
# MV4 of the same 4 (fallback, zero extra gens). Baseline = MV8 (8 temp-0.7 samples).
# Metrics: acceptRate, tier0acc, mv8, draftRate. Kill iff acceptRate*1.6<1
# (speedup 8/5=1.6 fixed) OR tier0acc<mv8-0.05. Pass reopens true specdec v1
# (same-vocab draft/verify pair). Harness gate mirrors exp1-vote.py.
# Usage: python3 halt-r/tier0-eval.py --bin ... --model ... --queries ... --slice 0 --nslice 4 --out out/tier0
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
    bad = (not schema_ok) or cap_hit
    why = 'both' if ((not schema_ok) and cap_hit) else 'cap' if cap_hit else 'schema' if not schema_ok else 'ok'
    return pred, bad, why

def majority(preds):
    if not preds:
        return None
    return max(set(preds), key=preds.count)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--queries', required=True)
    ap.add_argument('--slice', type=int, required=True)
    ap.add_argument('--nslice', type=int, required=True)
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
        draft, dbad, dwhy = sample(cli, a.model, q['prompt'], 0.0, 999)
        vpreds, excl, esch, ecap = [], (1 if dbad else 0), 0, 0
        if dbad:
            if dwhy in ('schema', 'both'):
                esch += 1
            if dwhy in ('cap', 'both'):
                ecap += 1
        for s in range(4):
            pred, bad, why = sample(cli, a.model, q['prompt'], 0.7, s)
            if bad:
                excl += 1
                if why in ('schema', 'both'):
                    esch += 1
                if why in ('cap', 'both'):
                    ecap += 1
            else:
                vpreds.append(pred)
        bpreds = list(vpreds)
        for s in range(4, 8):
            pred, bad, why = sample(cli, a.model, q['prompt'], 0.7, s)
            if bad:
                excl += 1
                if why in ('schema', 'both'):
                    esch += 1
                if why in ('cap', 'both'):
                    ecap += 1
            else:
                bpreds.append(pred)
        mv4 = majority(vpreds)
        mv8 = majority(bpreds)
        accept = (draft is not None) and (mv4 is not None) and (draft == mv4)
        ans = draft if accept else mv4
        rows.append({'id': q['id'], 'answer': q['answer'],
                     'draft_ok': (draft == q['answer']) if draft is not None else False,
                     'accept': accept,
                     'tier0_ok': (ans == q['answer']) if ans is not None else False,
                     'mv8_ok': (mv8 == q['answer']) if mv8 is not None else False,
                     'excluded': excl, 'excl_schema': esch, 'excl_cap': ecap})
        print('tier0 %s draft_ok=%s accept=%s tier0_ok=%s mv8_ok=%s excl=%d' % (
            q['id'], rows[-1]['draft_ok'], accept, rows[-1]['tier0_ok'],
            rows[-1]['mv8_ok'], excl), flush=True)
    json.dump({'slice': a.slice, 'rows': rows,
               'why': {'schema': sum(r['excl_schema'] for r in rows),
                       'cap': sum(r['excl_cap'] for r in rows)}},
              open('%s/tier0-slice-%d.json' % (a.out, a.slice), 'w'), indent=1)
    print('WHY schema=%d cap=%d' % (
        sum(r['excl_schema'] for r in rows), sum(r['excl_cap'] for r in rows)), flush=True)

main()
