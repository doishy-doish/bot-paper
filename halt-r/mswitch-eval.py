# mswitch-eval.py: ModelSwitch cross-donor agreement diagnostic (kill-row 5).
# Two frozen donors (Qwen2.5-0.5B Q4_K_M + Qwen3-0.6B Q8_0), same 20 queries:
# greedy + K=4 temp-0.7 each. Metrics: accA, accB, agreeWrong (both wrong AND same
# wrong answer / both wrong), fixable (exactly one donor right).
# Pre-reg: PASS (build cross-donor vote) iff fixable>=0.15 (oracle-switch headroom
# worth building for). KILL (shared failure modes, vote adds nothing) iff
# agreeWrong>=0.8 of jointly-wrong. Harness gate mirrors exp1-vote.py.
# Usage: python3 halt-r/mswitch-eval.py --bin ... --modelA ... --modelB ...
#        --queries halt-r/exp1-queries.json --slice 0 --nslice 2 --out out/mswitch
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

def donor_run(cli, model, prompt):
    preds, excl, esch, ecap = [], 0, 0, 0
    g, gb, gw = sample(cli, model, prompt, 0.0, 999)
    if gb:
        excl += 1
        if gw in ('schema', 'both'):
            esch += 1
        if gw in ('cap', 'both'):
            ecap += 1
    else:
        preds.append(g)
    for s in range(4):
        p, bad, why = sample(cli, model, prompt, 0.7, s)
        if bad:
            excl += 1
            if why in ('schema', 'both'):
                esch += 1
            if why in ('cap', 'both'):
                ecap += 1
        else:
            preds.append(p)
    return majority(preds), excl, esch, ecap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--modelA', required=True)
    ap.add_argument('--modelB', required=True)
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
        ma, ea, eas, eac = donor_run(cli, a.modelA, q['prompt'])
        mb, eb, ebs, ebc = donor_run(cli, a.modelB, q['prompt'])
        oka = (ma == q['answer']) if ma is not None else False
        okb = (mb == q['answer']) if mb is not None else False
        rows.append({'id': q['id'], 'answer': q['answer'], 'a_ok': oka, 'b_ok': okb,
                     'same_wrong': (not oka) and (not okb) and (ma is not None) and (ma == mb),
                     'fixable': oka != okb, 'excluded': ea + eb,
                     'excl_schema': eas + ebs, 'excl_cap': eac + ebc})
        print('mswitch %s a_ok=%s b_ok=%s same_wrong=%s fixable=%s excl=%d' % (
            q['id'], oka, okb, rows[-1]['same_wrong'], rows[-1]['fixable'], ea + eb), flush=True)
    json.dump({'slice': a.slice, 'rows': rows,
               'why': {'schema': sum(r['excl_schema'] for r in rows),
                       'cap': sum(r['excl_cap'] for r in rows)}},
              open('%s/mswitch-slice-%d.json' % (a.out, a.slice), 'w'), indent=1)
    print('WHY schema=%d cap=%d' % (
        sum(r['excl_schema'] for r in rows), sum(r['excl_cap'] for r in rows)), flush=True)

main()
