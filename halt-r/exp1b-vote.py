# exp1b-vote.py: verifier-vs-majority, self-contained (kill-row 1; subsumes exp1 refire).
# Per query: K=8 temp-0.7 samples (seeds 0-7, full texts kept) + 1 greedy temp-0.
# D1 room (descriptive): mv8 below oracleelse selector line dead.
# D2 keep rule: verifier top-1 beats MV8 same bench, McNemar p<0.05, abstain option.
# Verifier prompt: question + 8 completions in seed order, one extra donor call.
# Abstain (pre-reg, no tuning): vote margin==0 (top-two tie) OR verifier schema-fail;
# fallback MV8 deterministic tie-break (lowest seed). Collect does McNemar.
# Adaptive margin-2 simulated retrospectively: sequential seeds, stop at first prefix
# with margin>=2 else 8; report meanStop + stop-majority acc vs MV8.
# Harness gate mirrors exp1-vote.py (schema last-int, cap>=295 at -n 300, excluded
# counted; collect voids run if excluded>0).
# Usage: python3 halt-r/exp1b-vote.py --bin ... --model ... --queries ... --slice 0 --nslice 4 --k 8 --out out/exp1b
import argparse, json, os, re, subprocess

N_CAP = 300

def run(cmd, timeout=900):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or ''), (p.stderr or '')

def gen(cli, model, prompt, temp, seed, n=None, c=2048):
    n = N_CAP if n is None else n
    rc, out, _ = run([cli, '-m', model, '-p', prompt, '-n', str(n),
                   '-c', str(c), '--temp', str(temp), '--seed', str(seed),
                   '-t', '4', '--no-display-prompt'])
    assert rc == 0, 'FAIL llama-cli rc=%d' % rc
    return out

def extract_span(out):
    # CHROME-FIX (2026-11-14, probe-verified): llama-cli conversation UI
    # (auto-enabled by chat template) prints banner/echo/timing/Exiting... on
    # stdout. Donor span = after the first '> ' echo, up to the timing line,
    # '>', or 'Exiting...'. Fallback: trimmed whole stdout (clean-output safe:
    # a future -no-cnv flag that removes the chrome must not void everything).
    lines = out.split('\n')
    start = -1
    for i, l in enumerate(lines):
        if l.startswith('> '):
            start = i
            break
    if start >= 0:
        end = len(lines)
        for j in range(start + 1, len(lines)):
            lj = lines[j]
            if lj.startswith('[ Prompt:') or lj == '>' or lj == 'Exiting...' or 't/s' in lj:
                end = j
                break
        span = '\n'.join(lines[start + 1:end]).strip()
        if span:
            return span
    return out.strip()

def grade(out, n):
    # D2a (2026-11-14) + CHROME-FIX (probe-verified same day): grade the donor
    # span — extract_span cuts the llama-cli conversation chrome (banner/echo/
    # timing/Exiting...); fallback keeps clean-stdout safe. Cap proxy stays on
    # whole stdout (generation + chrome; probe: 0 truncations at n=300).
    # Truncation voids regardless (anti-poison); otherwise loose last-int on
    # the span (right-but-chatty counts; intermediate-digit scatter to the vote).
    span = extract_span(out)
    toks = out.split()
    lines = [l.strip() for l in span.split('\n') if l.strip()]
    strict = bool(lines) and re.fullmatch(r'-?\d{1,6}', lines[-1]) is not None
    ints = re.findall(r'(?<!\d)-?\d{1,9}(?!\d)', span)
    loose = int(ints[-1]) if ints else None
    if len(toks) >= n - 5:
        return None, True, 'cap', False
    if loose is None:
        return None, True, 'schema', False
    return loose, False, 'ok', (not strict)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--queries', required=True)
    ap.add_argument('--slice', type=int, required=True)
    ap.add_argument('--nslice', type=int, required=True)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--ncap', type=int, default=300)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    global N_CAP
    N_CAP = a.ncap
    os.makedirs(a.out, exist_ok=True)
    qs = json.load(open(a.queries))
    mine = [q for i, q in enumerate(qs) if i % a.nslice == a.slice]
    assert len(mine) > 0, 'FAIL empty slice'
    cli = os.path.join(a.bin, 'llama-cli')
    assert os.path.isfile(cli), 'FAIL no llama-cli'
    rows = []
    for q in mine:
        texts, preds, excl, esch, ecap, loose_n = [], [], 0, 0, 0, 0
        for s in range(a.k):
            out = gen(cli, a.model, q['prompt'], 0.7, s)
            pred, bad, why, loo = grade(out, N_CAP)
            texts.append(out[-2000:])
            if bad:
                excl += 1
                if why in ('schema', 'both'):
                    esch += 1
                if why in ('cap', 'both'):
                    ecap += 1
                preds.append(None)
            else:
                preds.append(pred)
                loose_n += 1 if loo else 0
        valid = [p for p in preds if p is not None]
        counts = {}
        for p in valid:
            counts[p] = counts.get(p, 0) + 1
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        mv8 = (top[0][0] == q['answer']) if top else False
        single = (valid[0] == q['answer']) if valid else False
        oracle = any(p == q['answer'] for p in valid)
        margin = (top[0][1] - (top[1][1] if len(top) > 1 else 0)) if top else 0
        vprompt = q['prompt'] + '\nThe 8 attempts above disagree. Give the correct answer. End your response with the final number on its own line.'
        vfull = 'Question: ' + q['prompt'] + '\n'
        for s in range(a.k):
            ex = texts[s][-600:] if preds[s] is not None else '[excluded]'
            vfull += 'Attempt %d: %s\n' % (s, ex)
        if len(vfull.split()) > 6000:
            print('WARN verifier prompt %d toks, truncating' % len(vfull.split()), flush=True)
            vfull = vfull[:24000]
        vout = gen(cli, a.model, vfull + vprompt, 0.0, 4242, n=N_CAP, c=8192)
        vpred, vbad, vwhy, _ = grade(vout, N_CAP)
        if vbad:
            if vwhy in ('schema', 'both'):
                esch += 1
            if vwhy in ('cap', 'both'):
                ecap += 1
        abstain = (margin == 0) or vbad
        if abstain:
            vpick = top[0][0] if top else None
        else:
            vpick = vpred
        # adaptive margin-2 retrospective simulation on seed order
        stop, run_counts = a.k, {}
        for m in range(1, a.k + 1):
            if preds[m - 1] is not None:
                run_counts[preds[m - 1]] = run_counts.get(preds[m - 1], 0) + 1
            srt = sorted(run_counts.values(), reverse=True)
            # GUARD (void-not-fallback, locked 2026-11-14): zero valid completions so far
            # (all excluded) leaves srt empty — margin 0, keep scanning. The row voids
            # downstream on k_valid==0 with counted exclusions; the run voids in collect.
            mg = (srt[0] - (srt[1] if len(srt) > 1 else 0)) if srt else 0
            if mg >= 2:
                stop = m
                break
        stopmaj = None
        if run_counts:
            stopmaj = max(run_counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        rows.append({'id': q['id'], 'answer': q['answer'], 'single': single,
                     'mv8': mv8, 'oracle': oracle, 'margin': margin,
                     'verifier_ok': (vpick == q['answer']) if vpick is not None else False,
                     'abstained': abstain, 'stop': stop,
                     'stopmaj_ok': (stopmaj == q['answer']) if stopmaj is not None else False,
                     'k_valid': len(valid), 'excluded': excl + (1 if vbad else 0),
                     'excl_schema': esch, 'excl_cap': ecap,
                     'loose_only': loose_n})
        print('exp1b %s single=%s mv8=%s oracle=%s verifier_ok=%s abstained=%s stop=%d k_valid=%d/%d' % (
            q['id'], single, mv8, oracle, rows[-1]['verifier_ok'], abstain, stop, len(valid), a.k), flush=True)
    json.dump({'slice': a.slice, 'k': a.k, 'rows': rows,
               'why': {'schema': sum(r['excl_schema'] for r in rows),
                       'cap': sum(r['excl_cap'] for r in rows)}},
              open('%s/exp1b-slice-%d.json' % (a.out, a.slice), 'w'), indent=1)
    print('WHY schema=%d cap=%d' % (
        sum(r['excl_schema'] for r in rows), sum(r['excl_cap'] for r in rows)), flush=True)

main()
