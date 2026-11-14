# exp1-vote.py: matched-K sampling for verifier-vs-majority + headroom gate (stdlib only).
# One matrix slice: runs llama-cli K times per query (temp 0.7, distinct seeds),
# grades by last-integer rule, writes slice JSON. No torch, no numpy, no training.
# Usage: python3 halt-r/exp1-vote.py --bin /tmp/llama.cpp/build/bin --model /tmp/qwen.gguf
#        --queries halt-r/exp1-queries.json --slice 0 --nslice 4 --k 8 --out out/exp1
import argparse, json, os, re, subprocess

def run(cmd, timeout=600):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or '') + (p.stderr or '')

def last_int(text):
    ms = re.findall(r'-?\d+', text)
    return int(ms[-1]) if ms else None

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
        hits = []
        for s in range(a.k):
            rc, out = run([cli, '-m', a.model, '-p', q['prompt'], '-n', '40',
                           '-c', '2048', '--temp', '0.7', '--seed', str(s),
                           '-t', '4', '--no-display-prompt'])
            pred = last_int(out)
            hits.append({'seed': s, 'rc': rc, 'pred': pred,
                         'ok': (pred == q['answer'])})
            assert rc == 0, 'FAIL llama-cli rc=%d' % rc
        votes = {}
        for h in hits[:4]:
            if h['pred'] is not None:
                votes[h['pred']] = votes.get(h['pred'], 0) + 1
        mv4 = max(votes, key=votes.get) if votes else None
        votes = {}
        for h in hits:
            if h['pred'] is not None:
                votes[h['pred']] = votes.get(h['pred'], 0) + 1
        mv8 = max(votes, key=votes.get) if votes else None
        rows.append({'id': q['id'], 'answer': q['answer'],
                     'single': hits[0]['ok'], 'mv4': (mv4 == q['answer']),
                     'mv8': (mv8 == q['answer']),
                     'oracle': any(h['ok'] for h in hits),
                     'hits': [h['pred'] for h in hits]})
        print('exp1 %s single=%s mv4=%s mv8=%s oracle=%s preds=%s' % (
            q['id'], rows[-1]['single'], rows[-1]['mv4'],
            rows[-1]['mv8'], rows[-1]['oracle'], rows[-1]['hits']), flush=True)
    json.dump({'slice': a.slice, 'k': a.k, 'rows': rows},
              open('%s/exp1-slice-%d.json' % (a.out, a.slice), 'w'), indent=1)
    print('exp1 slice %d done (%d queries)' % (a.slice, len(rows)), flush=True)

main()
