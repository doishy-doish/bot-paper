# delegation-eval.py: frozen-donor baseline for the all-delegation track (stdlib only).
# Pulls nothing itself (workflow fetches model + corpora). Verifies GGUF magic,
# runs llama-perplexity on small slices + llama-cli generation probes,
# writes delegation-baseline.json/md. No torch, no numpy, no training, ever.
# Usage: python3 halt-r/delegation-eval.py --bin build/bin --model /tmp/qwen.gguf
#        --ts /tmp/ts.txt --shakes /tmp/shakes.txt --out out/delegation
import argparse, json, os, re, subprocess, hashlib

def run(cmd, timeout=900):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or '') + (p.stderr or '')

def rep4(text):
    toks = text.split()
    if len(toks) < 16:
        return 0.0
    grams, reps = {}, 0
    for i in range(len(toks) - 3):
        g = ' '.join(toks[i:i+4])
        grams[g] = grams.get(g, 0) + 1
    for i in range(len(toks) - 3):
        if grams[' '.join(toks[i:i+4])] > 1:
            reps += 1
    return reps / max(1, (len(toks) - 3))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--ts', required=True)
    ap.add_argument('--shakes', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = []
    def say(s):
        print(s, flush=True)
        log.append(s)
    # 1. GGUF verify: magic + size + sha (first 1MB hash for speed, full size recorded)
    st = os.stat(a.model)
    with open(a.model, 'rb') as f:
        magic = f.read(4)
    assert magic == b'GGUF', 'FAIL magic %r' % magic
    h = hashlib.sha256()
    with open(a.model, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    say('model bytes=%d sha256=%s' % (st.st_size, h.hexdigest()))
    cli = os.path.join(a.bin, 'llama-cli')
    ppl = os.path.join(a.bin, 'llama-perplexity')
    assert os.path.isfile(cli), 'FAIL no llama-cli in %s' % a.bin
    assert os.path.isfile(ppl), 'FAIL no llama-perplexity in %s' % a.bin
    # 2. perplexity on 100KB slices (keeps CPU minutes, not hours)
    ppls = {}
    for tag, path in (('ts', a.ts), ('shakes', a.shakes)):
        sl = os.path.join(a.out, 'slice-%s.txt' % tag)
        with open(path, encoding='utf8', errors='replace') as f:
            data = f.read(100 * 1024)
        with open(sl, 'w', encoding='utf8') as f:
            f.write(data)
        rc, out = run([ppl, '-m', a.model, '-f', sl, '-c', '512', '-t', '4'])
        m = re.search(r'Final estimate:\s*PPL\s*=\s*([\d.]+)', out)
        if not m:
            m = re.search(r'perplexity[:\s]+([\d.]+)', out, re.I)
        val = float(m.group(1)) if m else -1.0
        ppls[tag] = val
        say('ppl %s slice-chars=%d PPL=%.2f rc=%d' % (tag, len(data), val, rc))
        assert rc == 0 and val > 0, 'FAIL perplexity %s (see log)' % tag
    # 3. generation probes (fluency + multilingual + reasoning)
    prompts = [
        ('story-en', 'Once upon a time there was a little girl named Lily. She'),
        ('explain-en', 'Explain why the sky is blue in one short paragraph.'),
        ('story-hi', 'Ek chhote gaon ki kahani shuru karo.'),
        ('story-mr', 'Eka chhotya gavachi goshta suru kar.'),
        ('reason-add', 'If there are 3 boxes with 2, 4 and 5 apples, how many apples in total? Reason in one line.'),
        ('carry', 'What is 57 + 68? Show the addition step by step.'),
    ]
    gens = []
    for tag, pr in prompts:
        pf = os.path.join(a.out, 'prompt-%s.txt' % tag)
        with open(pf, 'w', encoding='utf8') as f:
            f.write(pr)
        rc, out = run([cli, '-m', a.model, '-f', pf, '-n', '80', '-c', '2048',
                       '--temp', '0.7', '-t', '4', '-p', pr], timeout=600)
        # llama-cli echoes prompt; take text after last prompt occurrence as generation
        gen = out[out.rfind(pr) + len(pr):].strip() if pr in out else out.strip()
        gen = gen[:2000]
        r4 = rep4(gen)
        ok = len(gen) > 0 and r4 < 0.9
        gens.append({'tag': tag, 'rep4': round(r4, 3), 'chars': len(gen),
                     'ok': ok, 'text': gen[:600]})
        say('gen %s chars=%d rep4=%.3f rc=%d ok=%s' % (tag, len(gen), r4, rc, ok))
        assert rc == 0 and len(gen) > 0, 'FAIL generation %s' % tag
    report = {'model': os.path.basename(a.model), 'bytes': st.st_size,
              'sha256': h.hexdigest(), 'ppl': ppls,
              'gens': [{k: g[k] for k in ('tag', 'rep4', 'chars', 'ok')} for g in gens]}
    json.dump({'report': report, 'samples': gens}, open(a.out + '/delegation-baseline.json', 'w'), indent=1)
    md = '# Delegation baseline: frozen donor alone (measured, CI)\nDonor %s (%d bytes). Perplexity on 100KB slices (ctx 512). Generation: 6 probes, temp 0.7, 80 tokens.\n\n| corpus | PPL |\n|---|---|\n' % (report['model'], report['bytes'])
    for k, v in ppls.items():
        md += '| %s | %.2f |\n' % (k, v)
    md += '\n| probe | chars | rep4 | ok |\n|---|---|---|---|\n'
    for g in gens:
        md += '| %s | %d | %.3f | %s |\n' % (g['tag'], g['chars'], g['rep4'], g['ok'])
    open(a.out + '/delegation-baseline.md', 'w').write(md)
    open(a.out + '/delegation-log.txt', 'w').write('\n'.join(log) + '\n')
    print(md, flush=True)
    print('delegation-eval: done', flush=True)

main()
