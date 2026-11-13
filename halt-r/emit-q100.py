# emit-q100.py: emit bench-joinable donor dists for ANY (donor, corpus, W) triple.
# Why this file exists: the trainer's q100-gpt file joins the bench on val positions,
# which coincide iff bench W == trainer block (32). combo at W8/16 was void (cover
# 1-2/100) for exactly this reason. This emitter replays the bench's own query
# sampling (same LCG, same 90/10 char cut, same W, same seeds 2000/2001) through the
# frozen donor, so cover is 100/100 by construction at any W. No training, ever.
# Arch is a copy of train-tiny-lm.py rev 2 (RoPE/RMSNorm/SwiGLU/fused-QKV, no pos
# embedding so any T works). If the trainer arch revs, copy it here again.
# Outputs: q100 file in trainer schema (bench --gptq accepts it) + donor-alone.json
# (donor-alone acc/ppl per stream = the "before ingestion" pole of the difference
# measurement). Deps: torch only (no numpy: tolist, never .numpy()).
import argparse, json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

def lcg(seed):
    s = seed & 0xFFFFFFFF
    while True:
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        yield s / 4294967296

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

def rope_cache(T, Dh, base=10000.0, device='cpu'):
    inv = 1.0 / (base ** (torch.arange(0, Dh, 2, device=device).float() / Dh))
    t = torch.arange(T, device=device).float()
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)

def apply_rope(x, cos, sin):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_mlp):
        super().__init__()
        assert n_embd % n_head == 0 and (n_embd // n_head) % 2 == 0
        self.n_head = n_head
        self.hd = n_embd // n_head
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)
        self.gate = nn.Linear(n_embd, n_mlp, bias=False)
        self.up = nn.Linear(n_embd, n_mlp, bias=False)
        self.down = nn.Linear(n_mlp, n_embd, bias=False)
    def forward(self, x, cos, sin):
        B, T, E = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_head, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(a.transpose(1, 2).reshape(B, T, E))
        h = self.ln2(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x

class TinyGPT(nn.Module):
    def __init__(self, V, n_embd=64, n_head=2, n_layer=2, block=32):
        super().__init__()
        n_mlp = (8 * n_embd) // 3
        self.block = block
        self.hd = n_embd // n_head
        self.tok = nn.Embedding(V, n_embd)
        self.layers = nn.ModuleList([Block(n_embd, n_head, n_mlp) for _ in range(n_layer)])
        self.ln = RMSNorm(n_embd)
        self.head = nn.Linear(V, n_embd, bias=False)
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx)
        cos, sin = rope_cache(T, self.hd, device=idx.device)
        for l in self.layers:
            x = l(x, cos, sin)
        return self.head(self.ln(x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', required=True)
    ap.add_argument('--vocab', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--W', type=int, required=True)
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(a.vocab))
    V, vocab = cfg['V'], cfg['vocab']
    model = TinyGPT(V, n_embd=cfg['n_embd'], n_head=cfg['n_head'],
                    n_layer=cfg['n_layer'], block=cfg.get('block', 32))
    sd = torch.load(a.pt, map_location='cpu', weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    raw = open(a.data, encoding='utf8').read()
    cut = len(raw) * 9 // 10
    train, val = raw[:cut], raw[cut:]
    stoi = {c: i for i, c in enumerate(vocab)}
    oov = [c for c in set(val) if c not in stoi]
    if oov:
        raise SystemExit('FAIL val has OOV chars %r (vocab order would misalign)' % oov[:10])
    if sorted(set(train)) != sorted(vocab):
        raise SystemExit('FAIL train vocab != donor vocab (positional misalignment)')
    val_ids = torch.tensor([stoi[c] for c in val], dtype=torch.long)
    streams, alone = {}, {}
    with torch.no_grad():
        for seed in (2000, 2001):
            rq = lcg(seed)
            q_acc, q_ce = 0, 0.0
            qs = []
            for _ in range(a.trials):
                pos = int(next(rq) * (len(val_ids) - a.W - 1))
                ctx = val_ids[pos:pos + a.W].unsqueeze(0)
                tn = int(val_ids[pos + a.W])
                lg = model(ctx)[0, -1]
                pr = torch.softmax(lg, -1).tolist()
                if int(torch.tensor(pr).argmax()) == tn:
                    q_acc += 1
                q_ce += -math.log(max(pr[tn], 1e-12))
                qs.append({'pos': pos, 'tn': tn,
                           'probs': [round(float(x), 6) for x in pr]})
            streams[seed] = {'acc': q_acc / a.trials, 'ppl': math.exp(q_ce / a.trials),
                             'queries': qs}
            alone[seed] = {'acc': streams[seed]['acc'], 'ppl': streams[seed]['ppl']}
            print('stream %d donor-alone acc=%.3f ppl=%.2f' % (
                seed, alone[seed]['acc'], alone[seed]['ppl']), flush=True)
    out_q = {'tag': a.tag, 'block': a.W, 'V': V, 'vocab': vocab,
             'streams': {str(s): streams[s] for s in streams}}
    json.dump(out_q, open(a.out + '/q100-gpt-%s.json' % a.tag, 'w'))
    json.dump({'tag': a.tag, 'W': a.W, 'trials': a.trials, 'params': n_params,
               'donorAlone': {str(s): alone[s] for s in alone}},
              open(a.out + '/donor-alone.json', 'w'), indent=1)
    print('emit-q100: done cover=%d/%d per stream' % (a.trials, a.trials))

main()
