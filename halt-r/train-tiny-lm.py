# train-tiny-lm.py: trainer for the HALT-R tiny baseline (was torch-tiny-lm.py).
# Role: CI BUILD TOOL ONLY (runs on GitHub runners, pip torch CPU). It never ships and
# never touches a user device: shipped artifacts are tinygpt-*.gguf + vocab + ref +
# q100-gpt (plain data), consumed by the JS bench and (soon) the Rust twin. Python-free
# at inference by design. New research code (BPE, learned encoder) is JS-only.
# Arch rev 2 (2026-11-13): pre-RMSNorm decoder, RoPE (no absolute pos embedding),
# fused QKV + SDPA, SwiGLU MLP, cosine schedule + warmup + grad clip.
# Same CLI, same data, same 90/10 split, same 100-query seed-2000 protocol (LCG
# identical to the JS/Rust bench), so numbers stay directly comparable across revs.
# Usage: python3 halt-r/train-tiny-lm.py --data input.txt --out out --tag shakes --iters 2000
import argparse, json, math, os, time
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
    return torch.cos(f), torch.sin(f)  # (T, Dh/2) each

def apply_rope(x, cos, sin):
    # x: (B,H,T,Dh) with Dh even; rotate pairs
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
        assert n_embd % n_head == 0
        n_mlp = (8 * n_embd) // 3  # SwiGLU param-parity with 4E GELU MLP
        self.block = block
        self.hd = n_embd // n_head
        self.tok = nn.Embedding(V, n_embd)
        self.layers = nn.ModuleList([Block(n_embd, n_head, n_mlp) for _ in range(n_layer)])
        self.ln = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, V, bias=False)
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx)
        cos, sin = rope_cache(T, self.hd, device=idx.device)
        for l in self.layers:
            x = l(x, cos, sin)
        return self.head(self.ln(x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='shakes')
    ap.add_argument('--iters', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--block', type=int, default=32)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--n-embd', type=int, default=64)
    ap.add_argument('--n-head', type=int, default=2)
    ap.add_argument('--n-layer', type=int, default=2)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    t0 = time.time()
    raw = open(a.data, encoding='utf8').read()
    cut = len(raw) * 9 // 10
    train, val = raw[:cut], raw[cut:]
    vocab = sorted(set(train))
    V = len(vocab)
    if V > 200:
        raise SystemExit('FAIL vocab blowup V=%d' % V)
    stoi = {c: i for i, c in enumerate(vocab)}
    train_ids = torch.tensor([stoi[c] for c in train], dtype=torch.long)
    val_ids = torch.tensor([stoi.get(c, 0) for c in val], dtype=torch.long)
    print('corpus chars=%d train=%d val=%d V=%d iters=%d' % (len(raw), len(train), len(val), V, a.iters), flush=True)
    model = TinyGPT(V, n_embd=a.n_embd, n_head=a.n_head, n_layer=a.n_layer, block=a.block)
    n_params = sum(p.numel() for p in model.parameters())
    peak = 3e-4 if a.n_embd <= 64 else 2e-4
    opt = torch.optim.AdamW(model.parameters(), lr=peak)
    warm = max(1, a.iters // 10)
    def lr_at(it):
        if it < warm:
            return peak * it / warm
        prog = (it - warm) / max(1, a.iters - warm)
        return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
    rng = lcg(a.seed)
    # FRONTIER SCALE-DOWN (2022-2023 standard practice, written down 2026-11-13):
    # never full-val inside the loop. Fixed 2048-window subset checkpoints every 500 + full val once at end.
    eval_rng = lcg(999)
    n_eval_sub = 2048
    eval_pos = [int(next(eval_rng) * (len(val_ids) - a.block - 1)) for _ in range(n_eval_sub)]
    eval_vx = torch.stack([val_ids[p:p + a.block] for p in eval_pos])
    eval_vy = torch.stack([val_ids[p + 1:p + a.block + 1] for p in eval_pos])
    try:
        model = torch.compile(model)  # PyTorch 2.0 (2023 frontier), free 1.5-2x where supported
        compiled = True
    except Exception:
        compiled = False
    print('arch rev=2 rope+rms+swiglu+sdpa sched=cosine-warmup compile=%s evalSub=%d' % (compiled, n_eval_sub), flush=True)
    model.train()
    best = None
    for it in range(1, a.iters + 1):
        bx = torch.empty(a.batch, a.block, dtype=torch.long)
        by = torch.empty(a.batch, a.block, dtype=torch.long)
        for b in range(a.batch):
            pos = int(next(rng) * (len(train_ids) - a.block - 1))
            bx[b] = train_ids[pos:pos + a.block]
            by[b] = train_ids[pos + 1:pos + a.block + 1]
        opt.param_groups[0]['lr'] = lr_at(it)
        logits = model(bx)
        loss = F.cross_entropy(logits.reshape(-1, V), by.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 500 == 0 or it == a.iters:
            model.eval()
            with torch.no_grad():
                lg = model(eval_vx)
                sub_ce = F.cross_entropy(lg.reshape(-1, V), eval_vy.reshape(-1)).item()
                vppl = math.exp(sub_ce)
            print('iter %d/%d loss=%.4f lr=%.2e subPpl=%.2f sec=%d' % (it, a.iters, loss.item(), opt.param_groups[0]['lr'], vppl, time.time() - t0), flush=True)
            if best is None or vppl < best[0]:
                best = (vppl, {k: v.detach().clone() for k, v in model.state_dict().items()})
            model.train()
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        # (a) full-val acc/ppl
        tot_ce, tot_acc, tot_n = 0.0, 0, 0
        for s in range(0, len(val_ids) - a.block - 1, 512):
            e = min(s + 512, len(val_ids) - a.block - 1)
            if e <= s:
                break
            vx = torch.stack([val_ids[p:p + a.block] for p in range(s, e)])
            vy = torch.stack([val_ids[p + 1:p + a.block + 1] for p in range(s, e)])
            lg = model(vx)
            tot_ce += F.cross_entropy(lg.reshape(-1, V), vy.reshape(-1), reduction='sum').item()
            tot_acc += (lg.argmax(-1) == vy).sum().item()
            tot_n += (e - s) * a.block
        full = {'acc': tot_acc / tot_n, 'ppl': math.exp(tot_ce / tot_n)}
        # (b) same 100-query protocol as the HALT-R bench (seed 2000, trailing block context),
        # plus the seed-2001 report stream. Per-query GPT distributions are saved for the
        # hop+GPT mixing arm (gmix): bench joins on pos, so contexts are IDENTICAL.
        gstreams = {}
        for seed in (2000, 2001):
            rq = lcg(seed)
            q_acc, q_ce = 0, 0.0
            TR = 100
            qs = []
            for _ in range(TR):
                pos = int(next(rq) * (len(val_ids) - a.block - 1))
                ctx = val_ids[pos:pos + a.block].unsqueeze(0)
                tn = int(val_ids[pos + a.block])
                lg = model(ctx)[0, -1]
                pr = torch.softmax(lg, -1)
                if int(pr.argmax()) == tn:
                    q_acc += 1
                q_ce += -math.log(pr[tn].item() + 1e-12)
                qs.append({'pos': pos, 'tn': tn, 'probs': [round(float(x), 6) for x in pr]})
            gstreams[seed] = {'acc': q_acc / TR, 'ppl': math.exp(q_ce / TR), 'queries': qs}
        q100 = gstreams[2000]
    train_sec = round(time.time() - t0)
    print('RESULT full acc=%.3f ppl=%.2f | q100 acc=%.3f ppl=%.2f | params=%d sec=%d' % (
        full['acc'], full['ppl'], q100['acc'], q100['ppl'], n_params, train_sec), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'baseline.json'), 'w') as f:
        json.dump({'task': 'tiny-transformer-baseline', 'corpus': a.tag, 'V': V, 'params': n_params,
                   'arch': {'n_embd': a.n_embd, 'n_head': a.n_head, 'n_layer': a.n_layer, 'rev': 2,
                            'rope': True, 'rope_base': 10000, 'norm': 'rms', 'mlp': 'swiglu',
                            'attn': 'sdpa-fused-qkv', 'sched': 'cosine-warmup-10pct', 'pos': 'rope-only',
                            'clip': 1.0},
                   'iters': a.iters, 'block': a.block, 'seed': a.seed, 'chars': len(raw),
                   'fullVal': full, 'q100': {'acc': q100['acc'], 'ppl': q100['ppl']}, 'trainSec': train_sec}, f)
    with open(os.path.join(a.out, 'q100-gpt-%s.json' % a.tag), 'w') as f:
        json.dump({'tag': a.tag, 'block': a.block, 'V': V, 'vocab': vocab,
                   'streams': {str(s): gstreams[s] for s in gstreams}}, f)
    # GGUF-track artifacts: raw weights (compile-prefix stripped), vocab, reference
    # logits for Rust-twin parity (5 fixed contexts, full last-pos logit vectors).
    stripped = {(k[10:] if k.startswith('_orig_mod.') else k): v for k, v in best[1].items()}
    torch.save(stripped, os.path.join(a.out, 'tiny.pt'))
    with open(os.path.join(a.out, 'vocab.json'), 'w') as f:
        json.dump({'vocab': vocab, 'V': V, 'block': a.block,
                   'n_embd': a.n_embd, 'n_head': a.n_head, 'n_layer': a.n_layer}, f)
    ref_rng, refs = lcg(1234), []
    for _ in range(5):
        pos = int(next(ref_rng) * (len(val_ids) - a.block - 1))
        ctx = val_ids[pos:pos + a.block].unsqueeze(0)
        lg = model(ctx)[0, -1].tolist()
        refs.append({'ids': ctx[0].tolist(), 'logits': lg,
                     'argmax': int(torch.tensor(lg).argmax())})
    with open(os.path.join(a.out, 'ref-%s.json' % a.tag), 'w') as f:
        json.dump({'tag': a.tag, 'refs': refs}, f)
    print('train-tiny-lm: done')

main()
