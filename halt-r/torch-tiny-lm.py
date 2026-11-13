# torch-tiny-lm.py: the "smaller mature AI" baseline for HALT-R. Minimal char-level GPT,
# CPU-only, torch only. Same data, same 90/10 split, same 100-query seed-2000 protocol
# (LCG identical to the JS/Rust bench), so the numbers are directly comparable.
# Usage: python3 halt-r/torch-tiny-lm.py --data /tmp/corpus.txt --out out --tag shakes --iters 2000
import argparse, json, math, os, time
import torch
import torch.nn as nn

def lcg(seed):
    s = seed & 0xFFFFFFFF
    while True:
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        yield s / 4294967296

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))
    def forward(self, x, mask):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, V, n_embd=64, n_head=2, n_layer=2, block=32):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(V, n_embd)
        self.pos = nn.Embedding(block, n_embd)
        self.layers = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, V, bias=False)
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device)).unsqueeze(0)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=idx.device), diagonal=1)
        for l in self.layers:
            x = l(x, mask)
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
    model = TinyGPT(V, n_embd=64, n_head=2, n_layer=2, block=a.block)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = lcg(a.seed)
    model.train()
    best = None
    for it in range(1, a.iters + 1):
        bx = torch.empty(a.batch, a.block, dtype=torch.long)
        by = torch.empty(a.batch, a.block, dtype=torch.long)
        for b in range(a.batch):
            pos = int(next(rng) * (len(train_ids) - a.block - 1))
            bx[b] = train_ids[pos:pos + a.block]
            by[b] = train_ids[pos + 1:pos + a.block + 1]
        logits = model(bx)
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), by.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 250 == 0 or it == a.iters:
            model.eval()
            with torch.no_grad():
                tot_ce, tot_n = 0.0, 0
                for s in range(0, len(val_ids) - a.block - 1, 512):
                    e = min(s + 512, len(val_ids) - a.block - 1)
                    if e <= s:
                        break
                    vx = torch.stack([val_ids[p:p + a.block] for p in range(s, e)])
                    vy = torch.stack([val_ids[p + 1:p + a.block + 1] for p in range(s, e)])
                    lg = model(vx)
                    tot_ce += nn.functional.cross_entropy(lg.reshape(-1, V), vy.reshape(-1), reduction='sum').item()
                    tot_n += (e - s) * a.block
                vppl = math.exp(tot_ce / tot_n)
            print('iter %d/%d loss=%.4f valPpl=%.2f sec=%d' % (it, a.iters, loss.item(), vppl, time.time() - t0), flush=True)
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
            tot_ce += nn.functional.cross_entropy(lg.reshape(-1, V), vy.reshape(-1), reduction='sum').item()
            tot_acc += (lg.argmax(-1) == vy).sum().item()
            tot_n += (e - s) * a.block
        full = {'acc': tot_acc / tot_n, 'ppl': math.exp(tot_ce / tot_n)}
        # (b) same 100-query protocol as the HALT-R bench (seed 2000, trailing block context)
        rq = lcg(2000)
        q_acc, q_ce = 0, 0.0
        TR = 100
        for _ in range(TR):
            pos = int(next(rq) * (len(val_ids) - a.block - 1))
            ctx = val_ids[pos:pos + a.block].unsqueeze(0)
            tn = int(val_ids[pos + a.block])
            lg = model(ctx)[0, -1]
            pr = torch.softmax(lg, -1)
            if int(pr.argmax()) == tn:
                q_acc += 1
            q_ce += -math.log(pr[tn].item() + 1e-12)
        q100 = {'acc': q_acc / TR, 'ppl': math.exp(q_ce / TR)}
    train_sec = round(time.time() - t0)
    print('RESULT full acc=%.3f ppl=%.2f | q100 acc=%.3f ppl=%.2f | params=%d sec=%d' % (
        full['acc'], full['ppl'], q100['acc'], q100['ppl'], n_params, train_sec), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'baseline.json'), 'w') as f:
        json.dump({'task': 'tiny-transformer-baseline', 'corpus': a.tag, 'V': V, 'params': n_params,
                   'iters': a.iters, 'block': a.block, 'seed': a.seed, 'chars': len(raw),
                   'fullVal': full, 'q100': q100, 'trainSec': train_sec}, f)
    print('torch-tiny-lm: done')

main()
