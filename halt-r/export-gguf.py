# export-gguf.py: sloppy-python -> GGUF. Converts a train-tiny-lm.py tiny.pt +
# vocab.json into a GGUF v3 file (arch "halt-tinygpt" rev 2) + verifies it by reading
# the bytes back. Only deps: torch + struct (both already in the baseline job).
#
# Tensor layout for the Rust twin (row-major, F32) — rev 2 (RoPE, no abs pos):
#   tok.weight (V,E),
#   blk.L attn.qkv_w (3E,E, no bias), attn.out_w (E,E, no bias),
#   ln1/ln2/final ln.{w} (E, RMSNorm scale only, no bias),
#   mlp.gate_w (H,E), mlp.up_w (H,E), mlp.down_w (E,H, all no bias).
# Forward notes (must match torch exactly): pre-RMSNorm blocks, fused QKV split into
# q/k/v heads, RoPE base 10000 on q/k (head_dim even), causal SDPA scale=1/sqrt(head_dim),
# SwiGLU MLP silu(gate)*up -> down, final RMSNorm, untied head.
# Usage: python3 halt-r/export-gguf.py --pt out/tiny.pt --vocab out/vocab.json
#        --ref out/ref-tag.json --out out/tinygpt-tag.gguf
import argparse, hashlib, json, struct

MAGIC, VERSION, ALIGN = 0x46554747, 3, 32
T_F32 = 0
K_U8, K_I8, K_U16, K_I16, K_U32, K_I32, K_F32, K_BOOL, K_STR, K_ARR, K_U64, K_I64, K_F64 = range(13)

def w_str(b, s):
    e = s.encode('utf-8')
    b += struct.pack('<Q', len(e)) + e

def w_val(b, t, v):
    if t == K_U32: b += struct.pack('<I', v)
    elif t == K_I32: b += struct.pack('<i', v)
    elif t == K_U64: b += struct.pack('<Q', v)
    elif t == K_F32: b += struct.pack('<f', v)
    elif t == K_BOOL: b += struct.pack('<?', bool(v))
    elif t == K_STR: w_str(b, v)
    elif t == K_ARR:
        et, lst = v
        b += struct.pack('<I', et) + struct.pack('<Q', len(lst))
        for x in lst: w_val(b, et, x)
    else: raise SystemExit('FAIL bad kv type %r' % t)

def r_str(d, o):
    (n,) = struct.unpack_from('<Q', d, o); o += 8
    return d[o:o + n].decode('utf-8'), o + n

def r_val(d, o, t):
    if t == K_U32: return struct.unpack_from('<I', d, o)[0], o + 4
    if t == K_I32: return struct.unpack_from('<i', d, o)[0], o + 4
    if t == K_U64: return struct.unpack_from('<Q', d, o)[0], o + 8
    if t == K_F32: return struct.unpack_from('<f', d, o)[0], o + 4
    if t == K_BOOL: return struct.unpack_from('<?', d, o)[0], o + 1
    if t == K_STR: return r_str(d, o)
    if t == K_ARR:
        (et,) = struct.unpack_from('<I', d, o); o += 4
        (n,) = struct.unpack_from('<Q', d, o); o += 8
        out = []
        for _ in range(n):
            v, o = r_val(d, o, et); out.append(v)
        return (et, out), o
    raise SystemExit('FAIL bad kv type %r' % t)

ORDER = ['tok.weight']
def layer_names(l):
    p = 'blk.%d.' % l
    return [p + n for n in ('ln1.w', 'attn.qkv_w', 'attn.out_w', 'ln2.w',
                            'mlp.gate_w', 'mlp.up_w', 'mlp.down_w')]
TAIL = ['ln_f.w', 'head.weight']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', required=True)
    ap.add_argument('--vocab', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    sd = torch.load(a.pt, map_location='cpu', weights_only=True)
    cfg = json.load(open(a.vocab))
    V, E, T, H, L = cfg['V'], cfg['n_embd'], cfg['block'], cfg['n_head'], cfg['n_layer']
    ref = json.load(open(a.ref))
    names = list(ORDER)
    for l in range(L): names += layer_names(l)
    names += TAIL
    src = {
        'tok.weight': sd['tok.weight'],
        'ln_f.w': sd['ln.w'],
        'head.weight': sd['head.weight'],
    }
    for l in range(L):
        base, p = 'layers.%d.' % l, 'blk.%d.' % l
        src[p + 'ln1.w'] = sd[base + 'ln1.w']
        src[p + 'attn.qkv_w'] = sd[base + 'qkv.weight']
        src[p + 'attn.out_w'] = sd[base + 'out.weight']
        src[p + 'ln2.w'] = sd[base + 'ln2.w']
        src[p + 'mlp.gate_w'] = sd[base + 'gate.weight']
        src[p + 'mlp.up_w'] = sd[base + 'up.weight']
        src[p + 'mlp.down_w'] = sd[base + 'down.weight']
    blobs, total = [], 0
    for n in names:
        arr = src[n].detach().to(torch.float32).contiguous().numpy()
        blobs.append((n, arr))
        total += arr.size
    kv = [('general.architecture', K_STR, 'halt-tinygpt'),
          ('general.name', K_STR, 'halt-tinygpt-char'),
          ('general.alignment', K_U32, ALIGN),
          ('halt.n_embd', K_U32, E), ('halt.n_head', K_U32, H),
          ('halt.n_layer', K_U32, L), ('halt.block', K_U32, T),
          ('halt.vocab_size', K_U32, V),
          ('halt.vocab', K_ARR, (K_STR, cfg['vocab'])),
          ('halt.arch_rev', K_U32, 2), ('halt.norm', K_STR, 'rms'),
           ('halt.mlp', K_STR, 'swiglu'), ('halt.pos', K_STR, 'rope-only'),
           ('halt.rope_base', K_STR, '10000'), ('halt.mha_scale', K_STR, 'inv-sqrt-head-dim')]
    b = bytearray()
    b += struct.pack('<I', MAGIC) + struct.pack('<I', VERSION)
    b += struct.pack('<Q', len(names)) + struct.pack('<Q', len(kv))
    for k, t, v in kv:
        w_str(b, k); b += struct.pack('<I', t); w_val(b, t, v)
    off = 0
    for n, arr in blobs:
        w_str(b, n)
        b += struct.pack('<I', len(arr.shape))
        for d in reversed(arr.shape): b += struct.pack('<Q', d)
        b += struct.pack('<I', T_F32) + struct.pack('<Q', off)
        nb = arr.nbytes
        off += nb + (-nb % ALIGN)
    b += bytes(-len(b) % ALIGN)
    data_start = len(b)
    for n, arr in blobs:
        raw = arr.tobytes()
        b += raw + bytes(-len(raw) % ALIGN)
    open(a.out, 'wb').write(bytes(b))
    # verify: parse back, check every tensor byte-exact
    d, o = bytes(b), 0
    assert struct.unpack_from('<I', d, o)[0] == MAGIC; o += 4
    assert struct.unpack_from('<I', d, o)[0] == VERSION; o += 4
    (nt,) = struct.unpack_from('<Q', d, o); o += 8
    (nk,) = struct.unpack_from('<Q', d, o); o += 8
    assert nt == len(names) and nk == len(kv), (nt, nk)
    for k, t, v in kv:
        kk, o = r_str(d, o)
        (tt,) = struct.unpack_from('<I', d, o); o += 4
        vv, o = r_val(d, o, tt)
        assert kk == k and tt == t and vv == v, (kk, k)
    for n, arr in blobs:
        nn, o = r_str(d, o)
        (nd,) = struct.unpack_from('<I', d, o); o += 4
        dims = [struct.unpack_from('<Q', d, o + 8 * i)[0] for i in range(nd)]; o += 8 * nd
        (tt,) = struct.unpack_from('<I', d, o); o += 4
        (oo,) = struct.unpack_from('<Q', d, o); o += 8
        assert nn == n and tt == T_F32 and list(reversed(dims)) == list(arr.shape), (nn, dims, arr.shape)
        raw = d[data_start + oo:data_start + oo + arr.nbytes]
        assert hashlib.sha256(raw).hexdigest() == hashlib.sha256(arr.tobytes()).hexdigest(), n
    report = {'gguf': a.out, 'bytes': len(b), 'tensors': len(names),
              'params': total, 'sha256': hashlib.sha256(bytes(b)).hexdigest(),
              'verify': 'byte-exact', 'refs': len(ref['refs'])}
    print('export-gguf: %s bytes=%d tensors=%d params=%d sha256=%s' % (
        a.out, len(b), len(names), total, report['sha256']), flush=True)
    print('export-gguf: verified byte-exact (%d tensors read back)' % len(names), flush=True)

main()
