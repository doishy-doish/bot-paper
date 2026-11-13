// tinygpt-parity: Rust f32 twin for the halt-tinygpt GGUF (arch rev 2).
// Reads the GGUFv3 file written by halt-r/export-gguf.py (no deps), runs the
// exact torch forward (pre-RMSNorm, NeoX-pair RoPE base 10000, fused-QKV row
// split q=[0..E)/k=[E..2E)/v=[2E..3E), causal SDPA scale=1/sqrt(hd), SwiGLU,
// final RMSNorm, untied head) on each ref context, and checks full last-pos
// logits + argmax vs halt-r/ref-*.json. PASS = all argmax equal AND max|diff|
// under the fail bar (warn bar is informational; f32 summation order differs
// from torch, so tiny diffs are expected and reported, not hidden).
use std::collections::HashMap;
use std::fs;

fn rd_u32(d: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes(d[*o..*o + 4].try_into().unwrap());
    *o += 4;
    v
}
fn rd_u64(d: &[u8], o: &mut usize) -> u64 {
    let v = u64::from_le_bytes(d[*o..*o + 8].try_into().unwrap());
    *o += 8;
    v
}
fn rd_str(d: &[u8], o: &mut usize) -> String {
    let n = rd_u64(d, o) as usize;
    let s = String::from_utf8(d[*o..*o + n].to_vec()).unwrap();
    *o += n;
    s
}
fn rd_val(d: &[u8], o: &mut usize, t: u32) -> String {
    match t {
        4 => rd_u32(d, o).to_string(),
        8 => rd_str(d, o),
        9 => {
            let et = rd_u32(d, o);
            let n = rd_u64(d, o);
            let mut v = Vec::new();
            for _ in 0..n {
                v.push(rd_val(d, o, et));
            }
            v.join(",")
        }
        _ => panic!("unsupported kv type {}", t),
    }
}

fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    for i in 0..rows {
        let mut s = 0.0f32;
        let b = i * cols;
        for j in 0..cols {
            s += w[b + j] * x[j];
        }
        y[i] = s;
    }
    y
}

fn rmsnorm(x: &[f32], w: &[f32]) -> Vec<f32> {
    let mut m = 0.0f32;
    for v in x {
        m += v * v;
    }
    m /= x.len() as f32;
    let inv = 1.0 / (m + 1e-6f32).sqrt();
    x.iter().zip(w.iter()).map(|(a, b)| a * inv * b).collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

struct Model {
    e: usize,
    nh: usize,
    nl: usize,
    v: usize,
    mlp: usize,
    t: HashMap<String, Vec<f32>>,
}

impl Model {
    fn get(&self, n: &str) -> &[f32] {
        self.t.get(n).unwrap_or_else(|| panic!("missing tensor {}", n))
    }
    // logits for the full context; returns all positions (caller takes last)
    fn forward(&self, ids: &[usize]) -> Vec<f32> {
        let (e, nh, nl, v, mlp) = (self.e, self.nh, self.nl, self.v, self.mlp);
        let hd = e / nh;
        let tlen = ids.len();
        // embed
        let tok = self.get("tok.weight");
        let mut x: Vec<Vec<f32>> = ids
            .iter()
            .map(|id| tok[id * e..id * e + e].to_vec())
            .collect();
        // rope cache (T, hd/2), NeoX pair order
        let mut cos = vec![0.0f32; tlen * hd / 2];
        let mut sin = vec![0.0f32; tlen * hd / 2];
        for t in 0..tlen {
            for p in 0..hd / 2 {
                let f = t as f32 / 10000f32.powf((2 * p) as f32 / hd as f32);
                cos[t * hd / 2 + p] = f.cos();
                sin[t * hd / 2 + p] = f.sin();
            }
        }
        let rope = |vec: &mut [f32], t: usize| {
            for p in 0..hd / 2 {
                let a = vec[2 * p];
                let b = vec[2 * p + 1];
                let c = cos[t * hd / 2 + p];
                let s = sin[t * hd / 2 + p];
                vec[2 * p] = a * c - b * s;
                vec[2 * p + 1] = a * s + b * c;
            }
        };
        let scale = 1.0 / (hd as f32).sqrt();
        for l in 0..nl {
            let p = format!("blk.{}.", l);
            let g = |n: &str| -> String { format!("{}{}", p, n) };
            // attn
            let mut xn: Vec<Vec<f32>> = Vec::with_capacity(tlen);
            for t in 0..tlen {
                let h = rmsnorm(&x[t], self.get(&g("ln1.w")));
                xn.push(matvec(self.get(&g("attn.qkv_w")), &h, 3 * e, e));
            }
            // xn[t] = raw qkv rows; now attend causally per position
            let mut ks: Vec<Vec<f32>> = Vec::new();
            let mut vs: Vec<Vec<f32>> = Vec::new();
            let mut attn_out: Vec<Vec<f32>> = Vec::with_capacity(tlen);
            for t in 0..tlen {
                let qkv = &xn[t];
                let mut qh = vec![0.0f32; e];
                let mut kh = vec![0.0f32; e];
                let vh = qkv[2 * e..3 * e].to_vec();
                qh.copy_from_slice(&qkv[0..e]);
                kh.copy_from_slice(&qkv[e..2 * e]);
                for h in 0..nh {
                    rope(&mut qh[h * hd..h * hd + hd], t);
                    rope(&mut kh[h * hd..h * hd + hd], t);
                }
                ks.push(kh);
                vs.push(vh);
                // scores vs 0..=t
                let mut sc = vec![0.0f32; t + 1];
                let mut mx = f32::NEG_INFINITY;
                for s in 0..=t {
                    let mut d = 0.0f32;
                    for h in 0..nh {
                        for dd in 0..hd {
                            d += qh[h * hd + dd] * ks[s][h * hd + dd];
                        }
                    }
                    d *= scale;
                    sc[s] = d;
                    if d > mx {
                        mx = d;
                    }
                }
                let mut psum = 0.0f32;
                for s in 0..=t {
                    sc[s] = (sc[s] - mx).exp();
                    psum += sc[s];
                }
                let mut o = vec![0.0f32; e];
                for s in 0..=t {
                    let p = sc[s] / psum;
                    for j in 0..e {
                        o[j] += p * vs[s][j];
                    }
                }
                attn_out.push(o);
            }
            let ow = self.get(&g("attn.out_w"));
            for t in 0..tlen {
                let o = matvec(ow, &attn_out[t], e, e);
                for j in 0..e {
                    x[t][j] += o[j];
                }
            }
            // mlp
            for t in 0..tlen {
                let h = rmsnorm(&x[t], self.get(&g("ln2.w")));
                let ga = matvec(self.get(&g("mlp.gate_w")), &h, mlp, e);
                let up = matvec(self.get(&g("mlp.up_w")), &h, mlp, e);
                let mut hu = vec![0.0f32; mlp];
                for j in 0..mlp {
                    hu[j] = silu(ga[j]) * up[j];
                }
                let dn = matvec(self.get(&g("mlp.down_w")), &hu, e, mlp);
                for j in 0..e {
                    x[t][j] += dn[j];
                }
            }
        }
        // final norm + head, all positions
        let lnf = self.get("ln_f.w");
        let hw = self.get("head.weight");
        let last = rmsnorm(&x[tlen - 1], lnf);
        matvec(hw, &last, v, e)
    }
}

fn argmax(a: &[f32]) -> usize {
    let mut b = 0;
    for i in 1..a.len() {
        if a[i] > a[b] {
            b = i;
        }
    }
    b
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut gguf = String::new();
    let mut refp = String::new();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--gguf" {
            gguf = args[i + 1].clone();
            i += 2;
        } else if args[i] == "--ref" {
            refp = args[i + 1].clone();
            i += 2;
        } else {
            i += 1;
        }
    }
    if gguf.is_empty() || refp.is_empty() {
        eprintln!("usage: tinygpt-parity --gguf <file.gguf> --ref <ref-tag.json>");
        std::process::exit(2);
    }
    let d = fs::read(&gguf).unwrap();
    let mut o = 0usize;
    assert_eq!(rd_u32(&d, &mut o), 0x46554747, "bad magic");
    assert_eq!(rd_u32(&d, &mut o), 3, "bad version");
    let nt = rd_u64(&d, &mut o);
    let nk = rd_u64(&d, &mut o);
    let mut kv: HashMap<String, String> = HashMap::new();
    for _ in 0..nk {
        let k = rd_str(&d, &mut o);
        let t = rd_u32(&d, &mut o);
        kv.insert(k, rd_val(&d, &mut o, t));
    }
    struct TI {
        name: String,
        shape: Vec<usize>,
        off: u64,
    }
    let mut infos = Vec::new();
    for _ in 0..nt {
        let n = rd_str(&d, &mut o);
        let nd = rd_u32(&d, &mut o);
        let mut dims = Vec::new();
        for _ in 0..nd {
            dims.push(rd_u64(&d, &mut o));
        }
        let ty = rd_u32(&d, &mut o);
        assert_eq!(ty, 0, "only F32 supported");
        let off = rd_u64(&d, &mut o);
        dims.reverse(); // stored outer-first -> torch row-major order
        infos.push(TI { name: n, shape: dims, off });
    }
    let align = 32usize;
    let data_start = o + (align - o % align) % align;
    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    for ti in &infos {
        let n: usize = ti.shape.iter().product();
        let b = data_start + ti.off as usize;
        let mut v = Vec::with_capacity(n);
        for k in 0..n {
            let f = f32::from_le_bytes(d[b + 4 * k..b + 4 * k + 4].try_into().unwrap());
            v.push(f);
        }
        assert_eq!(v.len(), n, "short tensor {}", ti.name);
        tensors.insert(ti.name.clone(), v);
    }
    let gi = |k: &str| -> usize { kv.get(k).unwrap_or_else(|| panic!("missing kv {}", k)).parse().unwrap() };
    let (e, nh, nl, v) = (gi("halt.n_embd"), gi("halt.n_head"), gi("halt.n_layer"), gi("halt.vocab_size"));
    let mlp = (8 * e) / 3;
    let model = Model { e, nh, nl, v, mlp, t: tensors };
    // ref json: {"tag":..,"refs":[{"ids":[...],"logits":[...],"argmax":n}]}
    let rj = fs::read_to_string(&refp).unwrap();
    let refs = parse_refs(&rj);
    let mut worst = 0.0f32;
    let mut bad = 0;
    for (ri, r) in refs.iter().enumerate() {
        let got = model.forward(&r.ids);
        assert_eq!(got.len(), r.logits.len(), "V mismatch ref {}", ri);
        let mut md = 0.0f32;
        for k in 0..got.len() {
            let dd = (got[k] - r.logits[k]).abs();
            if dd > md {
                md = dd;
            }
        }
        let ga = argmax(&got);
        let ok = ga == r.argmax && md < 1e-2;
        if !ok {
            bad += 1;
        }
        if md > worst {
            worst = md;
        }
        println!("ref{} n={} argmax got={} want={} maxdiff={:.6} {}", ri, got.len(), ga, r.argmax, md, if ok { "OK" } else { "FAIL" });
    }
    println!("parity: refs={} worst_maxdiff={:.6} fails={} {}", refs.len(), worst, bad, if bad == 0 { "PASS" } else { "FAIL" });
    if bad > 0 {
        std::process::exit(1);
    }
}

struct Ref {
    ids: Vec<usize>,
    logits: Vec<f32>,
    argmax: usize,
}

// minimal JSON scan for the known ref shape (no deps): finds each
// {"ids": [...], "logits": [...], "argmax": N} object in order.
fn parse_refs(s: &str) -> Vec<Ref> {
    let mut out = Vec::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        // find '"ids"'
        if s[i..].starts_with("\"ids\"") {
            i += 5;
            let ids = parse_num_array(s, &mut i);
            // find '"logits"'
            while i < b.len() && !s[i..].starts_with("\"logits\"") {
                i += 1;
            }
            i += 8;
            let logits = parse_num_array_f(s, &mut i);
            while i < b.len() && !s[i..].starts_with("\"argmax\"") {
                i += 1;
            }
            i += 8;
            while i < b.len() && !(b[i].is_ascii_digit()) {
                i += 1;
            }
            let mut j = i;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            let argmax: usize = s[i..j].parse().unwrap();
            i = j;
            let ids: Vec<usize> = ids.iter().map(|x| *x as usize).collect();
            out.push(Ref { ids, logits, argmax });
        } else {
            i += 1;
        }
    }
    assert!(!out.is_empty(), "no refs parsed");
    out
}

fn skip_to_bracket(s: &str, i: &mut usize) {
    let b = s.as_bytes();
    while *i < b.len() && b[*i] != b'[' {
        *i += 1;
    }
    *i += 1; // consume '['
}

fn parse_num_array(s: &str, i: &mut usize) -> Vec<i64> {
    skip_to_bracket(s, i);
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut cur = String::new();
    loop {
        let c = b[*i] as char;
        if c == ']' {
            if !cur.trim().is_empty() {
                out.push(cur.trim().parse().unwrap());
            }
            *i += 1;
            break;
        } else if c == ',' {
            if !cur.trim().is_empty() {
                out.push(cur.trim().parse().unwrap());
            }
            cur.clear();
            *i += 1;
        } else {
            cur.push(c);
            *i += 1;
        }
    }
    out
}

fn parse_num_array_f(s: &str, i: &mut usize) -> Vec<f32> {
    skip_to_bracket(s, i);
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut cur = String::new();
    loop {
        let c = b[*i] as char;
        if c == ']' {
            if !cur.trim().is_empty() {
                out.push(cur.trim().parse().unwrap());
            }
            *i += 1;
            break;
        } else if c == ',' {
            if !cur.trim().is_empty() {
                out.push(cur.trim().parse().unwrap());
            }
            cur.clear();
            *i += 1;
        } else {
            cur.push(c);
            *i += 1;
        }
    }
    out
}
