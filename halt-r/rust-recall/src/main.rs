// halt-r-cli: terminal twin of the HALT-R retrieval LM. Zero dependencies, std only.
//   halt-r-cli selftest
//   halt-r-cli lm --data <txt> [--W 8] [--P 8000] [--trials 100] [--proj sim|raw] [--betas 0.1,0.3,1,3,10]
//   halt-r-cli lm --synth <chars> [...]   (offline smoke: LCG pseudo-text over 27 chars)
// Mirrors lm-shakespeare.mjs on purpose: same LCG streams, same Laplace bigram, same beta
// probe (seed 1000), same lam select-seed-2000 / report-seed-2001, same single-step
// self-key check (seed 999). Vocab sort is by char scalar value (== UTF-16 order for ASCII).
// Gaussian projection uses the same Box-Muller formula (f64 libm may differ in last ulp;
// SimHash signs can flip rarely — same protocol, not claimed bit-identical).
// Local-only tool: prints metrics, never encrypts anything, never uploads anything.
use halt_r_recall::recall_step;

const LAMBDAS: [f64; 7] = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];

struct Lcg(u32);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 as f64) / 4294967296.0
    }
}

fn fail(msg: &str) -> ! {
    eprintln!("FAIL {}", msg);
    std::process::exit(1);
}

fn opt(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|x| x == k).and_then(|i| args.get(i + 1).cloned())
}
fn has(args: &[String], k: &str) -> bool {
    args.iter().any(|x| x == k)
}
fn parse_usize(args: &[String], k: &str, def: usize) -> usize {
    opt(args, k).map(|v| v.parse::<usize>().unwrap_or_else(|_| fail(&format!("bad {}", k)))).unwrap_or(def)
}

fn argmax(v: &[f64]) -> usize {
    let mut bi = 0;
    for i in 1..v.len() {
        if v[i] > v[bi] {
            bi = i;
        }
    }
    bi
}

fn feat_raw(ids: &[usize], pos: usize, w: usize, v: usize, out: &mut [f32]) {
    for x in out.iter_mut() {
        *x = -1.0;
    }
    for k in 0..w {
        out[k * v + ids[pos + k]] = 1.0;
    }
}

fn feat_sim(ids: &[usize], pos: usize, w: usize, v: usize, h: usize, rmat: &[f32], out: &mut [f32]) {
    for x in out.iter_mut() {
        *x = 0.0;
    }
    for k in 0..w {
        let base = (k * v + ids[pos + k]) * h;
        for hh in 0..h {
            out[hh] += rmat[base + hh];
        }
    }
    for x in out.iter_mut() {
        *x = if *x >= 0.0 { 1.0 } else { -1.0 };
    }
}

fn run_recall(
    q: &[f32], pats: &[f32], p: usize, d: usize, beta: f32, max_iters: usize,
    state: &mut [f32], scratch: &mut [f32],
) -> usize {
    state.copy_from_slice(q);
    let mut done = 0;
    for _ in 0..max_iters {
        done += 1;
        if recall_step(pats, state, scratch, p, d, beta) < 1e-3 {
            break;
        }
    }
    done
}

fn dist_of_weights(scratch: &[f32], next: &[usize], v: usize) -> Vec<f64> {
    let mut tot = 0.0f64;
    for &w in scratch {
        tot += w as f64;
    }
    let mut dist = vec![0.0f64; v];
    for (i, &w) in scratch.iter().enumerate() {
        dist[next[i]] += w as f64;
    }
    let denom = tot + 1e-12 * v as f64;
    for d in dist.iter_mut() {
        *d /= denom;
    }
    dist
}

fn synth_text(n: usize, seed: u32) -> Vec<char> {
    let mut r = Lcg(seed);
    let mut s = String::new();
    while s.len() < n {
        let wl = 1 + (r.next_f64() * 8.0) as usize;
        for _ in 0..wl {
            s.push((b'a' + (r.next_f64() * 26.0) as u8) as char);
        }
        s.push(' ');
    }
    s.truncate(n);
    s.chars().collect()
}

fn selftest() {
    let mut fails = 0;
    let check = |name: &str, cond: bool, detail: String, fails: &mut i32| {
        println!("{} {} {}", if cond { "PASS" } else { "FAIL" }, name, detail);
        if !cond {
            *fails += 1;
        }
    };
    // tiny synthetic store, same shape as test-recall.mjs
    let (p, d) = (64usize, 64usize);
    let mut rng = Lcg(7);
    let mut pats = vec![0.0f32; p * d];
    for x in pats.iter_mut() {
        *x = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
    }
    let k = 3usize;
    let mut q = pats[k * d..(k + 1) * d].to_vec();
    for _ in 0..8 {
        let j = (rng.next_f64() * d as f64) as usize;
        q[j] = -q[j];
    }
    let mut state = vec![0.0f32; d];
    let mut scratch = vec![0.0f32; p];
    let n = run_recall(&q, &pats, p, d, 8.0, 12, &mut state, &mut scratch);
    check("cli-runs-iters", n <= 12, format!("n={}", n), &mut fails);
    let mut bi = 0;
    let mut bd = f32::INFINITY;
    for i in 0..p {
        let mut ham = 0.0f32;
        for j in 0..d {
            ham += (state[j] - pats[i * d + j]).abs();
        }
        if ham < bd {
            bd = ham;
            bi = i;
        }
    }
    check("cli-retrieves", bi == k, format!("nearest={} target={}", bi, k), &mut fails);
    // determinism: run again, bit-for-bit
    let mut state2 = vec![0.0f32; d];
    let mut scratch2 = vec![0.0f32; p];
    run_recall(&q, &pats, p, d, 8.0, 12, &mut state2, &mut scratch2);
    check("cli-deterministic", state == state2, "".to_string(), &mut fails);
    if fails > 0 {
        fail(&format!("{} selftest failures", fails));
    }
    println!("cli-selftest: all green");
}

fn cmd_lm(args: &[String]) {
    let w = parse_usize(args, "--W", 8);
    let p = parse_usize(args, "--P", 8000);
    let tr = parse_usize(args, "--trials", 100);
    let proj = opt(args, "--proj").unwrap_or_else(|| "sim".to_string());
    let h: usize = 256;
    let betas: Vec<f64> = opt(args, "--betas")
        .unwrap_or_else(|| "0.1,0.3,1.0,3.0,10.0".to_string())
        .split(',')
        .map(|s| s.trim().parse::<f64>().unwrap_or_else(|_| fail("bad --betas")))
        .collect();
    let text: Vec<char> = if let Some(n) = opt(args, "--synth") {
        let nch: usize = n.parse().unwrap_or_else(|_| fail("bad --synth"));
        synth_text(nch, 7)
    } else if let Some(path) = opt(args, "--data") {
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|_| fail("cannot read --data"));
        raw.chars().collect()
    } else {
        fail("need --data <txt> or --synth <chars>");
    };
    let cut = text.len() * 9 / 10;
    let (train, val) = text.split_at(cut);
    let mut vocab: Vec<char> = train.to_vec();
    vocab.sort_unstable();
    vocab.dedup();
    let v = vocab.len();
    if v > 200 {
        fail(&format!("vocab blowup V={}", v));
    }
    let id_of = |c: char| -> usize { vocab.binary_search(&c).unwrap_or(0) };
    let train_ids: Vec<usize> = train.iter().map(|&c| id_of(c)).collect();
    let val_ids: Vec<usize> = val.iter().map(|&c| vocab.binary_search(&c).unwrap_or(0)).collect();
    println!("corpus chars={} train={} val={} V={} W={} P={} proj={}", text.len(), train.len(), val.len(), v, w, p, proj);
    // unigram + bigram on full train
    let mut uni = vec![0.0f64; v];
    for &t in &train_ids {
        uni[t] += 1.0;
    }
    let ntr = train_ids.len() as f64;
    for u in uni.iter_mut() {
        *u /= ntr;
    }
    let uni_pred = argmax(&uni);
    let mut bi = vec![0.0f64; v * v];
    for i in 0..train_ids.len().saturating_sub(1) {
        bi[train_ids[i] * v + train_ids[i + 1]] += 1.0;
    }
    // store
    let din = w * v;
    let d = if proj == "sim" { h } else { din };
    let mut rmat = vec![0.0f32; din * h];
    if proj == "sim" {
        let mut rr = Lcg(8); // SEED+1, same stream as JS
        let inv = (1.0 / din as f64).sqrt();
        for x in rmat.iter_mut() {
            let mut u = 0.0;
            while u == 0.0 {
                u = rr.next_f64();
            }
            let mut vv = 0.0;
            while vv == 0.0 {
                vv = rr.next_f64();
            }
            *x = ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * vv).cos() * inv) as f32;
        }
    }
    let feat = |ids: &[usize], pos: usize, out: &mut [f32]| {
        if proj == "sim" {
            feat_sim(ids, pos, w, v, h, &rmat, out);
        } else {
            feat_raw(ids, pos, w, v, out);
        }
    };
    let mut rng = Lcg(7);
    let mut pats = vec![0.0f32; p * d];
    let mut next = vec![0usize; p];
    let mut spos = vec![0usize; p]; // stored train positions (self-key check re-featurizes these, like JS)
    let mut q = vec![0.0f32; d];
    for s in 0..p {
        let pos = (rng.next_f64() * (train_ids.len() - w) as f64) as usize;
        spos[s] = pos;
        feat(&train_ids, pos, &mut q);
        for j in 0..d {
            pats[s * d + j] = q[j];
        }
        next[s] = train_ids[pos + w];
    }
    let mut state = vec![0.0f32; d];
    let mut scratch = vec![0.0f32; p];
    let mut qq = vec![0.0f32; d];
    let mut flop_steps: usize = 0;
    // probe betas (10 queries, seed 1000)
    let mut best_beta = betas[0];
    let mut best_acc = -1.0f64;
    let mut best_ppl = f64::INFINITY;
    for &beta in &betas {
        let mut rr = Lcg(1000);
        let (mut acc, mut ce) = (0usize, 0.0f64);
        for _ in 0..10 {
            let pos = (rr.next_f64() * (val_ids.len() - w - 1) as f64) as usize;
            feat(&val_ids, pos, &mut qq);
            flop_steps += run_recall(&qq, &pats, p, d, beta as f32, 12, &mut state, &mut scratch);
            let dist = dist_of_weights(&scratch, &next, v);
            if argmax(&dist) == val_ids[pos + w] {
                acc += 1;
            }
            let mut tot = 0.0;
            for x in &dist {
                tot += x;
            }
            ce += -(((dist[val_ids[pos + w]] + 1e-12) / (tot + 1e-12 * v as f64)).ln());
        }
        let a = acc as f64 / 10.0;
        let pp = (ce / 10.0).exp();
        println!("probe beta={} acc={:.3} ppl={:.2}", beta, a, pp);
        if a > best_acc || (a == best_acc && pp < best_ppl) {
            best_beta = beta;
            best_acc = a;
            best_ppl = pp;
        }
    }
    // main eval (seed 2000)
    let eval_n = |seed: u32, beta: f64, flop: &mut usize, st: &mut [f32], sc: &mut [f32], qq: &mut [f32]| -> (f64, f64) {
        let mut rr = Lcg(seed);
        let (mut acc, mut ce) = (0usize, 0.0f64);
        for _ in 0..tr {
            let pos = (rr.next_f64() * (val_ids.len() - w - 1) as f64) as usize;
            feat(&val_ids, pos, qq);
            *flop += run_recall(qq, &pats, p, d, beta as f32, 12, st, sc);
            let dist = dist_of_weights(sc, &next, v);
            if argmax(&dist) == val_ids[pos + w] {
                acc += 1;
            }
            let mut tot = 0.0;
            for x in &dist {
                tot += x;
            }
            ce += -(((dist[val_ids[pos + w]] + 1e-12) / (tot + 1e-12 * v as f64)).ln());
        }
        (acc as f64 / tr as f64, (ce / tr as f64).exp())
    };
    let bigram_eval = |seed: u32| -> (f64, f64) {
        let mut rr = Lcg(seed);
        let (mut acc, mut ce) = (0usize, 0.0f64);
        for _ in 0..tr {
            let pos = (rr.next_f64() * (val_ids.len() - w - 1) as f64) as usize;
            let (prev, tn) = (val_ids[pos + w - 1], val_ids[pos + w]);
            let row = prev * v;
            let (mut tot, mut best, mut best_c) = (v as f64, 0usize, -1.0f64);
            for c in 0..v {
                let cc = bi[row + c] + 1.0;
                tot += bi[row + c];
                if cc > best_c {
                    best_c = cc;
                    best = c;
                }
            }
            if best == tn {
                acc += 1;
            }
            ce += -(((bi[row + tn] + 1.0) / tot).ln());
        }
        (acc as f64 / tr as f64, (ce / tr as f64).exp())
    };
    let (e_acc, e_ppl) = eval_n(2000, best_beta, &mut flop_steps, &mut state, &mut scratch, &mut qq);
    let (bi_acc, bi_ppl) = bigram_eval(2000);
    let (uni_acc, uni_ppl) = {
        let mut rr = Lcg(2000);
        let (mut acc, mut ce) = (0usize, 0.0f64);
        for _ in 0..tr {
            let pos = (rr.next_f64() * (val_ids.len() - w - 1) as f64) as usize;
            if uni_pred == val_ids[pos + w] {
                acc += 1;
            }
            ce += -((uni[val_ids[pos + w]] + 1e-12).ln());
        }
        (acc as f64 / tr as f64, (ce / tr as f64).exp())
    };
    // interpolation: paired hop+bigram dists per query
    let paired = |seed: u32, flop: &mut usize, st: &mut [f32], sc: &mut [f32], qq: &mut [f32]| -> Vec<(Vec<f64>, Vec<f64>, usize)> {
        let mut rr = Lcg(seed);
        let mut rows = Vec::with_capacity(tr);
        for _ in 0..tr {
            let pos = (rr.next_f64() * (val_ids.len() - w - 1) as f64) as usize;
            feat(&val_ids, pos, qq);
            *flop += run_recall(qq, &pats, p, d, best_beta as f32, 12, st, sc);
            let hd = dist_of_weights(sc, &next, v);
            let off = val_ids[pos + w - 1] * v;
            let mut rtot = v as f64;
            for c in 0..v {
                rtot += bi[off + c];
            }
            let mut bd = vec![0.0f64; v];
            for c in 0..v {
                bd[c] = (bi[off + c] + 1.0) / rtot;
            }
            rows.push((hd, bd, val_ids[pos + w]));
        }
        rows
    };
    let lam_stats = |rows: &[(Vec<f64>, Vec<f64>, usize)]| -> Vec<(f64, f64, f64)> {
        LAMBDAS
            .iter()
            .map(|&lam| {
                let (mut acc, mut ce) = (0usize, 0.0f64);
                for (hd, bd, tn) in rows {
                    let mut bp = -1.0f64;
                    let mut bci = 0usize;
                    for c in 0..v {
                        let pr = lam * hd[c] + (1.0 - lam) * bd[c];
                        if pr > bp {
                            bp = pr;
                            bci = c;
                        }
                    }
                    if bci == *tn {
                        acc += 1;
                    }
                    ce += -((lam * hd[*tn] + (1.0 - lam) * bd[*tn]).max(1e-12).ln());
                }
                (lam, acc as f64 / rows.len() as f64, (ce / rows.len() as f64).exp())
            })
            .collect()
    };
    let sel = lam_stats(&paired(2000, &mut flop_steps, &mut state, &mut scratch, &mut qq));
    // ppl-guarded: minimize select-set ppl first (kNN-LM tunes for ppl), tie-break higher acc.
    let mut best_lam = sel[0];
    for s in &sel {
        if s.2 < best_lam.2 || (s.2 == best_lam.2 && s.1 > best_lam.1) {
            best_lam = *s;
        }
    }
    let rep = lam_stats(&paired(2001, &mut flop_steps, &mut state, &mut scratch, &mut qq));
    let rep_locked = rep[LAMBDAS.iter().position(|&l| l == best_lam.0).unwrap()];
    println!("interp locked lam={} report acc={:.3} ppl={:.2}", best_lam.0, rep_locked.1, rep_locked.2);
    let mut fails = 0;
    // pairing self-check: lam=0 must equal bigram on select queries
    let lam0_ok = (sel[0].1 - bi_acc).abs() < 1e-9 && ((sel[0].2 - bi_ppl).abs() / bi_ppl) < 1e-9;
    println!("{} interp-lam0==bigram-select lam0 {:.3}/{:.2} vs bigram {:.3}/{:.2}", if lam0_ok { "PASS" } else { "FAIL" }, sel[0].1, sel[0].2, bi_acc, bi_ppl);
    if !lam0_ok {
        fails += 1;
    }
    // self-key retrieval (single step on re-featurized stored keys, seed 999 — same as JS)
    {
        let mut rs = Lcg(999);
        let mut got = 0;
        for _ in 0..20 {
            let s = (rs.next_f64() * p as f64) as usize;
            feat(&train_ids, spos[s], &mut qq);
            state.copy_from_slice(&qq);
            recall_step(&pats, &mut state, &mut scratch, p, d, best_beta as f32);
            let mut bi2 = 0;
            for i in 1..p {
                if scratch[i] > scratch[bi2] {
                    bi2 = i;
                }
            }
            let mut same = true;
            for j in 0..d {
                if (qq[j] > 0.0) != (pats[bi2 * d + j] > 0.0) {
                    same = false;
                    break;
                }
            }
            if same {
                got += 1;
            }
        }
        println!("{} self-key-retrieval {}/20", if got == 20 { "PASS" } else { "FAIL" }, got);
        if got != 20 {
            fails += 1;
        }
    }
    let mflops = (flop_steps as f64 * p as f64 * d as f64 * 2.0 / 1e6 * 10.0).round() / 10.0;
    println!("RESULT hop={:.3}/{:.2} bigram={:.3}/{:.2} unigram={:.3}/{:.2} lam={} interp={:.3}/{:.2} mflops={} steps_total={}",
        e_acc, e_ppl, bi_acc, bi_ppl, uni_acc, uni_ppl, best_lam.0, rep_locked.1, rep_locked.2, mflops, flop_steps);
    if fails > 0 {
        fail(&format!("{} failures", fails));
    }
    println!("halt-r-cli lm: done");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || has(&args, "--help") {
        println!("usage: halt-r-cli selftest | lm --data <txt> [--W 8] [--P 8000] [--trials 100] [--proj sim|raw] | lm --synth <chars> [...]");
        return;
    }
    match args[0].as_str() {
        "selftest" => selftest(),
        "lm" => cmd_lm(&args[1..]),
        _ => fail("unknown command (try: selftest, lm)"),
    }
}
