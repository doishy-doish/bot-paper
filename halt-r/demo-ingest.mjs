// demo-ingest.mjs: pull a finished small model, check it, ingest it as a frozen
// voter, save the NEW post-ingestion model. Zero training anywhere in this file.
//
// What it does (ts default, ~580KB download, seconds):
//   1. pull  tinygpt-{model}.gguf + vocab + ref + q100 voter streams (public raw,
//           or --dir cache)
//   2. check GGUF header (magic/version/17 tensors/arch KV/sha/size), ref argmax
//           consistency, q100 replay acc/ppl vs pre-reg bands, cover 100/100
//   3. save  demo-ingest-model.json = the post-ingestion artifact. Boot loads
//           THIS file (verify donor sha, route) instead of rebuilding from scratch.
// The old (donor) weights are archived in the PRIVATE repo (donors/), never in
// the workspace: this file + the model JSON are the only demo files that ship.
//
// Usage: node halt-r/demo-ingest.mjs --model ts --pull --out out/demo
//        node halt-r/demo-ingest.mjs --model both --pull --out out/demo
// Exit non-zero on any failed check (honest demo: red means red).

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import https from 'node:https';

const BASE_DEFAULT = 'https://raw.githubusercontent.com/doishy-doish/bot-paper/main/halt-r';

// Pre-reg bands (locked 2026-11-13, from committed q100 + ingest.json):
// replay must land inside band AND cover must be 100/100, or the demo FAILS.
const EXPECT = {
  ts: {
    V: 79, E: 64, H: 2, L: 2, block: 32, rev: 2,
    bytes: 436064, sha256: '26402e2d9eac0c7a2d26c8e876e7c17c3c2be52b5a13fcacad2e7327df2fba70',
    s2000: { acc: 0.57, ppl: 4.36 }, s2001: { acc: 0.65, ppl: 3.30 },
    pplTol: 0.15, accTol: 0.02,
  },
  shakes: {
    V: 65, E: 64, H: 2, L: 2, block: 32, rev: 2,
    bytes: 428768, sha256: 'fcd797be68c5d0948f3a836fc689773d7a6cf921da79b53f88b0dec376201af9',
    s2000: { acc: 0.49, ppl: 6.12 }, s2001: { acc: 0.46, ppl: 5.31 },
    pplTol: 0.15, accTol: 0.02,
  },
};
const ORDER = ['tok.weight'];
for (let l = 0; l < 2; l++) {
  const p = 'blk.' + l + '.';
  for (const n of ['ln1.w', 'attn.qkv_w', 'attn.out_w', 'ln2.w', 'mlp.gate_w', 'mlp.up_w', 'mlp.down_w']) ORDER.push(p + n);
}
ORDER.push('ln_f.w', 'head.weight'); // 17 total

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name);
  if (i < 0) return dflt;
  if (i + 1 >= process.argv.length || process.argv[i + 1].startsWith('--')) return dflt;
  return process.argv[i + 1];
}
function flag(name) { return process.argv.includes('--' + name); }
function fail(msg) { console.error('DEMO FAIL: ' + msg); process.exitCode = 1; throw new Error(msg); }

function get(url) {
  return new Promise((resolve, reject) => {
    https.get(url, { headers: { 'User-Agent': 'halt-r-demo' } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        resolve(get(res.headers.location));
        return;
      }
      if (res.statusCode !== 200) { reject(new Error('HTTP ' + res.statusCode + ' ' + url)); return; }
      const parts = [];
      res.on('data', (c) => parts.push(c));
      res.on('end', () => resolve(Buffer.concat(parts)));
      res.on('error', reject);
    }).on('error', reject);
  });
}

function parseGguf(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const dec = new TextDecoder();
  let o = 0;
  const u32 = () => { const v = dv.getUint32(o, true); o += 4; return v; };
  const u64 = () => { const v = Number(dv.getBigUint64(o, true)); o += 8; return v; };
  const rstr = () => { const n = u64(); const s = dec.decode(buf.subarray(o, o + n)); o += n; return s; };
  const magic = u32();
  const ver = u32();
  const nt = u64(), nk = u64();
  const kv = {};
  const rval = (t) => {
    if (t === 4) return u32();
    if (t === 8) return rstr();
    if (t === 9) {
      const et = u32(), n = u64(), arr = [];
      for (let i = 0; i < n; i++) arr.push(rval(et));
      return { et, arr };
    }
    fail('gguf: unsupported kv type ' + t);
  };
  for (let i = 0; i < nk; i++) { const k = rstr(), t = u32(); kv[k] = rval(t); }
  const tensors = [];
  for (let i = 0; i < nt; i++) {
    const name = rstr(), nd = u32(), dims = [];
    for (let j = 0; j < nd; j++) dims.push(u64());
    const tt = u32(), off = u64();
    tensors.push({ name, dims, tt, off });
  }
  return { magic, ver, nt, nk, kv, tensors };
}

function replay(queries, V) {
  let acc = 0, ce = 0, cov = 0;
  for (const q of queries) {
    if (!q.probs || q.probs.length !== V) continue;
    cov++;
    let am = 0;
    for (let i = 1; i < q.probs.length; i++) if (q.probs[i] > q.probs[am]) am = i;
    if (am === q.tn) acc++;
    ce += -Math.log(Math.max(q.probs[q.tn], 1e-12));
  }
  return { n: queries.length, cov, acc: acc / cov, ppl: Math.exp(ce / cov) };
}

async function runModel(model, base, dir, outDir) {
  const E = EXPECT[model];
  const files = {
    gguf: 'tinygpt-' + model + '.gguf',
    vocab: 'vocab-' + model + '.json',
    ref: 'ref-' + model + '.json',
    q100: 'q100-gpt-' + model + '.json',
  };
  const local = {};
  for (const [k, f] of Object.entries(files)) {
    const p = path.join(dir, f);
    if (fs.existsSync(p)) { local[k] = fs.readFileSync(p); console.log('demo: cache hit ' + f); }
    else {
      console.log('demo: pull ' + base + '/' + f);
      local[k] = await get(base + '/' + f);
      fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(p, local[k]);
    }
  }
  // --- check 1: GGUF header ---
  const g = parseGguf(local.gguf);
  if (g.magic !== 0x46554747) fail(model + ': gguf magic');
  if (g.ver !== 3) fail(model + ': gguf version ' + g.ver);
  if (g.nt !== 17) fail(model + ': tensor count ' + g.nt);
  if (g.nk !== 15) fail(model + ': kv count ' + g.nk);
  g.tensors.forEach((t, i) => { if (t.name !== ORDER[i]) fail(model + ': tensor[' + i + ']=' + t.name); });
  if (g.kv['general.architecture'] !== 'halt-tinygpt') fail(model + ': arch');
  if (g.kv['halt.vocab_size'] !== E.V) fail(model + ': V');
  if (g.kv['halt.arch_rev'] !== E.rev) fail(model + ': rev');
  if (local.gguf.length !== E.bytes) fail(model + ': bytes ' + local.gguf.length);
  const sha = crypto.createHash('sha256').update(local.gguf).digest('hex');
  if (sha !== E.sha256) fail(model + ': sha256 ' + sha);
  let params = 0;
  for (const t of g.tensors) params *= 1, params += t.dims.reduce((a, b) => a * b, 1);
  console.log('demo: ' + model + ' gguf OK bytes=' + local.gguf.length + ' params=' + params + ' sha=' + sha.slice(0, 12) + '...');
  // --- check 2: ref argmax consistency ---
  const ref = JSON.parse(local.ref.toString('utf8'));
  const vocab = JSON.parse(local.vocab.toString('utf8'));
  if (vocab.V !== E.V || vocab.block !== E.block) fail(model + ': vocab mismatch');
  for (const r of ref.refs) {
    if (r.logits.length !== E.V) fail(model + ': ref logits len');
    let am = 0;
    for (let i = 1; i < r.logits.length; i++) if (r.logits[i] > r.logits[am]) am = i;
    if (am !== r.argmax) fail(model + ': ref argmax');
    if (r.ids.length !== E.block) fail(model + ': ref ids len');
  }
  console.log('demo: ' + model + ' ref OK refs=' + ref.refs.length);
  // --- check 3: frozen-voter replay vs pre-reg bands ---
  const q = JSON.parse(local.q100.toString('utf8'));
  if (q.V !== E.V) fail(model + ': q100 V');
  const out = { model, V: E.V };
  for (const s of ['2000', '2001']) {
    const r = replay(q.streams[s].queries, E.V);
    const band = s === '2000' ? E.s2000 : E.s2001;
    if (r.cov !== 100 || r.n !== 100) fail(model + ': cover ' + r.cov + '/' + r.n);
    if (Math.abs(r.acc - band.acc) > E.accTol) fail(model + ': stream' + s + ' acc ' + r.acc.toFixed(3));
    if (Math.abs(r.ppl - band.ppl) > E.pplTol) fail(model + ': stream' + s + ' ppl ' + r.ppl.toFixed(2));
    console.log('demo: ' + model + ' stream' + s + ' acc=' + r.acc.toFixed(3) + ' ppl=' + r.ppl.toFixed(2) + ' cover=' + r.cov + '/100 IN BAND');
    out['s' + s] = { acc: +r.acc.toFixed(3), ppl: +r.ppl.toFixed(2), cover: r.cov };
  }
  return {
    donor: {
      gguf: files.gguf, bytes: local.gguf.length, sha256: sha,
      arch: 'halt-tinygpt', params, V: E.V, n_embd: E.E, n_head: E.H, n_layer: E.L, block: E.block, rev: E.rev,
      vocab: files.vocab, ref: files.ref, refN: ref.refs.length,
    },
    voter: { q100: files.q100, streams: out },
  };
}

async function main() {
  const modelArg = arg('model', 'ts');
  const models = modelArg === 'both' ? ['ts', 'shakes'] : [modelArg];
  if (!models.every((m) => EXPECT[m])) fail('usage: --model ts|shakes|both');
  const base = arg('base', BASE_DEFAULT);
  const dir = arg('dir', 'demo-cache');
  const outDir = arg('out', 'out/demo');
  fs.mkdirSync(outDir, { recursive: true });
  const pulled = [];
  for (const m of models) pulled.push(await runModel(m, base, dir, outDir));
  // --- save: the NEW post-ingestion model (boot loads this, no rebuild) ---
  const post = {
    version: 'demo-ingest/1',
    date: new Date().toISOString().slice(0, 10),
    task: 'ingest-demo',
    preReg: 'frozen voters ingested with zero training (store+bigram+tinyGPT); cover 100/100; replay in band; donor sha pinned',
    donors: pulled,
    router: {
      recipe: 'mix gpt over W32 join; Hedge online over LAMBDAS frozen on select stream, locked report',
      expected: { ts: { gptMixAcc: 0.65, gptMixPpl: 3.30 }, shakes: { gptMixAcc: 0.46, gptMixPpl: 5.31 } },
    },
    surgery: { ref: 'halt-r/c2.json', remember: '1.00/0.95', editFlip: '0.90-1.00', forgetLeak: 0.0, collateral: 0.0 },
    boot: 'load this file, verify donor sha256, route frozen voters. No rebuild, no training.',
    verdict: 'DEMO GREEN: finished small model(s) pulled, checked, ingested as frozen voter(s); post-ingestion artifact saved; zero gradients.',
  };
  fs.writeFileSync(path.join(outDir, 'demo-ingest-model.json'), JSON.stringify(post, null, 1));
  const md = '# Ingest demo (measured)\n\n' + models.map((m, i) => {
    const d = pulled[i];
    return '## ' + m + '\nDonor ' + d.donor.gguf + ': ' + d.donor.bytes + ' bytes, sha ' + d.donor.sha256.slice(0, 12) + '..., ' + d.donor.params + ' params, GGUFv3 halt-tinygpt rev 2, 17 tensors byte-exact layout. Ref ' + d.donor.refN + '/5 argmax-consistent. Voter streams: s2000 acc=' + d.voter.streams.s2000.acc + ' ppl=' + d.voter.streams.s2000.ppl + ', s2001 acc=' + d.voter.streams.s2001.acc + ' ppl=' + d.voter.streams.s2001.ppl + ', cover 100/100 both. All in pre-reg band.\n';
  }).join('\n') + '\nPost-ingestion artifact: demo-ingest-model.json (donor pins + voter replay + router recipe). Boot loads it; nothing rebuilds. Zero training.\n';
  fs.writeFileSync(path.join(outDir, 'demo-ingest.md'), md);
  process.stdout.write(md);
}
main().catch((e) => { if (!process.exitCode) process.exitCode = 1; console.error('DEMO ERROR: ' + (e && e.message)); });
