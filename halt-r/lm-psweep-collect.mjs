// lm-psweep-collect: merge P-sweep metrics into psweep.json + psweep.md + SVG curves.
// Usage: node halt-r/lm-psweep-collect.mjs <metrics.json...> --out <json> --md <md> --svg <prefix>
// Emits <prefix>-<corpus>-ppl.svg and <prefix>-<corpus>-acc.svg: log-P x-axis, per-W
// interp-rep lines, bigram dashed anchors. Plus auto verdict lines (wall moves or holds).
import { readFileSync, writeFileSync } from 'node:fs';
const a = process.argv.slice(2);
const gv = (k) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : null; };
const outP = gv('--out'), mdP = gv('--md'), svgPre = gv('--svg');
if (!outP || !mdP || !svgPre) { console.error('FAIL usage: <metrics...> --out <json> --md <md> --svg <prefix>'); process.exit(1); }
const files = a.filter(x => !['--out', '--md', '--svg', outP, mdP, svgPre].includes(x));
if (!files.length) { console.error('FAIL no metrics files'); process.exit(1); }
const runs = files.map(f => JSON.parse(readFileSync(f, 'utf8')));
runs.sort((x, y) => (x.corpus < y.corpus ? -1 : x.corpus > y.corpus ? 1 : x.W - y.W || x.P - y.P));
writeFileSync(outP, JSON.stringify({ date: new Date().toISOString().slice(0, 10), task: 'psweep-P-vs-ppl', runs }, null, 1));

const COLORS = { 2: '#4e79a7', 4: '#f28e2b', 8: '#e15759', 16: '#76b7b2', 32: '#59a14f' };
const corpora = [...new Set(runs.map(r => r.corpus))].sort();

// ---- markdown: P-vs-ppl table + auto verdicts ----
let md = `# HALT-R P-sweep (measured, CI)\nTask: next-char held-out, sim projection (D=H=256 fixed, so recall cost is exactly linear in P). Bigram anchor is P-independent (trained on full train set) — the delta is the retrieval win.\n\n| corpus | W | P | store | beta | hop-ppl | bigram-ppl | lam\\* | interp-rep acc/ppl | evalMflops | steps | sha |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n`;
for (const r of runs) md += `| ${r.corpus} | ${r.W} | ${r.P} | ${(r.bytes / 1048576).toFixed(1)} MB | ${r.beta} | ${r.ppl.toFixed(2)} | ${r.bigramPpl.toFixed(2)} | ${typeof r.interpLambda === 'number' ? r.interpLambda : '?'} | ${typeof r.interpAcc === 'number' ? r.interpAcc.toFixed(3) + '/' + r.interpPpl.toFixed(1) : '?'} | ${typeof r.evalMflops === 'number' ? r.evalMflops : '?'} | ${r.avgSteps.toFixed(2)} | \`${String(r.sha256).slice(0, 12)}\` |\n`;
md += `\n## Verdicts (auto, from locked-lam fresh-query numbers)\n`;
for (const c of corpora) {
  for (const W of [...new Set(runs.filter(r => r.corpus === c).map(r => r.W))].sort((x, y) => x - y)) {
    const rows = runs.filter(r => r.corpus === c && r.W === W);
    const first = rows[0], last = rows[rows.length - 1];
    const bi = first.bigramPpl;
    if (typeof last.interpPpl !== 'number' || typeof first.interpPpl !== 'number') { md += `- ${c} W${W}: interp numbers missing\n`; continue; }
    const moves = last.interpPpl < first.interpPpl * 0.95;
    const beats = last.interpPpl < bi;
    md += `- ${c} W${W}: interp-ppl ${first.interpPpl.toFixed(1)} (P=${first.P}) -> ${last.interpPpl.toFixed(1)} (P=${last.P}) vs bigram ${bi.toFixed(2)} — wall ${moves ? '**MOVES**' : 'holds'}${beats ? ', **beats bigram**' : ''}\n`;
  }
}
writeFileSync(mdP, md);
console.log(md);

// ---- SVG curves ----
function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }
function curveSVG(corpus, metric) {
  // metric: 'ppl' (log y, interpPpl + bigramPpl anchor) or 'acc' (linear y, interpAcc + bigramAcc anchor)
  const rows = runs.filter(r => r.corpus === corpus);
  const Ws = [...new Set(rows.map(r => r.W))].sort((x, y) => x - y);
  const Ps = [...new Set(rows.map(r => r.P))].sort((x, y) => x - y);
  const isPpl = metric === 'ppl';
  const valOf = (r) => isPpl ? r.interpPpl : r.interpAcc;
  const ancOf = (r) => isPpl ? r.bigramPpl : r.bigramAcc;
  const Wpx = 680, Hpx = 440, ML = 64, MR = 14, MT = 44, MB = 60;
  const PW = Wpx - ML - MR, PH = Hpx - MT - MB;
  const lx0 = Math.log10(Math.min(...Ps)), lx1 = Math.log10(Math.max(...Ps));
  let vals = [];
  for (const r of rows) { if (typeof valOf(r) === 'number' && isFinite(valOf(r))) vals.push(valOf(r)); if (typeof ancOf(r) === 'number') vals.push(ancOf(r)); }
  let yMin, yMax, yTicks;
  if (isPpl) {
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const cand = [2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 5000, 20000, 100000, 1e7];
    yTicks = cand.filter(t => t >= lo * 0.92 && t <= hi * 1.08);
    yMin = Math.min(...yTicks) * 0.94; yMax = Math.max(...yTicks) * 1.06;
  } else {
    yMin = 0; yMax = Math.ceil(Math.max(...vals) * 10) / 10 || 0.6;
    yTicks = []; for (let t = 0; t <= yMax + 1e-9; t += 0.1) yTicks.push(Math.round(t * 10) / 10);
  }
  const X = (p) => ML + (Math.log10(p) - lx0) / ((lx1 - lx0) || 1) * PW;
  const Y = (v) => isPpl
    ? MT + (1 - (Math.log10(Math.min(Math.max(v, yMin), yMax)) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * PH
    : MT + (1 - (v - yMin) / ((yMax - yMin) || 1)) * PH;
  let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${Wpx}" height="${Hpx}" font-family="monospace">`;
  s += `<rect width="${Wpx}" height="${Hpx}" fill="#ffffff"/>`;
  s += `<text x="${Wpx / 2}" y="22" text-anchor="middle" font-size="15" font-weight="bold">${esc(corpus)} — interp-rep ${isPpl ? 'perplexity (log)' : 'accuracy'} vs store size P (log)</text>`;
  // grid + y ticks
  for (const t of yTicks) {
    const y = Y(t);
    s += `<line x1="${ML}" y1="${y.toFixed(1)}" x2="${Wpx - MR}" y2="${y.toFixed(1)}" stroke="#e3e3e3"/>`;
    s += `<text x="${ML - 6}" y="${(y + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="#333">${t}</text>`;
  }
  for (const p of Ps) {
    const x = X(p);
    s += `<line x1="${x.toFixed(1)}" y1="${MT}" x2="${x.toFixed(1)}" y2="${Hpx - MB}" stroke="#e3e3e3"/>`;
    const lbl = p >= 1000 ? (p / 1000) + 'K' : String(p);
    s += `<text x="${x.toFixed(1)}" y="${Hpx - MB + 20}" text-anchor="middle" font-size="11" fill="#333">P=${lbl}</text>`;
  }
  s += `<text x="${ML - 46}" y="${MT + PH / 2}" text-anchor="middle" font-size="12" fill="#333" transform="rotate(-90 ${ML - 46} ${MT + PH / 2})">${isPpl ? 'ppl (lower better)' : 'acc (higher better)'}</text>`;
  // bigram dashed anchors (mean across P — bigram is P-independent by construction)
  for (const W of Ws) {
    const wr = rows.filter(r => r.W === W && typeof ancOf(r) === 'number');
    if (!wr.length) continue;
    const mean = wr.reduce((t, r) => t + ancOf(r), 0) / wr.length;
    const spread = Math.max(...wr.map(r => ancOf(r))) - Math.min(...wr.map(r => ancOf(r)));
    const y = Y(mean);
    s += `<line x1="${ML}" y1="${y.toFixed(1)}" x2="${Wpx - MR}" y2="${y.toFixed(1)}" stroke="${COLORS[W] || '#999'}" stroke-dasharray="6,4" stroke-width="1.5" opacity="0.7"/>`;
    if (spread / (mean || 1) > 1e-9) s += `<text x="${Wpx - MR}" y="${(y - 5).toFixed(1)}" text-anchor="end" font-size="10" fill="#a00">bigram W${W} VARIES ${(spread).toFixed(3)} — BUG?</text>`;
  }
  // per-W interp lines
  for (const W of Ws) {
    const pts = rows.filter(r => r.W === W && typeof valOf(r) === 'number' && isFinite(valOf(r)))
      .map(r => [X(r.P), Y(valOf(r)), r.P, valOf(r)]);
    if (!pts.length) continue;
    const col = COLORS[W] || '#999';
    if (pts.length > 1) s += `<polyline points="${pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ')}" fill="none" stroke="${col}" stroke-width="2.5"/>`;
    for (const [x, y, p, v] of pts) s += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4" fill="${col}"><title>W${W} P=${p} ${isPpl ? 'ppl' : 'acc'}=${typeof v === 'number' ? v.toFixed(isPpl ? 1 : 3) : v}</title></circle>`;
  }
  // legend
  let lx = ML + 8;
  s += `<text x="${lx}" y="${MT - 12}" font-size="11" fill="#333">solid=interp-rep</text>`;
  lx += 150;
  for (const W of Ws) {
    const col = COLORS[W] || '#999';
    s += `<rect x="${lx}" y="${MT - 22}" width="12" height="12" fill="${col}"/><text x="${lx + 15}" y="${MT - 12}" font-size="11" fill="#333">W${W}</text>`;
    lx += 52;
  }
  s += `<text x="${lx}" y="${MT - 12}" font-size="11" fill="#333">dashed=bigram anchor (P-independent)</text>`;
  s += `<text x="${ML}" y="${Hpx - 8}" font-size="10" fill="#666">HALT-R P-sweep, sim proj D=H=256, locked-lam fresh queries, AES-wrapped models. Cost per query ~ steps x P x D x 2 FLOP (linear in P).</text>`;
  s += `</svg>`;
  return s;
}
for (const c of corpora) {
  writeFileSync(`${svgPre}-${c}-ppl.svg`, curveSVG(c, 'ppl'));
  writeFileSync(`${svgPre}-${c}-acc.svg`, curveSVG(c, 'acc'));
  console.log(`svg: ${svgPre}-${c}-ppl.svg + ${svgPre}-${c}-acc.svg`);
}
console.log('psweep-collect: ' + runs.length + ' configs merged');
