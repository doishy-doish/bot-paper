// lm-collect: merge per-config lm metrics.json into lm-bench.json + lm-bench.md.
// Usage: node halt-r/lm-collect.mjs <metrics.json...> --out <json> --md <md>
import { readFileSync, writeFileSync } from 'node:fs';
const a = process.argv.slice(2);
const oi = a.indexOf('--out'), mi = a.indexOf('--md');
if (oi < 0 || mi < 0) { console.error('FAIL usage'); process.exit(1); }
const files = a.filter(x => x !== '--out' && x !== '--md' && x !== a[oi + 1] && x !== a[mi + 1]);
const runs = files.map(f => JSON.parse(readFileSync(f, 'utf8')));
runs.sort((x, y) => (x.corpus < y.corpus ? -1 : x.corpus > y.corpus ? 1 : x.W - y.W));
const table = { date: new Date().toISOString().slice(0, 10), task: 'nextchar-heldout', runs };
writeFileSync(a[oi + 1], JSON.stringify(table, null, 1));
let md = `# HALT-R LM bench (measured, CI)\nTask: next-char on held-out 10%. Hopfield retrieval store vs same-query n-gram baselines. Models AES-wrapped.\n\n| corpus | W | P | store | beta | hop-acc | hop-ppl | bigram acc/ppl | unigram acc/ppl | steps | sha |\n|---|---|---|---|---|---|---|---|---|---|---|\n`;
for (const r of runs) md += `| ${r.corpus} | ${r.W} | ${r.P} | ${(r.bytes / 1048576).toFixed(1)} MB | ${r.beta} | ${r.acc.toFixed(3)} | ${r.ppl.toFixed(2)} | ${r.bigramAcc.toFixed(3)}/${r.bigramPpl.toFixed(2)} | ${r.unigramAcc.toFixed(3)}/${r.unigramPpl.toFixed(2)} | ${r.avgSteps.toFixed(2)} | \`${String(r.sha256).slice(0, 12)}\` |\n`;
writeFileSync(a[mi + 1], md);
console.log(md);
console.log('collect: ' + runs.length + ' configs merged');
