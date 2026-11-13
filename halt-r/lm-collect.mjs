// lm-collect: merge per-config lm metrics.json into lm-bench.json + lm-bench.md.
// Usage: node halt-r/lm-collect.mjs <metrics.json...> --out <json> --md <md>
import { readFileSync, writeFileSync } from 'node:fs';
const a = process.argv.slice(2);
const oi = a.indexOf('--out'), mi = a.indexOf('--md');
if (oi < 0 || mi < 0) { console.error('FAIL usage'); process.exit(1); }
const files = a.filter(x => x !== '--out' && x !== '--md' && x !== a[oi + 1] && x !== a[mi + 1]);
const runs = files.map(f => JSON.parse(readFileSync(f, 'utf8')));
runs.sort((x, y) => (x.corpus < y.corpus ? -1 : x.corpus > y.corpus ? 1 : x.W - y.W || (x.proj < y.proj ? -1 : 1)));
const table = { date: new Date().toISOString().slice(0, 10), task: 'nextchar-heldout', runs };
writeFileSync(a[oi + 1], JSON.stringify(table, null, 1));
let md = `# HALT-R LM bench (measured, CI)\nTask: next-char on held-out 10%. Hopfield retrieval store vs same-query n-gram baselines. Models AES-wrapped. Generation samples in lm-samples/.\n\nkNN-LM interpolation (Khandelwal et al 2019): lam selected on seed-2000 queries, locked lam reported on fresh seed-2001 queries. lam0==bigram is a pairing self-check.\n\n| corpus | W | proj | P | store | beta | hop-acc | hop-ppl | bigram acc/ppl | unigram acc/ppl | lam* | interp-sel acc/ppl | interp-rep acc/ppl | evalMflops | steps | rep4 | wordok | sha |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n`;
for (const r of runs) md += `| ${r.corpus} | ${r.W} | ${r.proj || 'raw'} | ${r.P} | ${(r.bytes / 1048576).toFixed(1)} MB | ${r.beta} | ${r.acc.toFixed(3)} | ${r.ppl.toFixed(2)} | ${r.bigramAcc.toFixed(3)}/${r.bigramPpl.toFixed(2)} | ${r.unigramAcc.toFixed(3)}/${r.unigramPpl.toFixed(2)} | ${typeof r.interpLambda === 'number' ? r.interpLambda : '?'} | ${typeof r.interpAccSel === 'number' ? r.interpAccSel.toFixed(3) + '/' + r.interpPplSel.toFixed(1) : '?'} | ${typeof r.interpAcc === 'number' ? r.interpAcc.toFixed(3) + '/' + r.interpPpl.toFixed(1) : '?'} | ${typeof r.evalMflops === 'number' ? r.evalMflops : '?'} | ${r.avgSteps.toFixed(2)} | ${r.quality ? r.quality.rep4.toFixed(2) : '?'} | ${r.quality ? r.quality.wordValid.toFixed(2) : '?'} | \`${String(r.sha256).slice(0, 12)}\` |\n`;
writeFileSync(a[mi + 1], md);
console.log(md);
console.log('collect: ' + runs.length + ' configs merged');
