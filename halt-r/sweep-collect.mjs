// sweep-collect: merge per-scale metrics.json into scaling-table.json + scaling.md.
// Usage: node halt-r/sweep-collect.mjs <metrics.json...> --out <table.json> --md <table.md>
import { readFileSync, writeFileSync } from 'node:fs';
const a = process.argv.slice(2);
const oi = a.indexOf('--out'), mi = a.indexOf('--md');
if (oi < 0 || mi < 0) { console.error('FAIL usage: <metrics...> --out <json> --md <md>'); process.exit(1); }
const files = a.filter(x => x !== '--out' && x !== '--md' && x !== a[oi + 1] && x !== a[mi + 1]);
const runs = files.map(f => JSON.parse(readFileSync(f, 'utf8')));
runs.sort((x, y) => x.P - y.P);
const table = { date: new Date().toISOString().slice(0, 10), runs };
writeFileSync(a[oi + 1], JSON.stringify(table, null, 1));
let md = `# HALT-R pretrain sweep (measured, CI)\nDate: ${table.date}. Seeded synthetic (LCG ${runs[0] ? runs[0].seed : '?'}), 10% flips, beta via mini-probe. Models AES-wrapped; hashes below.\n\n| scale | P×D | store | beta·D | acc | steps | build | eval | maxRSS | sha256 |\n|---|---|---|---|---|---|---|---|---|---|\n`;
for (const r of runs) md += `| ${r.scale} | ${r.P}×${r.D} | ${(r.bytes / 1048576).toFixed(1)} MB | ${r.betaD} | ${r.acc.toFixed(3)} | ${r.avgSteps.toFixed(2)} | ${(r.buildMs / 1000).toFixed(0)}s | ${(r.evalMs / 1000).toFixed(0)}s | ${r.maxRssMB} MB | \`${String(r.sha256).slice(0, 16)}…\` |\n`;
writeFileSync(a[mi + 1], md);
console.log(md);
console.log('collect: ' + runs.length + ' scales merged');
