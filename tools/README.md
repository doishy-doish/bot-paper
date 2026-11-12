# confidence-audit

Zero-GPU trace audit from **"Calibration Is Real, Verification Is Not Enough"**
(Gemma 4 Developer Agent, Paper Track). Given banked run tables plus the raw
agent trace JSON files, it re-derives every per-trace verdict with no GPU and
no model calls — and reproduces the paper's headline counts exactly.

## Run it (10 seconds, zero GPU)

```bash
python3 tools/confidence-audit.py \
    --pool tools/data/fire-w1-table.json:tools/traces/w1 \
    --pool tools/data/fire-w2a-table.json:traces/w2a \
    --pool tools/data/fire-fam-table.json:traces/fam \
    --out /tmp/audit-results.json
```

Expected output on the banked data:

```
traces=44 GREEN=10 SOURCE_ONLY=9 pytest_traces=3 (requests_7315,rich_3067,requests_7309)
exit_histogram={"-1": 1, "0": 17, "1": 22, "2": 4}
empty_submit=rich_3472,rich_3506
... (one E/S/C/F/T/D gate line per trace; see script docstring for gates)
wrote /tmp/audit-results.json
```

That is 10 surgical greens, 9 source-only (the docs-only `fastapi_14492` solve
is cut by gate 6), pytest invoked in exactly 3 traces, four `exit-2` failures,
and two empty submits — the paper's §3.4, verified by re-running, not by trust.

## What it checks (per trace)

1. `exit_ok` — `exit_code == 0` and `resolved` true (from the banked table)
2. `size_ok` — `200 <= patch_chars <= 1200`
3. `conf_ok` — a `CONFIDENCE <float> BECAUSE <evidence>` line with
   non-trivial evidence (presence check — the paper's finding is that agents
   quote their own reproducer output, so the evidence text is printed for you)
4. `files1` — exactly 1 changed file (`files_changed == 1`), from structured
   `edit_file`/`write_file` calls plus edit-op text mining; reads never count;
   agent scratch (`repro.py`, `verify_*`, shell heredoc targets…) never counts
5. `no_test` — no repo-test touches (edit/write/execute; agent-created scratch
   scripts excluded — e.g. a heredoc'd `test_qualname.py` the agent runs and
   `rm`s is scratch, while overwriting the repo's `test_annotation.py` is a
   touch)
6. `no_docs` — no docs/config touches

`GREEN` = gates 1–5. `SOURCE_ONLY` = all six (the paper's training pool).

## Layout

- `tools/confidence-audit.py` — the audit (stdlib only, no dependencies)
- `tools/data/fire-{w1,w2a,fam}-table.json` — banked run tables (the paper's
  Tables 3.1/3.3 inputs: id, patch_chars, exit_code, resolved, tool_calls)
- `tools/traces/{w1,w2a,fam}/trace_<id>.json` — all 44 raw traces
