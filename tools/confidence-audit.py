#!/usr/bin/env python3
"""confidence-audit: zero-GPU trace audit for coding-agent runs.

Reproduces the surgical-green filter from "Calibration Is Real,
Verification Is Not Enough" (Gemma 4 Developer Agent Paper Track):
given banked run tables (id, patch_chars, exit_code, resolved) plus the
raw agent trace JSON files, it re-derives every per-trace verdict with
no GPU and no model calls.

The six gates (a trace is GREEN only if 1-5 pass; SOURCE-ONLY if 6 too):
  1. exit_ok - exit_code == 0 and resolved is true
  2. size_ok - 200 <= patch_chars <= 1200
  3. conf_ok - a CONFIDENCE <float> BECAUSE <evidence> line exists with
     non-trivial evidence (>= 8 chars). Presence check by design: the
     paper's finding is that agents quote their OWN reproducer output,
     so the gate keeps traces whose confidence cites *something* and the
     tool prints the evidence for human review.
  4. files1     - exactly 1 changed file (files_changed == 1: every
     edited file counts, test/docs included, scratch excluded — gates
     5/6 then carve test/docs touches out), where EDITED comes from
     structured tool calls (edit_file args, write_file args/results).
     READS (start_line/end_line args) never count. Scratch the agent
     created in-trace (repro.py, repro_*, verify_*, /tmp*, *.orig, shell
     redirect/heredoc targets) never counts.
  5. no_test - zero touches to repo test files. A touch is an edit/write
     op OR executing a test file in a shell command. Scratch scripts the
     agent created itself (e.g. heredoc'd test_qualname.py, run and rm'd
     in-trace) are NOT repo touches. Test paths: test_*.py, *_test.py,
     tests//test//testing/ dirs.
  6. no_docs - zero touches to docs/config files (docs_src/, docs/,
     *.md/.rst/.txt/.yaml/.yml/.toml/.cfg/.ini) via edit/write/command.

It also reports: pytest invocations as real shell commands (an `import
pytest` inside file contents does NOT count), the exit-code histogram,
and empty submits (patch_chars == 0).

Usage:
  python3 tools/confidence-audit.py \
      --pool tables/fire-w1-table.json:traces/w1 \
      --pool tables/fire-w2a-table.json:traces/w2a \
      --pool tables/fire-fam-table.json:traces/fam \
      [--out audit-results.json]

Expected output on the paper's banked data:
  44 traces, 10 GREEN, 9 SOURCE_ONLY, pytest in 3 traces
  (requests_7315, rich_3067, requests_7309).
"""

import argparse
import json
import re
import sys
from pathlib import Path

CONF_RE = re.compile(r"CONFIDENCE\s+[0-9.]+\s+BECAUSE\s+(.+)")
TEST_RE = re.compile(r"(^|/)(test_.*|.*_test\.py|tests/|test/|testing/)", re.I)
DOC_RE = re.compile(
    r"(^|/)(docs?_src/.*|docs?/.*|.*\.(md|rst|txt|yaml|yml|toml|cfg|ini))$", re.I
)
SCRATCH_RE = re.compile(r"(^|/)(repro\.py|repro_.*|verify_.*|.*\.orig)$")
TMP_RE = re.compile(r"^/tmp")
REDIRECT_RE = re.compile(r">{1,2}\s*(\S+\.(?:py|md|rst|txt))")
PY_EXT_RE = re.compile(r"\.(py|md|rst|txt|yaml|yml|toml|cfg|ini)$")
PYTEST_CMD_RE = re.compile(r'"command"\s*:\s*"[^"]*\bpytest\b')


def _calls(trace):
    """Yield (function_name, arguments, observation) for every tool call."""
    steps = trace.get("steps") or []
    for s in steps if isinstance(steps, list) else []:
        for c in s.get("tool_calls") or []:
            yield (c.get("function_name") or "",
                   c.get("arguments") or {},
                   c.get("observation") or {})


def _obs_file(obs):
    """filepath from a tool observation (dict or JSON string)."""
    if isinstance(obs, dict):
        if isinstance(obs.get("filepath"), str):
            return obs["filepath"]
        content = obs.get("content")
    else:
        content = obs
    if isinstance(content, str):
        m = re.search(r'"filepath"\s*:\s*"([^"]+)"', content)
        if m:
            return m.group(1)
    return None


def is_test(path):
    return bool(TEST_RE.search(path))


def is_docs(path):
    return bool(DOC_RE.search(path))


def is_scratch(path):
    return bool(SCRATCH_RE.search(path) or TMP_RE.match(path))


def analyze(text, trace):
    edits, writes, ran, created = set(), set(), set(), set()
    for fname, args, obs in _calls(trace):
        fp = args.get("filepath")
        if fname == "edit_file" and isinstance(fp, str):
            edits.add(fp)
        if fname == "write_file":
            if isinstance(fp, str):
                writes.add(fp)
            of = _obs_file(obs)
            if of:
                writes.add(of)
        if fname == "run_command" and isinstance(args.get("command"), str):
            cmd = args["command"]
            for tok in re.split(r"\s+", cmd):
                tok = tok.strip("\"'").rstrip(";,")
                if ((PY_EXT_RE.search(tok) and "/" in tok)
                        or tok.endswith(".py")):
                    ran.add(tok)
            for m in REDIRECT_RE.finditer(cmd):
                created.add(m.group(1))
    # text-mined edit ops complement structured calls (reads never
    # contain new_string, so unioning is safe)
    for m in re.finditer(
            r'"filepath"\s*:\s*"([^"]+)"[^}]{0,400}?"new_string"', text):
        edits.add(m.group(1))
    for m in re.finditer(
            r'"new_string"[^}]{0,2000}?"filepath"\s*:\s*"([^"]+)"', text):
        edits.add(m.group(1))
    made = set(created)  # files the agent created via shell redirect/heredoc
    scratchy = lambda f: is_scratch(f) or f in made
    # files_changed: every edited file counts here (test/docs included;
    # scratch excluded) — gates 5/6 then carve out test/docs touches.
    changed = {f for f in (edits | writes) if not scratchy(f)}
    src_edits = sorted(changed)
    # repo-test touch: edit/write of a test path, or executing one the
    # agent did not itself create in-trace (scratch scripts the agent
    # heredoc'd, ran and rm'd are NOT repo touches)
    test_touch = any(is_test(f) for f in (edits | writes | ran)
                     if not scratchy(f))
    docs_touch = any(is_docs(f) for f in (edits | writes | ran)
                     if not scratchy(f))
    conf_m = CONF_RE.search(text)
    conf_ev = (conf_m.group(1).strip().strip('"') if conf_m else "")[:200]
    return {
        "edited_source": sorted(src_edits),
        "test_touch": test_touch,
        "docs_touch": docs_touch,
        "conf_evidence": conf_ev[:120],
        "conf_ok": conf_m is not None and len(conf_ev.strip()) >= 8,
        "pytest_invocations": len(PYTEST_CMD_RE.findall(text)),
    }


def audit_row(row, trace_path):
    try:
        text = Path(trace_path).read_text()
        trace = json.loads(text)
    except (OSError, ValueError):
        return {"id": row.get("id"), "error": "missing-or-bad-trace"}
    a = analyze(text, trace)
    g_exit = row.get("exit_code") == 0 and bool(row.get("resolved"))
    g_size = 200 <= int(row.get("patch_chars", -1)) <= 1200
    g_conf = a["conf_ok"]
    g_files1 = len(a["edited_source"]) == 1
    g_notest = not a["test_touch"]
    g_nodocs = not a["docs_touch"]
    green = g_exit and g_size and g_conf and g_files1 and g_notest
    return {
        "id": row.get("id"),
        "patch": row.get("patch_chars"),
        "exit": row.get("exit_code"),
        "calls": row.get("tool_calls"),
        "edited": a["edited_source"],
        "conf_evidence": a["conf_evidence"],
        "pytest_invocations": a["pytest_invocations"],
        "gates": {"exit_ok": g_exit, "size_ok": g_size, "conf_ok": g_conf,
                  "files1": g_files1, "no_test": g_notest,
                  "no_docs": g_nodocs},
        "GREEN": green,
        "SOURCE_ONLY": green and g_nodocs,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Zero-GPU confidence-theater audit.")
    ap.add_argument("--pool", action="append", required=True,
                    help="TABLE.json:TRACES_DIR (repeatable)")
    ap.add_argument("--out", default=None, help="write per-trace JSON here")
    args = ap.parse_args(argv)
    verdicts = []
    for pool in args.pool:
        table_s, _, traces_s = pool.partition(":")
        table = json.loads(Path(table_s).read_text())
        for row in table if isinstance(table, list) else []:
            verdicts.append(audit_row(
                row, Path(traces_s) / ("trace_%s.json" % row.get("id"))))
    n = len(verdicts)
    greens = [v["id"] for v in verdicts if v.get("GREEN")]
    source_only = [v["id"] for v in verdicts if v.get("SOURCE_ONLY")]
    pytest_ids = [v["id"] for v in verdicts
                  if v.get("pytest_invocations", 0) > 0]
    exits = {}
    for v in verdicts:
        exits[str(v.get("exit"))] = exits.get(str(v.get("exit")), 0) + 1
    empty = [v["id"] for v in verdicts if v.get("patch") == 0]
    print("traces=%d GREEN=%d SOURCE_ONLY=%d pytest_traces=%d (%s)" % (
        n, len(greens), len(source_only), len(pytest_ids),
        ",".join(pytest_ids)))
    print("exit_histogram=%s" % json.dumps(exits, sort_keys=True))
    print("empty_submit=%s" % ",".join(empty))
    for v in verdicts:
        g = v.get("gates", {})
        print("%-16s pc=%-5s ex=%-3s E%s S%s C%s F%s T%s D%s GREEN=%s SRC=%s | %s | %s" % (
            v.get("id"), v.get("patch"), v.get("exit"),
            int(g.get("exit_ok", 0)), int(g.get("size_ok", 0)),
            int(g.get("conf_ok", 0)), int(g.get("files1", 0)),
            int(g.get("no_test", 0)), int(g.get("no_docs", 0)),
            int(v.get("GREEN", 0)), int(v.get("SOURCE_ONLY", 0)),
            ",".join(v.get("edited", []))[:80],
            v.get("conf_evidence", "")[:80]))
    if args.out:
        Path(args.out).write_text(json.dumps(verdicts, indent=1))
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
