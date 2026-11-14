model bytes=491400032 sha256=74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
ppl ts slice-chars=102400 PPL=5.95 rc=0
ppl shakes slice-chars=102400 PPL=43.31 rc=0
gen story-en chars=287 rep4=0.000 rc=0 ok=True
gen explain-en chars=484 rep4=0.000 rc=0 ok=True
gen story-hi chars=380 rep4=0.000 rc=0 ok=True
gen story-mr chars=413 rep4=0.000 rc=0 ok=True
gen reason-add chars=231 rep4=0.000 rc=0 ok=True
gen carry chars=280 rep4=0.000 rc=0 ok=True
# Delegation baseline: frozen donor alone (measured, CI)
Donor qwen.gguf (491400032 bytes). Perplexity on 100KB slices (ctx 512). Generation: 6 probes, temp 0.7, 80 tokens.

| corpus | PPL |
|---|---|
| ts | 5.95 |
| shakes | 43.31 |

| probe | chars | rep4 | ok |
|---|---|---|---|
| story-en | 287 | 0.000 | True |
| explain-en | 484 | 0.000 | True |
| story-hi | 380 | 0.000 | True |
| story-mr | 413 | 0.000 | True |
| reason-add | 231 | 0.000 | True |
| carry | 280 | 0.000 | True |

delegation-eval: done
