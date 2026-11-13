# rust-direct-challenger

Direct-serve Rust entry: tokio + hyper HTTP/1.1 keep-alive, rusqlite, manual JSON.
**Direct** (no Nginx): serves `$HOST:$PORT` straight, keep-alive on.

## Stack
- Language: Rust 1.89.0 (pinned toolchain + pinned deps, see Cargo.toml)
- HTTP: hyper 1.6 http1 server, tokio multi-thread (2 workers), DB work on the
  blocking pool so the loop never stalls on SQLite
- SQLite: rusqlite 0.32 (bundled), one writer connection + 4-reader pool

## Optimizations and why
- **Direct, no Nginx.** The video shows Nginx costs 14–20% CPU; going direct saves
  it. Trade-off: we hold ~15k keep-alive connections ourselves (event-driven, cheap).
- **Read/write split.** WAL lets readers run concurrently; a single writer mutex
  serializes the rare writes (2% posts, ~15% likes-of-reads in the load model).
  No write starvation, no lock errors under k6.
- **One round trip per read.** Feed and post fetch author + like_count in a single
  GROUP BY query (indexed join + `likes_post_id_idx`), not N+1.
- **Prepared statements reused** (`prepare_cached` per pooled connection): plans are
  cached, data never is (rule 5).
- **SQLite pragmas inside our process** (all allowed): `WAL`, `synchronous=NORMAL`
  (durable commits, no OFF/MEMORY tricks), `busy_timeout=5000`,
  `cache_size=-65536` (64MB page cache), `mmap_size=128MB`, `temp_store=MEMORY`.
- **Manual JSON writer** for exact key order and zero reason to get escaping wrong;
  request bodies and JWT claims parsed with serde_json.
- **Auth verified every request**: HS256 HMAC-SHA256, alg pinned (none/other rejected),
  signature + `exp` checked, `sub` must be a positive-integer string, `username` a
  string. Nothing remembered between requests.
- **Allocation discipline**: feed buffer pre-sized 8KB, per-row push, no intermediate
  strings for numbers.

## Phase 2 (if this holds up)
Replace the feed/post SELECTs with direct SQLite B-tree page reads (same trick as the
top io_uring entries) and move the loop to io_uring. The router, auth, and JSON layers
stay untouched.
