// twelve-direct: Rust feed API for the $12 Server Challenge.
// Direct serve (no nginx), tokio + hyper http1 keep-alive, rusqlite pool,
// single writer + WAL/NORMAL, per-connection prepare_cached, manual JSON
// (exact key order), HMAC-SHA256 auth verified on every request.
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use rusqlite::{params, Connection};
use std::collections::VecDeque;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const FEED_SQL: &str = "SELECT p.id, p.body, p.created_at, u.username, COUNT(l.post_id) \
    FROM posts p JOIN users u ON u.id = p.user_id \
    LEFT JOIN likes l ON l.post_id = p.id \
    GROUP BY p.id ORDER BY p.created_at DESC, p.id DESC LIMIT 20";
const POST_SQL: &str = "SELECT p.id, p.body, p.created_at, u.username, COUNT(l.post_id) \
    FROM posts p JOIN users u ON u.id = p.user_id \
    LEFT JOIN likes l ON l.post_id = p.id \
    WHERE p.id = ?1 GROUP BY p.id";
const INSERT_POST_SQL: &str =
    "INSERT INTO posts (user_id, body) VALUES (?1, ?2) RETURNING id, created_at";
const POST_EXISTS_SQL: &str = "SELECT 1 FROM posts WHERE id = ?1";
const INSERT_LIKE_SQL: &str =
    "INSERT INTO likes (user_id, post_id) VALUES (?1, ?2) ON CONFLICT (user_id, post_id) DO NOTHING";

struct App {
    reads: Mutex<VecDeque<Connection>>,
    read_sem: Arc<tokio::sync::Semaphore>,
    write: Mutex<Connection>,
    secret: Vec<u8>,
    start: Instant,
}

fn open_db(path: &str) -> Connection {
    let c = Connection::open(path).expect("open sqlite");
    c.execute_batch(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000; \
         PRAGMA cache_size=-65536; PRAGMA mmap_size=134217728; PRAGMA temp_store=MEMORY;",
    )
    .expect("pragmas");
    c
}

// ---- JSON (manual, exact key order) ----
fn esc(out: &mut Vec<u8>, s: &str) {
    out.push(b'"');
    for b in s.bytes() {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0x00..=0x1F => {
                out.extend_from_slice(b"\\u00");
                out.push(b"0123456789abcdef"[(b >> 4) as usize]);
                out.push(b"0123456789abcdef"[(b & 15) as usize]);
            }
            _ => out.push(b),
        }
    }
    out.push(b'"');
}

fn push_post(out: &mut Vec<u8>, id: i64, body: &str, created: &str, author: &str, likes: i64) {
    out.extend_from_slice(b"{\"id\":");
    out.extend_from_slice(id.to_string().as_bytes());
    out.extend_from_slice(b",\"body\":");
    esc(out, body);
    out.extend_from_slice(b",\"created_at\":");
    esc(out, created);
    out.extend_from_slice(b",\"author\":");
    esc(out, author);
    out.extend_from_slice(b",\"like_count\":");
    out.extend_from_slice(likes.to_string().as_bytes());
    out.push(b'}');
}

fn json_resp(status: StatusCode, body: Vec<u8>) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

fn err(status: StatusCode, msg: &str) -> Response<Full<Bytes>> {
    let mut b = Vec::with_capacity(32);
    b.extend_from_slice(b"{\"error\":");
    esc(&mut b, msg);
    b.push(b'}');
    json_resp(status, b)
}

// ---- auth: HS256, verified every request, nothing remembered ----
struct Authed {
    uid: i64,
    name: String,
}

fn b64url_dec(s: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(s).ok()
}

fn check_auth(secret: &[u8], header: Option<&str>) -> Result<Authed, Response<Full<Bytes>>> {
    let h = header.unwrap_or("");
    if !h.starts_with("Bearer ") {
        return Err(err(StatusCode::UNAUTHORIZED, "missing bearer token"));
    }
    let tok = &h[7..];
    let mut parts = tok.split('.');
    let (hp, pp, sp) = match (parts.next(), parts.next(), parts.next()) {
        (Some(a), Some(b), Some(c)) if parts.next().is_none() => (a, b, c),
        _ => return Err(err(StatusCode::UNAUTHORIZED, "invalid or expired token")),
    };
    let hjson: serde_json::Value = serde_json::from_slice(&b64url_dec(hp).unwrap_or_default())
        .unwrap_or(serde_json::Value::Null);
    if hjson.get("alg").and_then(|v| v.as_str()) != Some("HS256") {
        return Err(err(StatusCode::UNAUTHORIZED, "invalid or expired token"));
    }
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    let mut mac = Hmac::<Sha256>::new_from_slice(secret).unwrap();
    mac.update(format!("{hp}.{pp}").as_bytes());
    let sig = b64url_dec(sp).unwrap_or_default();
    if mac.verify_slice(&sig).is_err() {
        return Err(err(StatusCode::UNAUTHORIZED, "invalid or expired token"));
    }
    let claims: serde_json::Value = serde_json::from_slice(&b64url_dec(pp).unwrap_or_default())
        .unwrap_or(serde_json::Value::Null);
    let uid: i64 = match claims.get("sub").and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) => match s.parse() {
            Ok(v) if v > 0 => v,
            _ => return Err(err(StatusCode::UNAUTHORIZED, "invalid token payload")),
        },
        _ => return Err(err(StatusCode::UNAUTHORIZED, "invalid token payload")),
    };
    let name = match claims.get("username").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => return Err(err(StatusCode::UNAUTHORIZED, "invalid token payload")),
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    match claims.get("exp").and_then(|v| v.as_i64()) {
        Some(e) if e > now => {}
        _ => return Err(err(StatusCode::UNAUTHORIZED, "invalid or expired token")),
    }
    Ok(Authed { uid, name })
}

fn valid_id(s: &str) -> Option<i64> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    match s.parse::<i64>() {
        Ok(v) if v >= 1 => Some(v),
        _ => None,
    }
}

// ---- DB work (runs on the blocking pool) ----
// Never panic on a poisoned/empty pool: a panic while holding one of these locks
// poisons the mutex and collapses the whole server under load (found at 40 VUs:
// 5th concurrent read hit expect("pool") on the 4-conn pool -> cascade -> 99.5% err).
fn unpoison<T>(m: std::sync::LockResult<T>) -> T {
    m.unwrap_or_else(|e| e.into_inner())
}
fn with_read<F>(app: &Arc<App>, f: F) -> Response<Full<Bytes>>
where
    F: FnOnce(&Connection) -> Result<Response<Full<Bytes>>, String>,
{
    let mut pool = unpoison(app.reads.lock());
    let c = match pool.pop_front() {
        Some(c) => c,
        None => return err(StatusCode::INTERNAL_SERVER_ERROR, "internal server error"),
    };
    let r =
        f(&c).unwrap_or_else(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "internal server error"));
    pool.push_back(c);
    r
}

fn with_write<F>(app: &Arc<App>, f: F) -> Response<Full<Bytes>>
where
    F: FnOnce(&Connection) -> Result<Response<Full<Bytes>>, String>,
{
    let c = unpoison(app.write.lock());
    f(&c).unwrap_or_else(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "internal server error"))
}

fn health(app: &App) -> Response<Full<Bytes>> {
    let up = app.start.elapsed().as_secs();
    let ok = (|| -> rusqlite::Result<()> {
        let mut pool = unpoison(app.reads.lock());
        let c = match pool.pop_front() {
            Some(c) => c,
            None => return Err(rusqlite::Error::InvalidQuery),
        };
        let r = c.query_row("SELECT 1", [], |_| Ok(()));
        pool.push_back(c);
        r
    })();
    match ok {
        Ok(()) => {
            let b = format!("{{\"status\":\"ok\",\"db\":\"ok\",\"uptime_s\":{up}}}").into_bytes();
            json_resp(StatusCode::OK, b)
        }
        Err(e) => {
            let mut b = Vec::with_capacity(64);
            b.extend_from_slice(b"{\"status\":\"degraded\",\"db\":\"unreachable\",\"error\":");
            esc(&mut b, &e.to_string());
            b.push(b'}');
            json_resp(StatusCode::SERVICE_UNAVAILABLE, b)
        }
    }
}

fn db_feed(c: &Connection) -> Result<Response<Full<Bytes>>, String> {
    let mut st = c.prepare_cached(FEED_SQL).map_err(|e| e.to_string())?;
    let rows = st
        .query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, i64>(4)?,
            ))
        })
        .map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(8192);
    out.extend_from_slice(b"{\"posts\":[");
    let mut first = true;
    for row in rows {
        let (id, body, created, author, likes) = row.map_err(|e| e.to_string())?;
        if !first {
            out.push(b',');
        }
        first = false;
        push_post(&mut out, id, &body, &created, &author, likes);
    }
    out.extend_from_slice(b"]}");
    Ok(json_resp(StatusCode::OK, out))
}

fn db_post(c: &Connection, pid: i64) -> Result<Response<Full<Bytes>>, String> {
    let mut st = c.prepare_cached(POST_SQL).map_err(|e| e.to_string())?;
    let mut rows = st.query(params![pid]).map_err(|e| e.to_string())?;
    match rows.next().map_err(|e| e.to_string())? {
        Some(r) => {
            let id: i64 = r.get(0).map_err(|e: rusqlite::Error| e.to_string())?;
            let body: String = r.get(1).map_err(|e: rusqlite::Error| e.to_string())?;
            let created: String = r.get(2).map_err(|e: rusqlite::Error| e.to_string())?;
            let author: String = r.get(3).map_err(|e: rusqlite::Error| e.to_string())?;
            let likes: i64 = r.get(4).map_err(|e: rusqlite::Error| e.to_string())?;
            let mut out = Vec::with_capacity(512);
            out.extend_from_slice(b"{\"post\":");
            push_post(&mut out, id, &body, &created, &author, likes);
            out.push(b'}');
            Ok(json_resp(StatusCode::OK, out))
        }
        None => Ok(err(StatusCode::NOT_FOUND, "post not found")),
    }
}

fn db_create(c: &Connection, auth: Authed, raw: &[u8]) -> Result<Response<Full<Bytes>>, String> {
    let v: serde_json::Value = match serde_json::from_slice(raw) {
        Ok(v) => v,
        Err(_) => return Ok(err(StatusCode::BAD_REQUEST, "malformed JSON body")),
    };
    let body = match v.get("body").and_then(|b| b.as_str()) {
        Some(s) => s.trim().to_string(),
        None => return Ok(err(StatusCode::BAD_REQUEST, "body is required")),
    };
    if body.is_empty() {
        return Ok(err(StatusCode::BAD_REQUEST, "body is required"));
    }
    if body.chars().count() > 500 {
        return Ok(err(StatusCode::BAD_REQUEST, "body must be at most 500 characters"));
    }
    let mut st = c.prepare_cached(INSERT_POST_SQL).map_err(|e| e.to_string())?;
    let mut rows = st.query(params![auth.uid, body]).map_err(|e| e.to_string())?;
    let (id, created): (i64, String) = match rows.next().map_err(|e| e.to_string())? {
        Some(r) => (
            r.get(0).map_err(|e: rusqlite::Error| e.to_string())?,
            r.get(1).map_err(|e: rusqlite::Error| e.to_string())?,
        ),
        None => return Err("no returning row".into()),
    };
    let mut out = Vec::with_capacity(512);
    out.extend_from_slice(b"{\"post\":");
    push_post(&mut out, id, &body, &created, &auth.name, 0);
    out.push(b'}');
    Ok(json_resp(StatusCode::CREATED, out))
}

fn db_like(c: &Connection, auth: Authed, pid: i64) -> Result<Response<Full<Bytes>>, String> {
    let exists: bool = c.query_row(POST_EXISTS_SQL, params![pid], |_| Ok(())).is_ok();
    if !exists {
        return Ok(err(StatusCode::NOT_FOUND, "post not found"));
    }
    let n = c.execute(INSERT_LIKE_SQL, params![auth.uid, pid]).map_err(|e| e.to_string())?;
    let first = n == 1;
    let b = format!(
        "{{\"liked\":true,\"already_liked\":{},\"post_id\":{pid}}}",
        if first { "false" } else { "true" }
    )
    .into_bytes();
    Ok(json_resp(if first { StatusCode::CREATED } else { StatusCode::OK }, b))
}

// ---- async dispatch: route sync, collect POST bodies, DB work off-loop ----
async fn spawn_read<F>(app: Arc<App>, f: F) -> Response<Full<Bytes>>
where
    F: FnOnce(&Connection) -> Result<Response<Full<Bytes>>, String> + Send + 'static,
{
    // Gate read concurrency on the pool size: without this, the (N+1)th concurrent
    // read finds an empty pool and used to panic -> poison -> total collapse.
    // The permit is held across the blocking work, so at most POOL reads run at once.
    let app2 = app.clone();
    let permit = match app.read_sem.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return err(StatusCode::INTERNAL_SERVER_ERROR, "internal server error"),
    };
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        with_read(&app2, f)
    })
    .await
    .unwrap_or_else(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "internal server error"))
}

async fn spawn_write<F>(app: Arc<App>, f: F) -> Response<Full<Bytes>>
where
    F: FnOnce(&Connection) -> Result<Response<Full<Bytes>>, String> + Send + 'static,
{
    tokio::task::spawn_blocking(move || with_write(&app, f))
        .await
        .unwrap_or_else(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "internal server error"))
}

async fn dispatch(app: Arc<App>, req: Request<Incoming>) -> Result<Response<Full<Bytes>>, Infallible> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let segs: Vec<&str> = path.split('/').collect();
    let auth_h = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    match (method, segs.as_slice()) {
        (Method::GET, ["", "health"]) => Ok(health(&app)),
        (Method::GET, ["", "feed"]) => Ok(spawn_read(app, db_feed).await),
        (Method::GET, ["", "posts", id]) => match valid_id(id) {
            Some(pid) => Ok(spawn_read(app, move |c| db_post(c, pid)).await),
            None => Ok(err(StatusCode::BAD_REQUEST, "invalid post id")),
        },
        (Method::POST, ["", "posts"]) => {
            let auth = match check_auth(&app.secret, auth_h.as_deref()) {
                Ok(a) => a,
                Err(r) => return Ok(r),
            };
            let bytes = match req.into_body().collect().await {
                Ok(b) => b.to_bytes(),
                Err(_) => return Ok(err(StatusCode::INTERNAL_SERVER_ERROR, "internal server error")),
            };
            Ok(spawn_write(app, move |c| db_create(c, auth, &bytes)).await)
        }
        (Method::POST, ["", "posts", id, "like"]) => {
            let auth = match check_auth(&app.secret, auth_h.as_deref()) {
                Ok(a) => a,
                Err(r) => return Ok(r),
            };
            match valid_id(id) {
                Some(pid) => Ok(spawn_write(app, move |c| db_like(c, auth, pid)).await),
                None => Ok(err(StatusCode::BAD_REQUEST, "invalid post id")),
            }
        }
        _ => Ok(err(StatusCode::NOT_FOUND, "not found")),
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    let path = std::env::var("SQLITE_PATH").expect("SQLITE_PATH");
    let secret = std::env::var("JWT_SECRET").expect("JWT_SECRET").into_bytes();
    let host = std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(3000);

    let write = open_db(&path);
    let mut reads = VecDeque::with_capacity(4);
    for _ in 0..4 {
        reads.push_back(open_db(&path));
    }
    let app = Arc::new(App {
        read_sem: Arc::new(tokio::sync::Semaphore::new(reads.len())),
        reads: Mutex::new(reads),
        write: Mutex::new(write),
        secret,
        start: Instant::now(),
    });

    let addr: SocketAddr = format!("{host}:{port}").parse().expect("bind addr");
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    loop {
        let (stream, _) = listener.accept().await.expect("accept");
        let app = app.clone();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let svc = service_fn(move |req| dispatch(app.clone(), req));
            let _ = hyper::server::conn::http1::Builder::new()
                .keep_alive(true)
                .serve_connection(io, svc)
                .await;
        });
    }
}
