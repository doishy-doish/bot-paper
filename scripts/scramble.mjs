// scramble.mjs — public-runner encryption for bot-paper.
// Purpose: let the PUBLIC repo do unlimited-minutes compute while revealing nothing.
//   Actions encrypts data/*.json with AES-256-GCM (SCRAMBLE_KEY secret) before push.
//   We decrypt locally after `git pull` using the same key from the vault / private brain repo.
// Key format: 64 hex chars (32 bytes). Env: SCRAMBLE_KEY. Never log it, never commit it.
// Wrapper format (committed to public repo):
//   {enc:"AES-256-GCM-v1", nonce:<b64 12B>, data:<b64 ciphertext>, tag:<b64 16B>,
//    sha256:<hex of plaintext>, bytes:<plaintext len>, date:<YYYY-MM-DD>}
// Usage in Actions:
//   SCRAMBLE_KEY=${{ secrets.SCRAMBLE_KEY }} node scripts/scramble.mjs encrypt data/paper.json
// Usage on our side after pulling:
//   node scramble.mjs decrypt data/paper.json --key-file /path/to/scramble.key --out /tmp/paper.dec.json
//   (key file = raw 64-hex string, or JSON {"keyHex":"..."} — whitespace trimmed)
// Dashboard: see index.html unlock box — same AES-GCM via WebCrypto, key pasted locally only.
import {readFileSync, writeFileSync} from "fs";
import {createCipheriv, createDecipheriv, createHash, randomBytes} from "crypto";

const ALG = "AES-256-GCM-v1";

function loadKeyHex() {
  const h = (process.env.SCRAMBLE_KEY || "").trim();
  if (/^[0-9a-fA-F]{64}$/.test(h)) return h.toLowerCase();
  return null;
}

function loadKeyFromFile(path) {
  const raw = readFileSync(path, "utf8").trim();
  try {
    const j = JSON.parse(raw);
    if (j && typeof j.keyHex === "string" && /^[0-9a-fA-F]{64}$/.test(j.keyHex.trim()))
      return j.keyHex.trim().toLowerCase();
    if (j && typeof j.token === "string" && /^[0-9a-fA-F]{64}$/.test(j.token.trim()))
      return j.token.trim().toLowerCase();
  } catch { /* not JSON — treat as raw */ }
  if (/^[0-9a-fA-F]{64}$/.test(raw)) return raw.toLowerCase();
  throw new Error("key file must hold 64 hex chars (or JSON with keyHex)");
}

function sha256hex(buf) {
  return createHash("sha256").update(buf).digest("hex");
}

function encryptBuffer(plain, keyHex) {
  const key = Buffer.from(keyHex, "hex");
  const nonce = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", key, nonce);
  const ct = Buffer.concat([cipher.update(plain), cipher.final()]);
  const tag = cipher.getAuthTag();
  return {
    enc: ALG,
    nonce: nonce.toString("base64"),
    data: ct.toString("base64"),
    tag: tag.toString("base64"),
    sha256: sha256hex(plain),
    bytes: plain.length,
    date: new Date().toISOString().slice(0, 10),
  };
}

function decryptWrapper(w, keyHex) {
  if (!w || w.enc !== ALG) throw new Error("not a scramble wrapper (missing enc field)");
  const key = Buffer.from(keyHex, "hex");
  const nonce = Buffer.from(w.nonce, "base64");
  const ct = Buffer.from(w.data, "base64");
  const tag = Buffer.from(w.tag, "base64");
  const d = createDecipheriv("aes-256-gcm", key, nonce);
  d.setAuthTag(tag);
  const plain = Buffer.concat([d.update(ct), d.final()]);
  if (w.sha256 && sha256hex(plain) !== w.sha256) throw new Error("sha256 mismatch after decrypt");
  return plain;
}

function isWrapper(obj) {
  return obj && obj.enc === ALG && typeof obj.data === "string" && typeof obj.nonce === "string";
}

function cmdEncrypt(files) {
  const keyHex = loadKeyHex();
  if (!keyHex) {
    console.error("missing/invalid SCRAMBLE_KEY (need 64 hex chars in env)");
    process.exit(1);
  }
  for (const f of files) {
    let raw;
    try { raw = readFileSync(f); }
    catch (e) { console.error("skip " + f + ": cannot read"); process.exit(1); }
    // idempotent: skip files already wrapped (avoid double-encrypt on re-runs)
    try {
      const maybe = JSON.parse(raw.toString("utf8"));
      if (isWrapper(maybe)) { console.log("already scrambled: " + f); continue; }
    } catch { /* not JSON — encrypt raw */ }
    const w = encryptBuffer(raw, keyHex);
    writeFileSync(f, JSON.stringify(w, null, 1) + "\n");
    // public logs must not leak performance: hashes + sizes only, never symbols/prices/equity
    console.log("scrambled " + f + " bytes=" + w.bytes + " sha256=" + w.sha256.slice(0, 16) + "…");
  }
}

function cmdDecrypt(files, keyFile, out) {
  const keyHex = keyFile ? loadKeyFromFile(keyFile) : loadKeyHex();
  if (!keyHex) {
    console.error("no key: pass --key-file <path> or set SCRAMBLE_KEY env");
    process.exit(1);
  }
  const outs = [];
  for (const f of files) {
    const raw = readFileSync(f, "utf8");
    const w = JSON.parse(raw);
    const plain = decryptWrapper(w, keyHex);
    const dest = outs.length || files.length > 1
      ? null : null;
    if (out) {
      writeFileSync(out, plain);
      console.log("decrypted " + f + " -> " + out + " bytes=" + plain.length);
    } else {
      // default: sibling .dec.json (never overwrite the .json wrapper in place,
      // so a decrypted file can't be committed back by accident)
      const d = f.replace(/\.json$/, ".dec.json");
      const target = (d === f) ? (f + ".dec") : d;
      writeFileSync(target, plain);
      console.log("decrypted " + f + " -> " + target + " bytes=" + plain.length);
      outs.push(target);
    }
  }
  void dest;
  void outs;
}

const [, , cmd, ...rest] = process.argv;
if (cmd === "encrypt") {
  const files = rest.filter(a => !a.startsWith("--"));
  if (!files.length) { console.error("usage: scramble.mjs encrypt <file...>"); process.exit(1); }
  cmdEncrypt(files);
} else if (cmd === "decrypt") {
  let keyFile = null, out = null;
  const files = [];
  for (let i = 0; i < rest.length; i++) {
    if (rest[i] === "--key-file" && rest[i + 1]) { keyFile = rest[++i]; }
    else if (rest[i] === "--out" && rest[i + 1]) { out = rest[++i]; }
    else if (!rest[i].startsWith("--")) files.push(rest[i]);
  }
  if (!files.length) { console.error("usage: scramble.mjs decrypt <file...> [--key-file k] [--out o]"); process.exit(1); }
  if (out && files.length > 1) { console.error("decrypt: --out works with one input only"); process.exit(1); }
  cmdDecrypt(files, keyFile, out);
} else {
  console.error("usage: scramble.mjs encrypt <file...> | decrypt <file...> [--key-file k] [--out o]");
  process.exit(1);
}
