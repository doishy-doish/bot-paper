# HALT-R scramble coverage (2026-11-12)
Public compute, private numbers — same protocol as src/bot/SCRAMBLE.md.

## Covered (encrypt before any public push carrying private numbers)
- Future: real datasets, private eval results, operator-held targets.
  Wrapper shape: {enc:"AES-256-GCM-v1", nonce, data, tag, sha256, bytes, date}.
- NOT covered (synthetic, safe public): ref-hopfield.mjs, verify.mjs, toy JSONs
  (random patterns, synthetic DAGs — no private numbers in them).

## How
- Encrypt: public repo script scripts/scramble.mjs (already on bot-paper) or
  local src/bot/gh/scramble.mjs. Key from vault scratch/vault/keys/scramble.key
  (--key-file, never shell history) or SCRAMBLE_KEY env.
- Decrypt after pull: node scripts/scramble.mjs decrypt <file> --key-file <vault key>
  (writes .dec file, never overwrites wrapper — decrypted files must never be committed).
- Workflows: no pull_request triggers (forks can't see secrets); logs stay quiet
  (counts + hashes only) on any job touching private numbers. Current halt-r.yml
  is verify-only (synthetic prints, safe).

## Key locations (value never printed, never in chat)
1. scratch/vault/keys/scramble.key (ours, private scratch, NOT shipped)
2. private halt-r repo keys/scramble.key (storage only, Actions tab stays empty)
3. public bot-paper secret SCRAMBLE (Actions only, mapped to SCRAMBLE_KEY env)
Rotation: fresh 32 bytes -> replace all three together; keep old key till every
ciphertext file is re-encrypted. Git history: scrambling covers new commits only.
