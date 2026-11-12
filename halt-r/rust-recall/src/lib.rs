// halt-r-recall: modern Hopfield recall core. Zero dependencies (core only).
// Iterates x <- X^T softmax(beta X x) until step delta < tol.
// Identical math to ref-hopfield.mjs recallModern. Compiles to wasm with:
//   cargo build --release --target wasm32-unknown-unknown

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// One recall step, in place. Returns max |delta| of the state.
/// patterns: P*D row-major, state: D, scratch: P workspace.
pub fn recall_step(
    patterns: &[f32],
    state: &mut [f32],
    scratch: &mut [f32],
    p: usize,
    d: usize,
    beta: f32,
) -> f32 {
    for i in 0..p {
        scratch[i] = beta * dot(&patterns[i * d..(i + 1) * d], state);
    }
    let mut m = f32::NEG_INFINITY;
    for i in 0..p {
        if scratch[i] > m {
            m = scratch[i];
        }
    }
    let mut s = 0.0f32;
    for i in 0..p {
        scratch[i] = (scratch[i] - m).exp();
        s += scratch[i];
    }
    let inv = 1.0f32 / s;
    let mut maxd = 0.0f32;
    for j in 0..d {
        let mut v = 0.0f32;
        for i in 0..p {
            v += scratch[i] * inv * patterns[i * d + j];
        }
        let dl = (v - state[j]).abs();
        if dl > maxd {
            maxd = dl;
        }
        state[j] = v;
    }
    maxd
}

/// C ABI for wasm: runs recall on caller-owned buffers.
/// patterns_ptr/state_ptr/scratch_ptr are BYTE offsets into the exported memory.
/// Returns iterations run (<= iters).
#[no_mangle]
pub unsafe extern "C" fn recall(
    patterns_ptr: *const f32,
    state_ptr: *mut f32,
    scratch_ptr: *mut f32,
    p: usize,
    d: usize,
    beta: f32,
    iters: usize,
    tol: f32,
) -> usize {
    let pats = core::slice::from_raw_parts(patterns_ptr, p * d);
    let st = core::slice::from_raw_parts_mut(state_ptr, d);
    let sc = core::slice::from_raw_parts_mut(scratch_ptr, p);
    let mut done = 0;
    for _ in 0..iters {
        done += 1;
        if recall_step(pats, st, sc, p, d, beta) < tol {
            break;
        }
    }
    done
}
