// SCAL: x[i] = alpha * x[i]

/// SSCAL — scale a vector in place by alpha.
pub fn sscal(n: usize, alpha: f32, x: &mut [f32]) {
    let _ = (n, alpha, x);
    todo!("launch SSCAL kernel")
}
