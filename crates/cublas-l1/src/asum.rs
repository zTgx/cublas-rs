// ASUM: sum of absolute values, returns sum(|x[i]|) for i in 0..n
//
// Classic block-reduction kernel — same pattern reused by dot, nrm2.

/// SASUM — sum of absolute values.
pub fn sasum(n: usize, x: &[f32]) -> f32 {
    let _ = (n, x);
    todo!("launch SASUM kernel")
}
