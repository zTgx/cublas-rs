// SAXPY: y = alpha * x + y

/// SAXPY kernel launch.
///
/// Computes y[i] = alpha * x[i] + y[i] for i in 0..n.
pub fn saxpy(n: usize, alpha: f32, x: &[f32], y: &mut [f32]) {
    let _ = (n, alpha, x, y);
    todo!("launch SAXPY kernel")
}
