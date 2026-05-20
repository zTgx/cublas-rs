use half::f16;

/// Trait for scalar types supported by BLAS operations.
pub trait BlasScalar: Copy + Clone + Send + Sync + 'static {
    /// Additive identity (zero).
    fn zero() -> Self;
    /// Multiplicative identity (one).
    fn one() -> Self;
}

impl BlasScalar for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl BlasScalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl BlasScalar for f16 {
    fn zero() -> Self {
        f16::ZERO
    }
    fn one() -> Self {
        f16::ONE
    }
}
