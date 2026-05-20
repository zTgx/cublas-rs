/// Configuration for a GEMM operation: C = alpha * A * B + beta * C
#[derive(Debug, Clone)]
pub struct GemmConfig<T> {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub alpha: T,
    pub beta: T,
}

/// Layout convention for matrix operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    RowMajor,
    ColMajor,
}

/// Whether the matrix is transposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transpose {
    NoTrans,
    Trans,
}
