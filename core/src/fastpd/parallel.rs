/// Parallelization infrastructure for FastPD.
///
/// This module provides the `Joiner` trait for abstracting over sequential vs parallel
/// execution, and `ParallelSettings` for configuring parallelization.

/// Trait for abstracting over sequential vs parallel join operations.
///
/// This trait allows compile-time monomorphization: the compiler generates
/// separate code paths for sequential and parallel execution, eliminating
/// runtime branching in hot paths.
pub trait Joiner: Copy + Send + Sync {
    /// Joins two computations, executing them either sequentially or in parallel.
    ///
    /// # Type Parameters
    /// * `A`, `B`: Result types (must be `Send`)
    /// * `FA`, `FB`: Closures that produce `A` and `B` (must be `Send`)
    ///
    /// # Safety
    /// The closures must be safe to execute concurrently if `Self` is a parallel joiner.
    fn join<A, B, FA, FB>(left: FA, right: FB) -> (A, B)
    where
        FA: FnOnce() -> A + Send,
        FB: FnOnce() -> B + Send,
        A: Send,
        B: Send;
}

/// Sequential joiner: executes computations one after another.
#[derive(Debug, Clone, Copy)]
pub struct SeqJoin;

impl Joiner for SeqJoin {
    #[inline(always)]
    fn join<A, B, FA, FB>(left: FA, right: FB) -> (A, B)
    where
        FA: FnOnce() -> A + Send,
        FB: FnOnce() -> B + Send,
        A: Send,
        B: Send,
    {
        // Strictly sequential: no Rayon anywhere
        let a = left();
        let b = right();
        (a, b)
    }
}

/// Parallel joiner: executes computations using Rayon's `join()`.
#[derive(Debug, Clone, Copy)]
pub struct RayonJoin;

impl Joiner for RayonJoin {
    #[inline(always)]
    fn join<A, B, FA, FB>(left: FA, right: FB) -> (A, B)
    where
        FA: FnOnce() -> A + Send,
        FB: FnOnce() -> B + Send,
        A: Send,
        B: Send,
    {
        rayon::join(left, right)
    }
}

/// Minimal parallelization configuration.
///
/// This struct provides a single knob for controlling parallelization:
/// - `n_threads == 0 or 1`: Fully sequential execution, no Rayon overhead
/// - `n_threads >= 2`: Parallel execution using Rayon with specified thread count
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelSettings {
    /// Number of threads to use for parallel execution.
    /// Values <= 1 result in sequential execution.
    pub n_threads: usize,
}

impl ParallelSettings {
    /// Returns `true` if parallel execution should be used.
    #[inline]
    pub fn is_parallel(self) -> bool {
        self.n_threads > 1
    }

    /// Creates a sequential configuration (no parallelization).
    #[inline]
    pub fn sequential() -> Self {
        Self { n_threads: 1 }
    }

    /// Creates a parallel configuration with the specified number of threads.
    ///
    /// # Arguments
    /// * `n` - Number of threads. Values <= 1 are clamped to 1 (sequential).
    #[inline]
    pub fn with_n_threads(n: usize) -> Self {
        Self {
            n_threads: n.max(1),
        }
    }

    /// Creates a parallel configuration using all available CPU cores.
    ///
    /// Uses `std::thread::available_parallelism()` to determine the number of threads.
    /// Falls back to 2 threads if parallelism cannot be determined.
    #[inline]
    pub fn auto() -> Self {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .max(2);
        Self { n_threads: n }
    }
}

impl Default for ParallelSettings {
    fn default() -> Self {
        Self::sequential()
    }
}
