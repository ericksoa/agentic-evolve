//! Fibonacci Benchmark
//!
//! Evolve from naive O(2^n) recursion to faster algorithms.

pub mod baselines;
pub mod evolved;

/// Trait for Fibonacci implementations
pub trait Fibonacci {
    /// Calculate the nth Fibonacci number
    /// fib(0) = 0, fib(1) = 1, fib(n) = fib(n-1) + fib(n-2)
    fn fib(&self, n: u64) -> u64;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_fib<F: Fibonacci>(f: &F) {
        // Known values
        assert_eq!(f.fib(0), 0);
        assert_eq!(f.fib(1), 1);
        assert_eq!(f.fib(2), 1);
        assert_eq!(f.fib(3), 2);
        assert_eq!(f.fib(4), 3);
        assert_eq!(f.fib(5), 5);
        assert_eq!(f.fib(10), 55);
        assert_eq!(f.fib(20), 6765);
        assert_eq!(f.fib(30), 832040);
        assert_eq!(f.fib(40), 102334155);
        assert_eq!(f.fib(50), 12586269025);
        // fib(93) is the largest that fits in u64
        assert_eq!(f.fib(92), 7540113804746346429);
    }

    #[test]
    fn test_naive() {
        // Only test small values for naive (too slow otherwise)
        let f = baselines::NaiveFib;
        assert_eq!(f.fib(0), 0);
        assert_eq!(f.fib(1), 1);
        assert_eq!(f.fib(10), 55);
        assert_eq!(f.fib(20), 6765);
    }

    #[test]
    fn test_iterative() {
        test_fib(&baselines::IterativeFib);
    }

    #[test]
    fn test_matrix() {
        test_fib(&baselines::MatrixFib);
    }

    #[test]
    fn test_evolved() {
        test_fib(&evolved::EvolvedFib);
    }
}
