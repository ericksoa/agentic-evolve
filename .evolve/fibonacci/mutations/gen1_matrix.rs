use crate::Fibonacci;

pub struct EvolvedFib;

impl Fibonacci for EvolvedFib {
    fn fib(&self, n: u64) -> u64 {
        if n <= 1 {
            return n;
        }

        fn matrix_mult(a: [[u64; 2]; 2], b: [[u64; 2]; 2]) -> [[u64; 2]; 2] {
            [
                [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
                [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
            ]
        }

        fn matrix_pow(mut base: [[u64; 2]; 2], mut exp: u64) -> [[u64; 2]; 2] {
            let mut result = [[1, 0], [0, 1]];
            while exp > 0 {
                if exp & 1 == 1 {
                    result = matrix_mult(result, base);
                }
                base = matrix_mult(base, base);
                exp >>= 1;
            }
            result
        }

        let fib_matrix = [[1, 1], [1, 0]];
        let result = matrix_pow(fib_matrix, n - 1);
        result[0][0]
    }
}
