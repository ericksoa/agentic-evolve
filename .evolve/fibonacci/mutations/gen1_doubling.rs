use crate::Fibonacci;

pub struct EvolvedFib;

impl Fibonacci for EvolvedFib {
    fn fib(&self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        if n == 1 {
            return 1;
        }

        fn fib_fast_doubling(n: u64) -> (u64, u64) {
            if n == 0 {
                return (0, 1);
            }

            let (a, b) = fib_fast_doubling(n >> 1);
            let c = a.wrapping_mul(b.wrapping_mul(2).wrapping_sub(a));
            let d = a.wrapping_mul(a).wrapping_add(b.wrapping_mul(b));

            if n & 1 == 0 {
                (c, d)
            } else {
                (d, c.wrapping_add(d))
            }
        }

        fib_fast_doubling(n).0
    }
}
