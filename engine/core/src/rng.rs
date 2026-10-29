//! CPython's `random.Random`, bit for bit: MT19937, CPython's integer seeding
//! (`init_by_array` over the seed's 32-bit words), `random()`'s 53-bit
//! construction and `randrange()`'s rejection sampling. A seed therefore
//! yields the same mazes, spawns and Laika coin flips as the Python port.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

#[derive(Clone)]
pub struct Rng {
    pub seed: u64,
    mt: [u32; N],
    index: usize,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut rng = Rng { seed, mt: [0; N], index: N + 1 };
        let mut key = Vec::new();
        let mut rest = seed;
        loop {
            key.push((rest & 0xffff_ffff) as u32);
            rest >>= 32;
            if rest == 0 {
                break;
            }
        }
        rng.init_by_array(&key);
        rng
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1] ^ (self.mt[i - 1] >> 30);
            self.mt[i] = 1_812_433_253u32.wrapping_mul(prev).wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            let prev = self.mt[i - 1] ^ (self.mt[i - 1] >> 30);
            self.mt[i] = (self.mt[i] ^ prev.wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let prev = self.mt[i - 1] ^ (self.mt[i - 1] >> 30);
            self.mt[i] = (self.mt[i] ^ prev.wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            let mag = |y: u32| if y & 1 != 0 { MATRIX_A } else { 0 };
            for kk in 0..N - M {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag(y);
            }
            for kk in N - M..N - 1 {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ mag(y);
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag(y);
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `random.random()`: uniform in [0, 1) with 53 bits of resolution.
    pub fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `random.randrange(n)` for 0 < n < 2^32.
    pub fn randrange(&mut self, n: u32) -> u32 {
        assert!(n > 0, "randrange({n})");
        let k = 32 - n.leading_zeros();
        let mut r = self.next_u32() >> (32 - k);
        while r >= n {
            r = self.next_u32() >> (32 - k);
        }
        r
    }

    /// AS2 `random(n)`.
    pub fn random_int(&mut self, n: f64) -> f64 {
        (self.random() * n).floor()
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn matches_cpython() {
        // python3 -c "import random; r = random.Random(42); print([r.random() for _ in range(3)], r.randrange(1000))"
        let mut r = Rng::new(42);
        assert_eq!(r.random(), 0.6394267984578837);
        assert_eq!(r.random(), 0.025010755222666936);
        assert_eq!(r.random(), 0.27502931836911926);
        assert_eq!(r.randrange(1000), 228);
        // A seed above 2^32 is split into two 32-bit key words.
        assert_eq!(Rng::new(2u64.pow(40) + 7).random(), 0.6137037779936511);
    }
}
