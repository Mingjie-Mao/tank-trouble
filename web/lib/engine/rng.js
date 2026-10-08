/**
 * CPython's `random.Random`, bit for bit.
 *
 * The Python port of the game (`tank_trouble_original/`) draws its mazes,
 * spawns, headings, crate timer and Laika's coin flips from `random.Random`.
 * Reproducing that generator exactly — MT19937, CPython's integer seeding
 * (`init_by_array` over the seed's 32-bit words), `random()`'s 53-bit
 * construction and `randrange()`'s rejection sampling — is what lets this
 * engine be checked against the Python one frame by frame.
 */

const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;

export class Rng {
  /** @param {number|null} seed a non-negative integer; null draws one at random */
  constructor(seed = null) {
    if (seed === null || seed === undefined) {
      seed = Math.floor(Math.random() * 0x100000000);
    }
    if (!Number.isSafeInteger(seed) || seed < 0) {
      throw new Error(`Rng seed must be a non-negative safe integer, got ${seed}`);
    }
    this.seed = seed;
    this.mt = new Uint32Array(N);
    this.index = N + 1;
    // CPython: the seed's absolute value, as little-endian 32-bit words.
    const key = [];
    let rest = seed;
    do {
      key.push(rest % 0x100000000);
      rest = Math.floor(rest / 0x100000000);
    } while (rest > 0);
    this.initByArray(key);
  }

  initGenrand(s) {
    const mt = this.mt;
    mt[0] = s >>> 0;
    for (let i = 1; i < N; i++) {
      const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = (Math.imul(1812433253, prev) + i) >>> 0;
    }
    this.index = N;
  }

  initByArray(key) {
    const mt = this.mt;
    this.initGenrand(19650218);
    let i = 1;
    let j = 0;
    for (let k = Math.max(N, key.length); k > 0; k--) {
      const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = ((mt[i] ^ Math.imul(prev, 1664525)) + key[j] + j) >>> 0;
      i++;
      j++;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
      if (j >= key.length) j = 0;
    }
    for (let k = N - 1; k > 0; k--) {
      const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = ((mt[i] ^ Math.imul(prev, 1566083941)) - i) >>> 0;
      i++;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
    }
    mt[0] = 0x80000000;
  }

  /** One raw 32-bit output. */
  nextUint32() {
    const mt = this.mt;
    if (this.index >= N) {
      let kk = 0;
      for (; kk < N - M; kk++) {
        const y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
        mt[kk] = mt[kk + M] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      }
      for (; kk < N - 1; kk++) {
        const y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
        mt[kk] = mt[kk + (M - N)] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      }
      const y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
      mt[N - 1] = mt[M - 1] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      this.index = 0;
    }
    let y = mt[this.index++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** `random.random()`: uniform in [0, 1) with 53 bits of resolution. */
  random() {
    const a = this.nextUint32() >>> 5;
    const b = this.nextUint32() >>> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
  }

  /** `random.randrange(n)` for 0 < n < 2^32, by CPython's rejection sampling. */
  randrange(n) {
    if (n <= 0) throw new Error(`randrange(${n})`);
    const k = 32 - Math.clz32(n);
    let r = this.nextUint32() >>> (32 - k);
    while (r >= n) r = this.nextUint32() >>> (32 - k);
    return r;
  }

  /** Full generator state, as a copy; assigning one restores it. */
  get state() {
    return { mt: Uint32Array.from(this.mt), index: this.index };
  }

  set state(value) {
    this.mt = Uint32Array.from(value.mt);
    this.index = value.index;
  }
}
