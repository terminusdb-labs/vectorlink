#![feature(test)]
extern crate test;
use rand::{rngs::StdRng, SeedableRng};
use test::Bencher;

use vectorlink::vecmath::*;

#[bench]
fn bench_cpu_distance(b: &mut Bencher) {
    let seed: u64 = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let e1 = random_normalized_embedding(&mut rng);
    let e2 = random_normalized_embedding(&mut rng);

    b.iter(move || normalized_cosine_distance_scalar(&e1, &e2));
}

mod simd_benches {
    use crate::simd::aligned_box;
    use rand::{rngs::StdRng, SeedableRng};
    use test::Bencher;
    use vectorlink::vecmath::*;

    #[bench]
    fn bench_simd_aligned_distance(b: &mut Bencher) {
        let seed: u64 = 42;
        let mut rng = StdRng::seed_from_u64(seed);
        let e1 = aligned_box(random_normalized_embedding(&mut rng));
        let e2 = aligned_box(random_normalized_embedding(&mut rng));

        b.iter(move || unsafe { normalized_cosine_distance_simd(&e1, &e2) });
    }

    #[bench]
    fn bench_simd_unaligned_distance(b: &mut Bencher) {
        let seed: u64 = 42;
        let mut rng = StdRng::seed_from_u64(seed);
        let e1 = random_normalized_embedding(&mut rng);
        let e2 = random_normalized_embedding(&mut rng);

        b.iter(move || normalized_cosine_distance_simd(&e1, &e2));
    }
}
