#![feature(test)]
extern crate test;
use rand::{rngs::StdRng, SeedableRng};
use test::Bencher;

use terminusdb_semantic_indexer::vecmath::*;

#[bench]
fn bench_cpu_distance(b: &mut Bencher) {
    let seed: u64 = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let e1 = random_normalized_embedding(&mut rng);
    let e2 = random_normalized_embedding(&mut rng);

    b.iter(move || normalized_cosine_distance_cpu(&e1, &e2));
}

#[cfg(feature = "simd")]
mod simd_benches {
    use rand::{rngs::StdRng, SeedableRng};
    use terminusdb_semantic_indexer::vecmath::simd::*;
    use terminusdb_semantic_indexer::vecmath::*;
    use test::Bencher;

    #[bench]
    fn bench_simd_aligned_distance(b: &mut Bencher) {
        let seed: u64 = 42;
        let mut rng = StdRng::seed_from_u64(seed);
        let e1 = aligned_box(random_normalized_embedding(&mut rng));
        let e2 = aligned_box(random_normalized_embedding(&mut rng));

        b.iter(move || unsafe { normalized_cosine_distance_simd_aligned_unchecked(&e1, &e2) });
    }

    #[bench]
    fn bench_simd_unaligned_distance(b: &mut Bencher) {
        let seed: u64 = 42;
        let mut rng = StdRng::seed_from_u64(seed);
        let e1 = random_normalized_embedding(&mut rng);
        let e2 = random_normalized_embedding(&mut rng);

        b.iter(move || normalized_cosine_distance_simd_unaligned(&e1, &e2));
    }
}
