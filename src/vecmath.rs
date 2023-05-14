use rand::Rng;

pub const EMBEDDING_LENGTH: usize = 1536;
pub const EMBEDDING_BYTE_LENGTH: usize = EMBEDDING_LENGTH * 4;
pub type Embedding = [f32; EMBEDDING_LENGTH];
pub type EmbeddingBytes = [u8; EMBEDDING_BYTE_LENGTH];

pub fn empty_embedding() -> Embedding {
    [0.0; EMBEDDING_LENGTH]
}

pub fn random_embedding<R: Rng>(rng: &mut R) -> Embedding {
    let mut embedding = [0.0; EMBEDDING_LENGTH];
    rng.fill(&mut embedding[..]);

    embedding
}

pub fn random_normalized_embedding<R: Rng>(rng: &mut R) -> Embedding {
    let mut embedding = random_embedding(rng);
    normalize_vec(&mut embedding);

    embedding
}

#[inline]
fn clamp_01(f: f32) -> f32 {
    if f <= 0.0 {
        0.0
    } else if f >= 1.0 {
        1.0
    } else {
        f
    }
}

fn normalize_cosine_distance(f: f32) -> f32 {
    clamp_01((f-1.0)/-2.0)
}


pub fn normalized_cosine_distance_cpu(left: &Embedding, right: &Embedding) -> f32 {
    normalize_cosine_distance(left.iter().zip(right.iter()).map(|(l,r)|l*r).sum::<f32>())
}

#[cfg(feature = "simd")]
pub fn normalized_cosine_distance_simd(left: &Embedding, right: &Embedding) -> f32 {
    simd::normalized_cosine_distance_simd(left, right)
}

#[cfg(not(feature = "simd"))]
pub fn normalized_cosine_distance_simd(left: &Embedding, right: &Embedding) -> f32 {
    unimplemented!("simd support is not enabled");
}

pub fn normalize_vec_cpu(vec: &mut Embedding) {
    let mut sum = 0.0;
    for f in vec.iter() {
        sum += f*f;
    }
    let magnitude = sum.sqrt();
    eprintln!("cpu magnitude: {}", magnitude);

    for f in vec.iter_mut() {
        *f /= magnitude;
    }
}

#[cfg(feature = "simd")]
pub fn normalized_cosine_distance(left: &Embedding, right: &Embedding) -> f32 {
    simd::normalized_cosine_distance_simd(left, right)
}

#[cfg(feature = "simd")]
pub fn normalize_vec(vec: &mut Embedding) {
    simd::normalize_vec_simd(vec)
}

#[cfg(not(feature = "simd"))]
pub fn normalized_cosine_distance(left: &Embedding, right: &Embedding) -> f32 {
    normalized_cosine_distance_cpu(left, right)
}

#[cfg(not(feature = "simd"))]
pub fn normalize_vec(vec: &mut Embedding) {
    normalize_vec_cpu(vec)
}

macro_rules! simd_module {
    ($t:ident, $lanes:literal) => {
        pub mod simd {
            use packed_simd::$t;
            use aligned_box::AlignedBox;
            use super::*;

            pub fn aligned_box(e: Embedding) -> AlignedBox<Embedding> {
                AlignedBox::new(std::mem::align_of::<$t>(), e).unwrap()
            }

            pub fn normalized_cosine_distance_simd(left: &Embedding, right: &Embedding) -> f32 {
                if left.as_ptr().align_offset(std::mem::align_of::<$t>()) == 0
                    && right.as_ptr().align_offset(std::mem::align_of::<$t>()) == 0
                {
                    normalized_cosine_distance_simd_aligned(left, right)
                } else {
                    normalized_cosine_distance_simd_unaligned(left, right)
                }
            }

            pub fn normalize_vec_simd(vec: &mut Embedding) {
                if vec.as_ptr().align_offset(std::mem::align_of::<$t>()) == 0 {
                    normalize_vec_simd_aligned(vec)
                } else {
                    normalize_vec_simd_unaligned(vec)
                }
            }

            pub fn normalized_cosine_distance_simd_aligned(left: &Embedding, right: &Embedding) -> f32 {
                eprintln!("using {} ({} lanes)", stringify!($t), $lanes);
                let mut sum = <$t>::splat(0.);
                for x in 0..left.len()/$lanes {
                    let l = <$t>::from_slice_aligned(&left[x*$lanes..(x+1)*$lanes]);
                    let r = <$t>::from_slice_aligned(&right[x*$lanes..(x+1)*$lanes]);
                    sum += l * r;
                }
                normalize_cosine_distance(sum.sum())
            }

            pub fn normalize_vec_simd_aligned(vec: &mut Embedding) {
                eprintln!("using {} ({} lanes)", stringify!($t), $lanes);
                let mut sum = <$t>::splat(0.);
                let exp = <$t>::splat(2.);
                for x in 0..vec.len()/$lanes {
                    let part = <$t>::from_slice_aligned(&vec[x*$lanes..(x+1)*$lanes]);
                    sum += part*part;
                }
                let magnitude = sum.sum().sqrt();
                eprintln!("simd magnitude: {}", magnitude);
                let magnitude = <$t>::splat(magnitude);

                for x in 0..vec.len()/$lanes {
                    let scaled = <$t>::from_slice_aligned(&vec[x*$lanes..(x+1)*$lanes]) / magnitude;
                    scaled.write_to_slice_aligned(&mut vec[x*$lanes..(x+1)*$lanes]);
                }
            }

            pub fn normalized_cosine_distance_simd_unaligned(left: &Embedding, right: &Embedding) -> f32 {
                eprintln!("using {} ({} lanes, unaligned)", stringify!($t), $lanes);
                let mut sum = <$t>::splat(0.);
                for x in 0..left.len()/$lanes {
                    let l = <$t>::from_slice_unaligned(&left[x*$lanes..(x+1)*$lanes]);
                    let r = <$t>::from_slice_unaligned(&right[x*$lanes..(x+1)*$lanes]);
                    sum += l * r;
                }
                normalize_cosine_distance(sum.sum())
            }

            pub fn normalize_vec_simd_unaligned(vec: &mut Embedding) {
                eprintln!("using {} ({} lanes, unaligned)", stringify!($t), $lanes);
                let mut sum = <$t>::splat(0.);
                //let exp = <$t>::splat(2.);
                for x in 0..vec.len()/$lanes {
                    let part = <$t>::from_slice_unaligned(&vec[x*$lanes..(x+1)*$lanes]);
                    sum += part*part;
                }
                let magnitude = sum.sum().sqrt();
                eprintln!("simd magnitude: {}", magnitude);
                let magnitude = <$t>::splat(magnitude);

                for x in 0..vec.len()/$lanes {
                    let scaled = <$t>::from_slice_unaligned(&vec[x*$lanes..(x+1)*$lanes]) / magnitude;
                    scaled.write_to_slice_unaligned(&mut vec[x*$lanes..(x+1)*$lanes]);
                }
            }
        }
    }
}

// Here, we define the same module 4 times for different lane widths,
// depending on what cpu features are available to us. The algorithm
// is pretty much the same (see above), except it operates on more
// lanes at once.

// we're basically gonna assume that at the minimum, we're gonna be
// able to do 2 parallel f32 simd instructions. If either sse2 or neon
// as cpu features are available, we can do more.
#[cfg(all(feature = "simd",
          not(any(target_feature = "sse2",
                  target_feature = "neon"))))]
simd_module!(f32x2, 2);

// at most 4 lanes are available unless we're at least on avx (for
// x86) or aarch64 with neon support (for arm)
#[cfg(all(feature = "simd",
          not(any(target_feature = "avx",
                  target_cpu = "aarch64")),
          any(target_feature = "sse2",
              target_feature = "neon")))]
simd_module!(f32x4, 4);

// At most 8 lanes are available on aarch64 with neon support. x86
// might have 16 lane support though for avx512
#[cfg(all(feature = "simd",
          not(target_feature = "avx512"),
          any(target_feature = "avx",
              all(target_cpu = "aarch64",
                  target_feature = "neon"))))]
simd_module!(f32x8, 8);

#[cfg(all(feature = "simd", target_feature = "avx512"))]
simd_module!(f32x16, 16);

#[cfg(all(feature = "simd", test))]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use crate::vecmath::simd::{normalize_vec_simd_unaligned, normalized_cosine_distance_simd_unaligned};

    use super::*;
    #[test]
    fn ensure_normalize_equivalent() {
        let seed: u64 = 42;
        let mut rng = StdRng::seed_from_u64(seed);

        let mut e1 = random_embedding(&mut rng);
        let mut e2 = e1.clone();

        assert_eq!(e1, e2);

        normalize_vec_cpu(&mut e1);
        normalize_vec_simd_unaligned(&mut e2);

        eprintln!("distance (cpu): {}", normalized_cosine_distance_cpu(&e1, &e2));
        eprintln!("distance (simd): {}", normalized_cosine_distance_simd_unaligned(&e1, &e2));
        eprintln!("distance (simd same): {}", normalized_cosine_distance_simd_unaligned(&e1, &e1));

        let mut e3 = random_embedding(&mut rng);
        normalize_vec_cpu(&mut e3);
        eprintln!("distance (cpu): {}", normalized_cosine_distance_cpu(&e1, &e3));
        eprintln!("distance (simd): {}", normalized_cosine_distance_simd_unaligned(&e1, &e3));

        assert_eq!(e1, e2);
    }
}
