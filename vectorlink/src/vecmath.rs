#![allow(unused)]
use rand::Rng;

pub const EMBEDDING_LENGTH: usize = 1536;
pub const EMBEDDING_BYTE_LENGTH: usize = EMBEDDING_LENGTH * 4;
pub type Embedding = [f32; EMBEDDING_LENGTH];
pub type EmbeddingBytes = [u8; EMBEDDING_BYTE_LENGTH];

pub const QUANTIZED_32_EMBEDDING_LENGTH: usize = 48;
pub const QUANTIZED_32_EMBEDDING_BYTE_LENGTH: usize = QUANTIZED_32_EMBEDDING_LENGTH * 2;
pub type Quantized32Embedding = [u16; QUANTIZED_32_EMBEDDING_LENGTH];
pub type Quantized32EmbeddingBytes = [u8; QUANTIZED_32_EMBEDDING_BYTE_LENGTH];

pub const CENTROID_32_LENGTH: usize = 32;
pub const CENTROID_32_BYTE_LENGTH: usize = CENTROID_32_LENGTH * 4;
pub type Centroid32 = [f32; CENTROID_32_LENGTH];
pub type Centroid32Bytes = [u8; CENTROID_32_BYTE_LENGTH];

pub const QUANTIZED_16_EMBEDDING_LENGTH: usize = 96;
pub const QUANTIZED_16_EMBEDDING_BYTE_LENGTH: usize = QUANTIZED_16_EMBEDDING_LENGTH * 2;
pub type Quantized16Embedding = [u16; QUANTIZED_16_EMBEDDING_LENGTH];
pub type Quantized16EmbeddingBytes = [u8; QUANTIZED_16_EMBEDDING_BYTE_LENGTH];

pub const CENTROID_16_LENGTH: usize = 16;
pub const CENTROID_16_BYTE_LENGTH: usize = CENTROID_16_LENGTH * 4;
pub type Centroid16 = [f32; CENTROID_16_LENGTH];
pub type Centroid16Bytes = [u8; CENTROID_16_BYTE_LENGTH];

pub const QUANTIZED_8_EMBEDDING_LENGTH: usize = 192;
pub const QUANTIZED_8_EMBEDDING_BYTE_LENGTH: usize = QUANTIZED_8_EMBEDDING_LENGTH * 2;
pub type Quantized8Embedding = [u16; QUANTIZED_8_EMBEDDING_LENGTH];
pub type Quantized8EmbeddingBytes = [u8; QUANTIZED_8_EMBEDDING_BYTE_LENGTH];

pub const CENTROID_8_LENGTH: usize = 8;
pub const CENTROID_8_BYTE_LENGTH: usize = CENTROID_8_LENGTH * 4;
pub type Centroid8 = [f32; CENTROID_8_LENGTH];
pub type Centroid8Bytes = [u8; CENTROID_8_BYTE_LENGTH];

pub const QUANTIZED_4_EMBEDDING_LENGTH: usize = 384;
pub const QUANTIZED_4_EMBEDDING_BYTE_LENGTH: usize = QUANTIZED_4_EMBEDDING_LENGTH * 2;
pub type Quantized4Embedding = [u16; QUANTIZED_4_EMBEDDING_LENGTH];
pub type Quantized4EmbeddingBytes = [u8; QUANTIZED_4_EMBEDDING_BYTE_LENGTH];

pub const CENTROID_4_LENGTH: usize = 4;
pub const CENTROID_4_BYTE_LENGTH: usize = CENTROID_4_LENGTH * 4;
pub type Centroid4 = [f32; CENTROID_4_LENGTH];
pub type Centroid4Bytes = [u8; CENTROID_4_BYTE_LENGTH];

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
pub fn clamp_01(f: f32) -> f32 {
    if f <= 0.0 {
        0.0
    } else if f >= 1.0 {
        1.0
    } else {
        f
    }
}

fn normalize_cosine_distance(f: f32) -> f32 {
    clamp_01((f - 1.0) / -2.0)
}

pub fn cosine_distance_scalar(left: &Embedding, right: &Embedding) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(l, r)| l * r)
        .sum::<f32>()
}

pub fn normalized_cosine_distance_scalar(left: &Embedding, right: &Embedding) -> f32 {
    normalize_cosine_distance(cosine_distance_scalar(left, right))
}

pub fn normalized_cosine_distance_simd(left: &Embedding, right: &Embedding) -> f32 {
    simd::normalized_cosine_distance_simd(left, right)
}

pub fn normalize_vec_scalar(vec: &mut Embedding) {
    let mut sum = 0.0;
    for f in vec.iter() {
        sum += f * f;
    }
    let magnitude = sum.sqrt();
    //eprintln!("scalar magnitude: {}", magnitude);

    for f in vec.iter_mut() {
        *f /= magnitude;
    }
}

pub fn normalized_cosine_distance(left: &Embedding, right: &Embedding) -> f32 {
    simd::normalized_cosine_distance_simd(left, right)
}

pub fn normalized_cosine_distance_32(v1: &Centroid32, v2: &Centroid32) -> f32 {
    normalized_cosine_distance_32_simd(v1, v2)
}

pub fn normalized_cosine_distance_32_scalar(v1: &Centroid32, v2: &Centroid32) -> f32 {
    normalize_cosine_distance(
        v1.iter()
            .zip(v2.iter())
            .map(|(f1, f2)| f1 * f2)
            .sum::<f32>(),
    )
}

pub use simd::normalized_cosine_distance_32_simd;

use crate::comparator::DistanceCalculator;

pub fn euclidean_distance_32(v1: &Centroid32, v2: &Centroid32) -> f32 {
    simd::euclidean_distance_32_simd(v1, v2)
}

pub fn euclidean_partial_distance_32(v1: &Centroid32, v2: &Centroid32) -> f32 {
    simd::euclidean_partial_distance_32_simd(v1, v2)
}

pub fn euclidean_distance_16(v1: &Centroid16, v2: &Centroid16) -> f32 {
    simd::euclidean_distance_16_simd(v1, v2)
}

pub fn euclidean_partial_distance_4(v1: &Centroid4, v2: &Centroid4) -> f32 {
    simd::euclidean_partial_distance_4_simd(v1, v2)
}

pub fn euclidean_partial_distance_8(v1: &Centroid8, v2: &Centroid8) -> f32 {
    simd::euclidean_partial_distance_8_simd(v1, v2)
}

pub fn euclidean_partial_distance_16(v1: &Centroid16, v2: &Centroid16) -> f32 {
    simd::euclidean_partial_distance_16_simd(v1, v2)
}

pub fn cosine_partial_distance_32(v1: &Centroid32, v2: &Centroid32) -> f32 {
    simd::cosine_partial_distance_32_simd(v1, v2)
}

#[derive(Default)]
pub struct EuclideanDistance32;
impl DistanceCalculator for EuclideanDistance32 {
    type T = Centroid32;

    fn partial_distance(&self, left: &Self::T, right: &Self::T) -> f32 {
        euclidean_partial_distance_32(left, right)
    }

    fn finalize_partial_distance(&self, distance: f32) -> f32 {
        distance.sqrt()
    }

    fn aggregate_partial_distances(&self, distances: &[f32]) -> f32 {
        assert!(distances.len() == QUANTIZED_32_EMBEDDING_LENGTH);
        let cast = unsafe { &*(distances.as_ptr() as *const [f32; QUANTIZED_32_EMBEDDING_LENGTH]) };
        simd::sum_48(cast).sqrt()
    }
}

#[derive(Default)]
pub struct EuclideanDistance16;
impl DistanceCalculator for EuclideanDistance16 {
    type T = Centroid16;

    fn partial_distance(&self, left: &Self::T, right: &Self::T) -> f32 {
        euclidean_partial_distance_16(left, right)
    }

    fn finalize_partial_distance(&self, distance: f32) -> f32 {
        distance.sqrt()
    }

    fn aggregate_partial_distances(&self, distances: &[f32]) -> f32 {
        assert!(distances.len() == QUANTIZED_16_EMBEDDING_LENGTH);
        let cast = unsafe { &*(distances.as_ptr() as *const [f32; QUANTIZED_16_EMBEDDING_LENGTH]) };
        simd::sum_96(cast).sqrt()
    }
}

#[derive(Default)]
pub struct EuclideanDistance8;
impl DistanceCalculator for EuclideanDistance8 {
    type T = Centroid8;

    fn partial_distance(&self, left: &Self::T, right: &Self::T) -> f32 {
        euclidean_partial_distance_8(left, right)
    }

    fn finalize_partial_distance(&self, distance: f32) -> f32 {
        distance.sqrt()
    }

    fn aggregate_partial_distances(&self, distances: &[f32]) -> f32 {
        assert!(distances.len() == QUANTIZED_8_EMBEDDING_LENGTH);
        let cast = unsafe { &*(distances.as_ptr() as *const [f32; QUANTIZED_8_EMBEDDING_LENGTH]) };
        simd::sum_192(cast).sqrt()
    }
}

#[derive(Default)]
pub struct EuclideanDistance4;
impl DistanceCalculator for EuclideanDistance4 {
    type T = Centroid4;

    fn partial_distance(&self, left: &Self::T, right: &Self::T) -> f32 {
        euclidean_partial_distance_4(left, right)
    }

    fn finalize_partial_distance(&self, distance: f32) -> f32 {
        distance.sqrt()
    }

    fn aggregate_partial_distances(&self, distances: &[f32]) -> f32 {
        assert!(distances.len() == QUANTIZED_4_EMBEDDING_LENGTH);
        let cast = unsafe { &*(distances.as_ptr() as *const [f32; QUANTIZED_4_EMBEDDING_LENGTH]) };
        simd::sum_384(cast).sqrt()
    }
}

pub fn normalize_vec(vec: &mut Embedding) {
    simd::normalize_vec_simd(vec)
}

pub fn sum_48(vec: &[f32; 48]) -> f32 {
    simd::sum_48(vec)
}

pub fn sum_96(vec: &[f32; 96]) -> f32 {
    simd::sum_96(vec)
}

pub fn sum_192(vec: &[f32; 192]) -> f32 {
    simd::sum_192(vec)
}

pub fn sum_384(vec: &[f32; 384]) -> f32 {
    simd::sum_384(vec)
}

pub mod simd {
    use super::*;
    use aligned_box::AlignedBox;
    use std::simd::{f32x16, f32x4, f32x8, num::SimdFloat, Simd};

    pub fn aligned_box(e: Embedding) -> AlignedBox<Embedding> {
        AlignedBox::new(std::mem::align_of::<f32x16>(), e).unwrap()
    }

    pub fn normalized_cosine_distance_simd(left: &Embedding, right: &Embedding) -> f32 {
        let mut sum = <f32x16>::splat(0.);
        for x in 0..left.len() / 16 {
            let l = <f32x16>::from_slice(&left[x * 16..(x + 1) * 16]);
            let r = <f32x16>::from_slice(&right[x * 16..(x + 1) * 16]);
            sum += l * r;
        }
        normalize_cosine_distance(sum.reduce_sum())
    }

    pub fn normalized_cosine_distance_32_simd(left: &Centroid32, right: &Centroid32) -> f32 {
        let mut sum = <f32x16>::splat(0.);
        let l = <f32x16>::from_slice(&left[0..16]);
        let r = <f32x16>::from_slice(&right[0..16]);
        sum += l * r;
        let l = <f32x16>::from_slice(&left[16..32]);
        let r = <f32x16>::from_slice(&right[16..32]);
        sum += l * r;
        normalize_cosine_distance(sum.reduce_sum())
    }

    pub fn cosine_partial_distance_32_simd(left: &Centroid32, right: &Centroid32) -> f32 {
        let mut sum = <f32x16>::splat(0.);
        let l = <f32x16>::from_slice(&left[0..16]);
        let r = <f32x16>::from_slice(&right[0..16]);
        sum += l * r;
        let l = <f32x16>::from_slice(&left[16..32]);
        let r = <f32x16>::from_slice(&right[16..32]);
        sum += l * r;
        sum.reduce_sum()
    }

    pub fn euclidean_distance_32_simd(left: &Centroid32, right: &Centroid32) -> f32 {
        euclidean_partial_distance_32_simd(left, right).sqrt()
    }

    pub fn euclidean_partial_distance_32_simd(left: &Centroid32, right: &Centroid32) -> f32 {
        let mut sum = <f32x16>::splat(0.);
        let l = <f32x16>::from_slice(&left[0..16]);
        let r = <f32x16>::from_slice(&right[0..16]);
        let res = (l - r);
        sum += res * res;
        let l = <f32x16>::from_slice(&left[16..32]);
        let r = <f32x16>::from_slice(&right[16..32]);
        let res = (l - r);
        sum += res * res;
        sum.reduce_sum()
    }

    pub fn euclidean_distance_16_simd(left: &Centroid16, right: &Centroid16) -> f32 {
        euclidean_partial_distance_16_simd(left, right).sqrt()
    }

    pub fn euclidean_partial_distance_4_simd(left: &Centroid4, right: &Centroid4) -> f32 {
        let mut sum = <f32x4>::splat(0.);
        let l = <f32x4>::from_slice(&left[0..16]);
        let r = <f32x4>::from_slice(&right[0..16]);
        let res = (l - r);
        sum += res * res;
        sum.reduce_sum()
    }

    pub fn euclidean_partial_distance_8_simd(left: &Centroid8, right: &Centroid8) -> f32 {
        let mut sum = <f32x8>::splat(0.);
        let l = <f32x8>::from_slice(&left[0..16]);
        let r = <f32x8>::from_slice(&right[0..16]);
        let res = (l - r);
        sum += res * res;
        sum.reduce_sum()
    }

    pub fn euclidean_partial_distance_16_simd(left: &Centroid16, right: &Centroid16) -> f32 {
        let mut sum = <f32x16>::splat(0.);
        let l = <f32x16>::from_slice(&left[0..16]);
        let r = <f32x16>::from_slice(&right[0..16]);
        let res = (l - r);
        sum += res * res;
        sum.reduce_sum()
    }

    pub fn normalize_vec_simd(vec: &mut Embedding) {
        //eprintln!("using {} ({} lanes, unaligned)", stringify!(f32x16), 16);
        let mut sum = <f32x16>::splat(0.);
        //let exp = <f32x16>::splat(2.);
        for x in 0..vec.len() / 16 {
            let part = <f32x16>::from_slice(&vec[x * 16..(x + 1) * 16]);
            sum += part * part;
        }
        let magnitude = sum.reduce_sum().sqrt();
        //eprintln!("simd magnitude: {}", magnitude);
        let magnitude = <f32x16>::splat(magnitude);

        for x in 0..vec.len() / 16 {
            let subvecs = &mut vec[x * 16..(x + 1) * 16];
            let scaled = <f32x16>::from_slice(subvecs) / magnitude;
            let array = scaled.to_array();
            subvecs.copy_from_slice(array.as_ref());
        }
    }

    pub fn sum_48(array: &[f32; 48]) -> f32 {
        let mut sum = <f32x16>::from_slice(&array[..16]);
        sum += <f32x16>::from_slice(&array[16..32]);
        sum += <f32x16>::from_slice(&array[32..48]);

        sum.reduce_sum()
    }

    pub fn sum_96(array: &[f32; 96]) -> f32 {
        let mut sum = <f32x16>::from_slice(&array[..16]);
        sum += <f32x16>::from_slice(&array[16..32]);
        sum += <f32x16>::from_slice(&array[32..48]);
        sum += <f32x16>::from_slice(&array[48..64]);
        sum += <f32x16>::from_slice(&array[48..80]);
        sum += <f32x16>::from_slice(&array[80..96]);

        sum.reduce_sum()
    }

    pub fn sum_192(array: &[f32; 192]) -> f32 {
        let mut sum = <f32x16>::from_slice(&array[..16]);
        for i in 1..12 {
            sum += <f32x16>::from_slice(&array[16 * i..16 * (i + 1)]);
        }
        sum.reduce_sum()
    }

    pub fn sum_384(array: &[f32; 384]) -> f32 {
        let mut sum = <f32x16>::from_slice(&array[..16]);
        for i in 1..24 {
            sum += <f32x16>::from_slice(&array[16 * i..16 * (i + 1)]);
        }
        sum.reduce_sum()
    }
}

#[cfg(test)]
mod tests {
    use assert_float_eq::*;
    use rand::{rngs::StdRng, SeedableRng};
    use tests::simd::normalize_vec_simd;

    use super::*;
    #[test]
    fn ensure_normalize_equivalent() {
        let seed: u64 = 42;
        let mut rng = StdRng::seed_from_u64(seed);

        let mut e1 = random_embedding(&mut rng);
        let mut e2 = e1.clone();

        assert_eq!(e1, e2);

        normalize_vec_scalar(&mut e1);
        normalize_vec_simd(&mut e2);

        eprintln!(
            "distance (scalar): {}",
            normalized_cosine_distance_scalar(&e1, &e2)
        );
        eprintln!(
            "distance (simd): {}",
            normalized_cosine_distance_simd(&e1, &e2)
        );
        eprintln!(
            "distance (simd same): {}",
            normalized_cosine_distance_simd(&e1, &e1)
        );

        let mut e3 = random_embedding(&mut rng);
        normalize_vec_scalar(&mut e3);
        eprintln!(
            "distance (scalar): {}",
            normalized_cosine_distance_scalar(&e1, &e3)
        );
        eprintln!(
            "distance (simd): {}",
            normalized_cosine_distance_simd(&e1, &e3)
        );

        for (x1, x2) in e1.iter().zip(e2.iter()) {
            assert_float_absolute_eq!(x1, x2)
        }
    }

    #[test]
    fn ensure_nonsimd_and_simd_dotproducts_are_similar() {
        let seed: u64 = 42;
        let mut rng = StdRng::seed_from_u64(seed);

        let mut e1 = random_normalized_embedding(&mut rng);
        let mut e2 = random_normalized_embedding(&mut rng);
        let d1 = normalized_cosine_distance_scalar(&e1, &e2);
        let d2 = normalized_cosine_distance_simd(&e1, &e2);

        assert_float_absolute_eq!(d1, d2);
    }
}
