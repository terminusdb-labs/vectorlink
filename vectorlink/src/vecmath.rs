#![allow(unused)]
use rand::Rng;

pub const EMBEDDING_LENGTH: usize = 1536;
pub const EMBEDDING_BYTE_LENGTH: usize = EMBEDDING_LENGTH * 4;
pub type Embedding = [f32; EMBEDDING_LENGTH];
pub type EmbeddingBytes = [u8; EMBEDDING_BYTE_LENGTH];

pub const QUANTIZED_EMBEDDING_LENGTH: usize = 48;
pub const QUANTIZED_EMBEDDING_BYTE_LENGTH: usize = QUANTIZED_EMBEDDING_LENGTH * 2;
pub type QuantizedEmbedding = [u16; QUANTIZED_EMBEDDING_LENGTH];
pub type QuantizedEmbeddingBytes = [u8; QUANTIZED_EMBEDDING_BYTE_LENGTH];

pub const CENTROID_32_LENGTH: usize = 32;
pub const CENTROID_32_BYTE_LENGTH: usize = CENTROID_32_LENGTH * 4;
pub type Centroid32 = [f32; CENTROID_32_LENGTH];
pub type CentroidBytes = [u8; CENTROID_32_BYTE_LENGTH];

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

pub fn euclidean_distance_32(v1: &Centroid32, v2: &Centroid32) -> f32 {
    simd::euclidean_distance_simd(v1, v2)
}

pub fn cosine_partial_distance_32(v1: &Centroid32, v2: &Centroid32) -> f32 {
    simd::cosine_partial_distance_32_simd(v1, v2)
}

pub fn normalize_vec(vec: &mut Embedding) {
    simd::normalize_vec_simd(vec)
}

pub mod simd {
    use super::*;
    use aligned_box::AlignedBox;
    use std::simd::{f32x16, num::SimdFloat, Simd};

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

    pub fn euclidean_distance_simd(left: &Centroid32, right: &Centroid32) -> f32 {
        euclidean_partial_distance_simd(left, right).sqrt()
    }

    pub fn euclidean_partial_distance_simd(left: &Centroid32, right: &Centroid32) -> f32 {
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
