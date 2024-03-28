use rayon::prelude::*;

use parallel_hnsw::{
    pq::{PartialDistance, QuantizedHnsw, Quantizer, VectorSelector, VectorStore},
    Comparator,
};

use crate::comparator::QuantizedData;

pub struct QuantizationStatistics {
    pub sample_avg: f32,
    pub sample_var: f32,
    pub sample_deviation: f32,
}

pub fn test_quantization<
    const SIZE: usize,
    const CENTROID_SIZE: usize,
    const QUANTIZED_SIZE: usize,
    CentroidComparator: 'static + Comparator<T = [f32; CENTROID_SIZE]>,
    QuantizedComparator: Comparator<T = [u16; QUANTIZED_SIZE]>
        + VectorStore<T = [u16; QUANTIZED_SIZE]>
        + PartialDistance
        + QuantizedData<Quantized = [u16; QUANTIZED_SIZE]>
        + 'static,
    FullComparator: Comparator<T = [f32; SIZE]> + VectorSelector<T = [f32; SIZE]> + 'static,
>(
    hnsw: &QuantizedHnsw<
        SIZE,
        CENTROID_SIZE,
        QUANTIZED_SIZE,
        CentroidComparator,
        QuantizedComparator,
        FullComparator,
    >,
) -> QuantizationStatistics {
    let c = hnsw.quantized_comparator();
    let quantized_vecs = c.data().vecs();
    let mut cursor: &[[u16; QUANTIZED_SIZE]] = &quantized_vecs;
    let quantizer = hnsw.quantizer();
    // sample_avg = sum(errors)/|errors|
    // sample_var = sum((error - sample_avg)^2)/|errors|

    let fc = hnsw.full_comparator();

    let errors = vec![0.0_f32; hnsw.vector_count()];

    eprintln!("starting processing of vector chunks");
    let mut offset = 0;
    for chunk in fc.vector_chunks() {
        let len = chunk.len();
        let quantized_chunk = &cursor[..len];
        cursor = &cursor[len..];

        chunk
            .into_par_iter()
            .zip(quantized_chunk.into_par_iter())
            .map(|(full_vec, quantized_vec)| {
                let reconstructed = quantizer.reconstruct(quantized_vec);

                fc.compare_raw(&full_vec, &reconstructed)
            })
            .enumerate()
            .for_each(|(ix, distance)| unsafe {
                let ptr = errors.as_ptr().add(offset + ix) as *mut f32;
                *ptr = distance;
            });

        offset += len;
    }

    let sample_avg: f32 = errors.iter().sum::<f32>() / errors.len() as f32;
    let sample_var = errors
        .iter()
        .map(|e| (e - sample_avg))
        .map(|x| x * x)
        .sum::<f32>()
        / (errors.len() - 1) as f32;
    let sample_deviation = sample_var.sqrt();

    QuantizationStatistics {
        sample_avg,
        sample_var,
        sample_deviation,
    }
}
