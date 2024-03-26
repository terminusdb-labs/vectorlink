use parallel_hnsw::pq::{HnswQuantizer, PartialDistance, Quantizer};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{self, BufReader, Read, Write};
use std::marker::PhantomData;
use std::{path::Path, sync::Arc};

use parallel_hnsw::{Comparator, Serializable, SerializationError, VectorId};

use crate::domain::PqDerivedDomainInfo;
use crate::store::{ImmutableVectorFile, LoadedVectorRange, VectorFile};
use crate::vecmath::{
    self, EuclideanDistance16, EuclideanDistance32, CENTROID_16_LENGTH, CENTROID_32_LENGTH,
    EMBEDDING_LENGTH, QUANTIZED_16_EMBEDDING_LENGTH, QUANTIZED_32_EMBEDDING_LENGTH,
};
use crate::{
    vecmath::{normalized_cosine_distance, Embedding},
    vectors::VectorStore,
};

#[derive(Clone)]
pub struct DiskOpenAIComparator {
    domain: String,
    vectors: Arc<ImmutableVectorFile<Embedding>>,
}

impl DiskOpenAIComparator {
    pub fn new(domain: String, vectors: Arc<ImmutableVectorFile<Embedding>>) -> Self {
        Self { domain, vectors }
    }
}

impl Comparator for DiskOpenAIComparator {
    type T = Embedding;
    type Borrowable<'a> = Box<Embedding>
        where Self: 'a;
    fn lookup(&self, v: VectorId) -> Box<Embedding> {
        Box::new(self.vectors.vec(v.0).unwrap())
    }

    fn compare_raw(&self, v1: &Embedding, v2: &Embedding) -> f32 {
        normalized_cosine_distance(v1, v2)
    }
}

impl Serializable for DiskOpenAIComparator {
    type Params = Arc<VectorStore>;
    fn serialize<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError> {
        let mut comparator_file: std::fs::File = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(path)?;
        eprintln!("opened comparator serialize file");
        // How do we get this value?
        let comparator = ComparatorMeta {
            domain_name: self.domain.clone(),
            size: self.vectors.num_vecs(),
        };
        let comparator_meta = serde_json::to_string(&comparator)?;
        eprintln!("serialized comparator");
        comparator_file.write_all(&comparator_meta.into_bytes())?;
        eprintln!("wrote comparator to file");
        Ok(())
    }

    fn deserialize<P: AsRef<Path>>(
        path: P,
        store: &Arc<VectorStore>,
    ) -> Result<Self, SerializationError> {
        let mut comparator_file = OpenOptions::new().read(true).open(path)?;
        let mut contents = String::new();
        comparator_file.read_to_string(&mut contents)?;
        let ComparatorMeta { domain_name, .. } = serde_json::from_str(&contents)?;
        let domain = store.get_domain(&domain_name)?;
        Ok(DiskOpenAIComparator {
            domain: domain.name().to_owned(),
            vectors: Arc::new(domain.immutable_file()),
        })
    }
}

#[derive(Clone)]
pub struct OpenAIComparator {
    domain_name: String,
    range: Arc<LoadedVectorRange<Embedding>>,
}

impl OpenAIComparator {
    pub fn new(domain_name: String, range: Arc<LoadedVectorRange<Embedding>>) -> Self {
        Self { domain_name, range }
    }
}

#[derive(Serialize, Deserialize)]
pub struct ComparatorMeta {
    domain_name: String,
    size: usize,
}

impl Comparator for OpenAIComparator {
    type T = Embedding;
    type Borrowable<'a> = &'a Embedding
        where Self: 'a;
    fn lookup(&self, v: VectorId) -> &Embedding {
        self.range.vec(v.0)
    }

    fn compare_raw(&self, v1: &Embedding, v2: &Embedding) -> f32 {
        normalized_cosine_distance(v1, v2)
    }
}

impl Serializable for OpenAIComparator {
    type Params = Arc<VectorStore>;
    fn serialize<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError> {
        let mut comparator_file: std::fs::File =
            OpenOptions::new().write(true).create(true).open(path)?;
        eprintln!("opened comparator serialize file");
        // How do we get this value?
        let comparator = ComparatorMeta {
            domain_name: self.domain_name.clone(),
            size: self.range.len(),
        };
        let comparator_meta = serde_json::to_string(&comparator)?;
        eprintln!("serialized comparator");
        comparator_file.write_all(&comparator_meta.into_bytes())?;
        eprintln!("wrote comparator to file");
        Ok(())
    }

    fn deserialize<P: AsRef<Path>>(
        path: P,
        store: &Arc<VectorStore>,
    ) -> Result<Self, SerializationError> {
        let mut comparator_file = OpenOptions::new().read(true).open(path)?;
        let mut contents = String::new();
        comparator_file.read_to_string(&mut contents)?;
        let ComparatorMeta { domain_name, .. } = serde_json::from_str(&contents)?;
        let domain = store.get_domain(&domain_name)?;
        Ok(OpenAIComparator {
            domain_name,
            range: Arc::new(domain.all_vecs()?),
        })
    }
}

struct MemoizedPartialDistances {
    partial_distances: Vec<f32>,
    size: usize,
}

pub trait DistanceCalculator {
    type T;
    fn partial_distance(&self, left: &Self::T, right: &Self::T) -> f32;
    fn finalize_partial_distance(&self, distance: f32) -> f32;
    fn aggregate_partial_distances(&self, distances: &[f32]) -> f32;

    fn distance(&self, left: &Self::T, right: &Self::T) -> f32 {
        self.finalize_partial_distance(self.partial_distance(left, right))
    }
}

impl MemoizedPartialDistances {
    fn new<T, P: DistanceCalculator<T = T>>(partial_distance_calculator: P, vectors: &[T]) -> Self {
        let size = vectors.len();
        let mut partial_distances: Vec<f32> = vec![0.0; size * size];
        for c in 0..size * size {
            let i = c / size;
            let j = c % size;
            partial_distances[c] =
                partial_distance_calculator.partial_distance(&vectors[i], &vectors[j]);
            //vecmath::euclidean_partial_distance_32(&vectors[i], &vectors[j]);
        }

        Self {
            partial_distances,
            size,
        }
    }

    #[allow(dead_code)]
    fn all_distances(&self) -> &[f32] {
        &self.partial_distances
    }

    fn partial_distance(&self, i: u16, j: u16) -> f32 {
        self.partial_distances[(i * self.size as u16 + j) as usize]
    }
}

pub struct ArrayCentroidComparator<const N: usize, C> {
    distances: Arc<MemoizedPartialDistances>,
    centroids: Arc<LoadedVectorRange<[f32; N]>>,
    calculator: PhantomData<C>,
}

impl<const N: usize, C: DistanceCalculator<T = [f32; N]> + Default> ArrayCentroidComparator<N, C> {
    pub fn new(centroids: Vec<[f32; N]>) -> Self {
        let len = centroids.len();
        Self {
            distances: Arc::new(MemoizedPartialDistances::new(C::default(), &centroids)),
            centroids: Arc::new(LoadedVectorRange::new(centroids, 0..len)),
            calculator: PhantomData,
        }
    }
}

impl<const N: usize, C> Clone for ArrayCentroidComparator<N, C> {
    fn clone(&self) -> Self {
        Self {
            distances: self.distances.clone(),
            centroids: self.centroids.clone(),
            calculator: PhantomData,
        }
    }
}
unsafe impl<const N: usize, C> Sync for ArrayCentroidComparator<N, C> {}

pub type Centroid16Comparator = ArrayCentroidComparator<CENTROID_16_LENGTH, EuclideanDistance16>;
pub type Centroid32Comparator = ArrayCentroidComparator<CENTROID_32_LENGTH, EuclideanDistance32>;

impl<const SIZE: usize, C: DistanceCalculator<T = [f32; SIZE]> + Default> Comparator
    for ArrayCentroidComparator<SIZE, C>
{
    type T = [f32; SIZE];

    type Borrowable<'a> = &'a Self::T where C: 'a;

    fn lookup(&self, v: VectorId) -> Self::Borrowable<'_> {
        &self.centroids[v.0]
    }

    fn compare_raw(&self, v1: &Self::T, v2: &Self::T) -> f32 {
        let calculator = C::default();
        calculator.distance(v1, v2)
    }
}

impl<const N: usize, C> PartialDistance for ArrayCentroidComparator<N, C> {
    fn partial_distance(&self, i: u16, j: u16) -> f32 {
        self.distances.partial_distance(i, j)
    }
}

impl<const N: usize, C: DistanceCalculator<T = [f32; N]> + Default> Serializable
    for ArrayCentroidComparator<N, C>
{
    type Params = ();

    fn serialize<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError> {
        let mut vector_file: VectorFile<[f32; N]> = VectorFile::create(path, true)?;
        vector_file.append_vector_range(self.centroids.vecs())?;

        Ok(())
    }

    fn deserialize<P: AsRef<Path>>(
        path: P,
        _params: &Self::Params,
    ) -> Result<Self, SerializationError> {
        let vector_file: VectorFile<[f32; N]> = VectorFile::open(path, true)?;
        let centroids = Arc::new(vector_file.all_vectors()?);

        Ok(Self {
            distances: Arc::new(MemoizedPartialDistances::new(
                C::default(),
                centroids.vecs(),
            )),
            centroids,
            calculator: PhantomData,
        })
    }
}

pub struct QuantizedDomainComparator<
    const SIZE: usize,
    const CENTROID_SIZE: usize,
    const QUANTIZED_SIZE: usize,
    C,
> {
    domain: String,
    subdomain: String,
    cc: ArrayCentroidComparator<CENTROID_SIZE, C>,
    data: Arc<LoadedVectorRange<[u16; QUANTIZED_SIZE]>>,
}

impl<const SIZE: usize, const CENTROID_SIZE: usize, const QUANTIZED_SIZE: usize, C> Clone
    for QuantizedDomainComparator<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, C>
{
    fn clone(&self) -> Self {
        Self {
            domain: self.domain.clone(),
            subdomain: self.subdomain.clone(),
            cc: self.cc.clone(),
            data: self.data.clone(),
        }
    }
}

pub type Quantized16Comparator = QuantizedDomainComparator<
    EMBEDDING_LENGTH,
    CENTROID_16_LENGTH,
    QUANTIZED_16_EMBEDDING_LENGTH,
    EuclideanDistance16,
>;
pub type Quantized32Comparator = QuantizedDomainComparator<
    EMBEDDING_LENGTH,
    CENTROID_32_LENGTH,
    QUANTIZED_32_EMBEDDING_LENGTH,
    EuclideanDistance32,
>;

#[derive(Serialize, Deserialize)]
struct QuantizedDomainComparatorMeta {
    domain: String,
    subdomain: String,
}

impl<
        const SIZE: usize,
        const CENTROID_SIZE: usize,
        const QUANTIZED_SIZE: usize,
        C: 'static + DistanceCalculator<T = [f32; CENTROID_SIZE]>,
    > QuantizedDomainComparator<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, C>
where
    ArrayCentroidComparator<CENTROID_SIZE, C>: 'static + Comparator<T = [f32; CENTROID_SIZE]>,
{
    pub fn load(store: &VectorStore, domain: String, subdomain: String) -> io::Result<Self> {
        assert_eq!(SIZE, CENTROID_SIZE * QUANTIZED_SIZE); // TODO compile-time macro check this
        let domain_info = store.get_domain(&domain)?;
        let derived_domain_info: PqDerivedDomainInfo<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, C> =
            domain_info
                .get_derived_domain_info(&subdomain)
                .expect("pq subdomain not found");

        Ok(Self {
            domain,
            subdomain,
            cc: derived_domain_info.quantizer.quantizer.comparator().clone(),
            data: Arc::new(derived_domain_info.file.all_vectors()?),
        })
    }
}
impl<
        const SIZE: usize,
        const CENTROID_SIZE: usize,
        const QUANTIZED_SIZE: usize,
        C: 'static + DistanceCalculator<T = [f32; CENTROID_SIZE]>,
    > PartialDistance for QuantizedDomainComparator<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, C>
where
    ArrayCentroidComparator<CENTROID_SIZE, C>: 'static + Comparator<T = [f32; CENTROID_SIZE]>,
{
    fn partial_distance(&self, i: u16, j: u16) -> f32 {
        self.cc.partial_distance(i, j)
    }
}

impl<
        const SIZE: usize,
        const CENTROID_SIZE: usize,
        const QUANTIZED_SIZE: usize,
        C: 'static + DistanceCalculator<T = [f32; CENTROID_SIZE]>,
    > Serializable for QuantizedDomainComparator<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, C>
where
    ArrayCentroidComparator<CENTROID_SIZE, C>: 'static + Comparator<T = [f32; CENTROID_SIZE]>,
{
    type Params = Arc<VectorStore>;

    fn serialize<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError> {
        let meta = QuantizedDomainComparatorMeta {
            domain: self.domain.clone(),
            subdomain: self.subdomain.clone(),
        };
        let meta_string = serde_json::to_string(&meta)?;
        std::fs::write(path, meta_string)?;

        Ok(())
    }

    fn deserialize<P: AsRef<Path>>(
        path: P,
        params: &Self::Params,
    ) -> Result<Self, SerializationError> {
        let comparator_file = OpenOptions::new().read(true).open(path)?;
        let QuantizedDomainComparatorMeta { domain, subdomain } =
            serde_json::from_reader(BufReader::new(comparator_file))?;

        Ok(Self::load(&params, domain, subdomain)?)
    }
}

pub type QuantizedDomainComparator16 = QuantizedDomainComparator<
    EMBEDDING_LENGTH,
    CENTROID_16_LENGTH,
    QUANTIZED_16_EMBEDDING_LENGTH,
    Centroid16Comparator,
>;
pub type QuantizedDomainComparator32 = QuantizedDomainComparator<
    EMBEDDING_LENGTH,
    CENTROID_32_LENGTH,
    QUANTIZED_32_EMBEDDING_LENGTH,
    Centroid32Comparator,
>;

pub struct QuantizedEmbeddingSizeCombination<
    const CENTROID_SIZE: usize,
    const QUANTIZED_SIZE: usize,
>;
pub trait ImplementedQuantizedEmbeddingSizeCombination<
    const CENTROID_SIZE: usize,
    const QUANTIZED_SIZE: usize,
>
{
    fn compare_quantized<C: PartialDistance>(
        comparator: &C,
        v1: &[u16; QUANTIZED_SIZE],
        v2: &[u16; QUANTIZED_SIZE],
    ) -> f32;
}

impl ImplementedQuantizedEmbeddingSizeCombination<CENTROID_16_LENGTH, QUANTIZED_16_EMBEDDING_LENGTH>
    for QuantizedEmbeddingSizeCombination<CENTROID_16_LENGTH, QUANTIZED_16_EMBEDDING_LENGTH>
{
    fn compare_quantized<C: PartialDistance>(
        comparator: &C,
        v1: &[u16; QUANTIZED_16_EMBEDDING_LENGTH],
        v2: &[u16; QUANTIZED_16_EMBEDDING_LENGTH],
    ) -> f32 {
        let mut partial_distances = [0.0_f32; QUANTIZED_16_EMBEDDING_LENGTH];
        for ix in 0..QUANTIZED_16_EMBEDDING_LENGTH {
            let partial_1 = v1[ix];
            let partial_2 = v2[ix];
            let partial_distance = comparator.partial_distance(partial_1, partial_2);
            partial_distances[ix] = partial_distance;
        }

        vecmath::sum_96(&partial_distances).sqrt()
    }
}

impl ImplementedQuantizedEmbeddingSizeCombination<CENTROID_32_LENGTH, QUANTIZED_32_EMBEDDING_LENGTH>
    for QuantizedEmbeddingSizeCombination<CENTROID_32_LENGTH, QUANTIZED_32_EMBEDDING_LENGTH>
{
    fn compare_quantized<C: PartialDistance>(
        comparator: &C,
        v1: &[u16; QUANTIZED_32_EMBEDDING_LENGTH],
        v2: &[u16; QUANTIZED_32_EMBEDDING_LENGTH],
    ) -> f32 {
        let mut partial_distances = [0.0_f32; QUANTIZED_32_EMBEDDING_LENGTH];
        for ix in 0..QUANTIZED_32_EMBEDDING_LENGTH {
            let partial_1 = v1[ix];
            let partial_2 = v2[ix];
            let partial_distance = comparator.partial_distance(partial_1, partial_2);
            partial_distances[ix] = partial_distance;
        }

        vecmath::sum_48(&partial_distances).sqrt()
    }
}

impl<const SIZE: usize, const CENTROID_SIZE: usize, const QUANTIZED_SIZE: usize, C: 'static>
    Comparator for QuantizedDomainComparator<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, C>
where
    QuantizedEmbeddingSizeCombination<CENTROID_SIZE, QUANTIZED_SIZE>:
        ImplementedQuantizedEmbeddingSizeCombination<CENTROID_SIZE, QUANTIZED_SIZE>,
{
    type T = [u16; QUANTIZED_SIZE];

    type Borrowable<'a> = &'a [u16; QUANTIZED_SIZE];

    fn lookup(&self, v: VectorId) -> Self::Borrowable<'_> {
        &self.data[v.0]
    }

    fn compare_raw(&self, v1: &Self::T, v2: &Self::T) -> f32 {
        QuantizedEmbeddingSizeCombination::<CENTROID_SIZE, QUANTIZED_SIZE>::compare_quantized(
            &self.cc, v1, v2,
        )
    }
}

pub type HnswQuantizer16 = HnswQuantizer<
    EMBEDDING_LENGTH,
    CENTROID_16_LENGTH,
    QUANTIZED_16_EMBEDDING_LENGTH,
    Centroid16Comparator,
>;
pub type HnswQuantizer32 = HnswQuantizer<
    EMBEDDING_LENGTH,
    CENTROID_32_LENGTH,
    QUANTIZED_32_EMBEDDING_LENGTH,
    Centroid32Comparator,
>;

pub struct DomainQuantizer<
    const SIZE: usize,
    const CENTROID_SIZE: usize,
    const QUANTIZED_SIZE: usize,
    C,
> {
    domain: String,
    derived_domain: String,
    quantizer: Arc<
        HnswQuantizer<
            SIZE,
            CENTROID_SIZE,
            QUANTIZED_SIZE,
            ArrayCentroidComparator<CENTROID_SIZE, C>,
        >,
    >,
}

impl<const SIZE: usize, const CENTROID_SIZE: usize, const QUANTIZED_SIZE: usize, C: 'static> Clone
    for DomainQuantizer<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, C>
{
    fn clone(&self) -> Self {
        Self {
            domain: self.domain.clone(),
            derived_domain: self.derived_domain.clone(),
            quantizer: self.quantizer.clone(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct DomainQuantizerMeta {
    domain: String,
    derived_domain: String,
}

impl<const SIZE: usize, const CENTROID_SIZE: usize, const QUANTIZED_SIZE: usize, C: 'static>
    Quantizer<SIZE, QUANTIZED_SIZE> for DomainQuantizer<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, C>
where
    ArrayCentroidComparator<CENTROID_SIZE, C>: Comparator<T = [f32; CENTROID_SIZE]>,
{
    fn quantize(&self, vec: &[f32; SIZE]) -> [u16; QUANTIZED_SIZE] {
        self.quantizer.quantize(vec)
    }

    fn reconstruct(&self, qvec: &[u16; QUANTIZED_SIZE]) -> [f32; SIZE] {
        self.quantizer.reconstruct(qvec)
    }
}

impl<const SIZE: usize, const CENTROID_SIZE: usize, const QUANTIZED_SIZE: usize, C: 'static>
    Serializable for DomainQuantizer<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, C>
{
    type Params = Arc<VectorStore>;

    fn serialize<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError> {
        let meta = DomainQuantizerMeta {
            domain: self.domain.clone(),
            derived_domain: self.derived_domain.clone(),
        };
        let data = serde_json::to_string(&meta)?;
        std::fs::write(path, data)?;

        Ok(())
    }

    fn deserialize<P: AsRef<Path>>(
        path: P,
        params: &Self::Params,
    ) -> Result<Self, SerializationError> {
        let DomainQuantizerMeta {
            domain,
            derived_domain,
        } = serde_json::from_reader(BufReader::new(std::fs::File::open(path)?))?;

        let d = params.get_domain(&domain).expect("domain not found");
        let dd: PqDerivedDomainInfo<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, C> = d
            .get_derived_domain_info(&derived_domain)
            .expect("derived domain not found");

        Ok(dd.quantizer.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use parallel_hnsw::pq::CentroidComparatorConstructor;
    use parallel_hnsw::AbstractVector;

    use crate::comparator::Centroid32Comparator;
    use crate::comparator::Comparator;
    use crate::comparator::MemoizedPartialDistances;
    #[test]
    fn centroid32test() {
        /*
        let vectors = (0..1000)
            .map(|_| {
                let range = Uniform::from(0.0..1.0);
                let v: Vec<f32> = prng.sample_iter(&range).take(CENTROID_32_LENGTH).collect();
                v
            })
            .collect();
         */
        let cc = Centroid32Comparator::new(Vec::new());
        let mut v1 = [0.0_f32; 32];
        v1[0] = 1.0;
        v1[1] = 1.0;
        let mut v2 = [0.0_f32; 32];
        v2[30] = 1.0;
        v2[31] = 1.0;
        let res = cc.compare_vec(AbstractVector::Unstored(&v1), AbstractVector::Unstored(&v2));
        assert_eq!(res, 2.0);
    }
}
