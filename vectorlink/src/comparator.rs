use parallel_hnsw::pq::{
    CentroidComparatorConstructor, HnswQuantizer, PartialDistance, QuantizedComparatorConstructor,
};
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::{path::Path, sync::Arc};

use parallel_hnsw::{pq, Comparator, Serializable, SerializationError, VectorId};

use crate::store::{ImmutableVectorFile, LoadedVectorRange, VectorFile};
use crate::vecmath::{
    self, EuclideanDistance16, EuclideanDistance32, Quantized16Embedding, Quantized32Embedding,
    CENTROID_16_LENGTH, CENTROID_32_LENGTH, EMBEDDING_LENGTH, QUANTIZED_16_EMBEDDING_LENGTH,
    QUANTIZED_32_EMBEDDING_LENGTH,
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
        store: Arc<VectorStore>,
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

impl pq::VectorSelector for DiskOpenAIComparator {
    type T = Embedding;

    fn selection(&self, size: usize) -> Vec<Self::T> {
        // TODO do something else for sizes close to number of vecs
        if size >= self.vectors.num_vecs() {
            return self.vectors.all_vectors().unwrap().clone().into_vec();
        }
        let mut rng = thread_rng();
        let mut set = HashSet::new();
        let range = Uniform::from(0_usize..self.vectors.num_vecs());
        while set.len() != size {
            let candidate = rng.sample(&range);
            set.insert(candidate);
        }

        set.into_iter()
            .map(|index| self.vectors.vec(index).unwrap())
            .collect()
    }

    fn vector_chunks(&self) -> impl Iterator<Item = Vec<Self::T>> {
        self.vectors
            .vector_chunks(1_000_000)
            .unwrap()
            .map(|x| x.unwrap())
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
        store: Arc<VectorStore>,
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

impl<const SIZE: usize, C: DistanceCalculator<T = [f32; SIZE]> + Default>
    CentroidComparatorConstructor for ArrayCentroidComparator<SIZE, C>
{
    fn new(centroids: Vec<Self::T>) -> Self {
        let len = centroids.len();
        Self {
            distances: Arc::new(MemoizedPartialDistances::new(C::default(), &centroids)),
            centroids: Arc::new(LoadedVectorRange::new(centroids, 0..len)),
            calculator: PhantomData,
        }
    }
}

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
        _params: Self::Params,
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

#[derive(Clone)]
pub struct Quantized32Comparator {
    pub cc: Centroid32Comparator,
    pub data: Arc<LoadedVectorRange<Quantized32Embedding>>,
}

impl QuantizedComparatorConstructor for Quantized32Comparator {
    type CentroidComparator = Centroid32Comparator;

    fn new(cc: &Self::CentroidComparator) -> Self {
        Self {
            cc: cc.clone(),
            data: Default::default(),
        }
    }
}

#[derive(Clone)]
pub struct Quantized16Comparator {
    pub cc: Centroid16Comparator,
    pub data: Arc<LoadedVectorRange<Quantized16Embedding>>,
}

impl QuantizedComparatorConstructor for Quantized16Comparator {
    type CentroidComparator = Centroid16Comparator;

    fn new(cc: &Self::CentroidComparator) -> Self {
        Self {
            cc: cc.clone(),
            data: Default::default(),
        }
    }
}

impl PartialDistance for Quantized32Comparator {
    fn partial_distance(&self, i: u16, j: u16) -> f32 {
        self.cc.partial_distance(i, j)
    }
}

impl PartialDistance for Quantized16Comparator {
    fn partial_distance(&self, i: u16, j: u16) -> f32 {
        self.cc.partial_distance(i, j)
    }
}

impl Comparator for Quantized32Comparator
where
    Quantized32Comparator: PartialDistance,
{
    type T = Quantized32Embedding;

    type Borrowable<'a> = &'a Quantized32Embedding;

    fn lookup(&self, v: VectorId) -> Self::Borrowable<'_> {
        &self.data[v.0]
    }

    fn compare_raw(&self, v1: &Self::T, v2: &Self::T) -> f32 {
        let mut partial_distances = [0.0_f32; QUANTIZED_32_EMBEDDING_LENGTH];
        for ix in 0..QUANTIZED_32_EMBEDDING_LENGTH {
            let partial_1 = v1[ix];
            let partial_2 = v2[ix];
            let partial_distance = self.cc.partial_distance(partial_1, partial_2);
            partial_distances[ix] = partial_distance;
        }

        vecmath::sum_48(&partial_distances).sqrt()
    }
}

impl Serializable for Quantized32Comparator {
    type Params = ();

    fn serialize<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError> {
        let path_buf: PathBuf = path.as_ref().into();
        std::fs::create_dir_all(&path_buf)?;

        let index_path = path_buf.join("index");
        self.cc.serialize(index_path)?;

        let vector_path = path_buf.join("vectors");
        let mut vector_file = VectorFile::open(vector_path, true)?;
        vector_file.append_vector_range(self.data.vecs())?;
        Ok(())
    }

    fn deserialize<P: AsRef<Path>>(
        path: P,
        _params: Self::Params,
    ) -> Result<Self, SerializationError> {
        let path_buf: PathBuf = path.as_ref().into();
        let index_path = path_buf.join("index");
        let cc = Centroid32Comparator::deserialize(index_path, ())?;

        let vector_path = path_buf.join("vectors");
        let vector_file = VectorFile::open(vector_path, true)?;
        let range = vector_file.all_vectors()?;

        let data = Arc::new(range);
        Ok(Self { cc, data })
    }
}

impl pq::VectorStore for Quantized32Comparator {
    type T = <Quantized32Comparator as Comparator>::T;

    fn store(&mut self, i: Box<dyn Iterator<Item = Self::T>>) -> Vec<VectorId> {
        // this is p retty stupid, but then, these comparators should not be storing in the first place
        let mut new_contents: Vec<Self::T> = Vec::with_capacity(self.data.len() + i.size_hint().0);
        new_contents.extend(self.data.vecs().iter());
        let vid = self.data.len();
        let mut vectors: Vec<VectorId> = Vec::new();
        new_contents.extend(i.enumerate().map(|(i, v)| {
            vectors.push(VectorId(vid + i));
            v
        }));
        let end = new_contents.len();

        let data = LoadedVectorRange::new(new_contents, 0..end);
        self.data = Arc::new(data);

        vectors
    }
}

impl pq::VectorSelector for OpenAIComparator {
    type T = Embedding;

    fn selection(&self, size: usize) -> Vec<Self::T> {
        // TODO do something else for sizes close to number of vecs
        let mut rng = thread_rng();
        let mut set = HashSet::new();
        let range = Uniform::from(0_usize..size);
        while set.len() != size {
            let candidate = rng.sample(&range);
            set.insert(candidate);
        }

        set.into_iter()
            .map(|index| *self.range.vec(index))
            .collect()
    }

    fn vector_chunks(&self) -> impl Iterator<Item = Vec<Self::T>> {
        // low quality make better
        self.range.vecs().chunks(1_000_000).map(|c| c.to_vec())
    }
}

impl Comparator for Quantized16Comparator
where
    Quantized16Comparator: PartialDistance,
{
    type T = Quantized16Embedding;

    type Borrowable<'a> = &'a Self::T;

    fn lookup(&self, v: VectorId) -> Self::Borrowable<'_> {
        self.data.vec(v.0)
    }

    fn compare_raw(&self, v1: &Self::T, v2: &Self::T) -> f32 {
        let mut partial_distances = [0.0_f32; QUANTIZED_16_EMBEDDING_LENGTH];
        for ix in 0..QUANTIZED_16_EMBEDDING_LENGTH {
            let partial_1 = v1[ix];
            let partial_2 = v2[ix];
            let partial_distance = self.cc.partial_distance(partial_1, partial_2);
            partial_distances[ix] = partial_distance;
        }

        vecmath::sum_96(&partial_distances).sqrt()
    }
}

impl Serializable for Quantized16Comparator {
    type Params = ();

    fn serialize<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError> {
        let path_buf: PathBuf = path.as_ref().into();
        std::fs::create_dir_all(&path_buf)?;

        let index_path = path_buf.join("index");
        self.cc.serialize(index_path)?;

        let vector_path = path_buf.join("vectors");
        let mut vector_file = VectorFile::create(vector_path, true)?;
        vector_file.append_vector_range(self.data.vecs())?;
        Ok(())
    }

    fn deserialize<P: AsRef<Path>>(
        path: P,
        _params: Self::Params,
    ) -> Result<Self, SerializationError> {
        let path_buf: PathBuf = path.as_ref().into();
        let index_path = path_buf.join("index");
        let cc = Centroid16Comparator::deserialize(index_path, ())?;

        let vector_path = path_buf.join("vectors");
        let vector_file = VectorFile::open(vector_path, true)?;
        let range = vector_file.all_vectors()?;

        let data = Arc::new(range);
        Ok(Self { cc, data })
    }
}

impl pq::VectorStore for Quantized16Comparator {
    type T = <Quantized16Comparator as Comparator>::T;

    fn store(&mut self, i: Box<dyn Iterator<Item = Self::T>>) -> Vec<VectorId> {
        // this is p retty stupid, but then, these comparators should not be storing in the first place
        let mut new_contents: Vec<Self::T> = Vec::with_capacity(self.data.len() + i.size_hint().0);
        new_contents.extend(self.data.vecs().iter());
        let vid = self.data.len();
        let mut vectors: Vec<VectorId> = Vec::new();
        new_contents.extend(i.enumerate().map(|(i, v)| {
            vectors.push(VectorId(vid + i));
            v
        }));

        let end = new_contents.len();

        let data = LoadedVectorRange::new(new_contents, 0..end);
        self.data = Arc::new(data);

        vectors
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
