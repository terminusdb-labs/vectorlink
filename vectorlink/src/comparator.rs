use itertools::{IntoChunks, Itertools};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::ops::Deref;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{RwLock, RwLockReadGuard};
use std::{path::Path, sync::Arc};

use parallel_hnsw::{pq, AbstractVector, Comparator, Serializable, SerializationError, VectorId};

use crate::vecmath::{
    self, Centroid32, QuantizedEmbedding, CENTROID_32_BYTE_LENGTH, QUANTIZED_EMBEDDING_LENGTH,
};
use crate::vectors::LoadedVec;
use crate::{
    vecmath::{normalized_cosine_distance, Embedding},
    vectors::{Domain, VectorStore},
};

#[derive(Clone)]
pub struct OpenAIComparator {
    pub domain: Arc<Domain>,
    pub store: Arc<VectorStore>,
}

#[derive(Serialize, Deserialize)]
pub struct ComparatorMeta {
    domain: String,
    size: usize,
}

impl Comparator for OpenAIComparator {
    type T = Embedding;
    type Borrowable<'a> = LoadedVec
        where Self: 'a;
    fn lookup(&self, v: VectorId) -> LoadedVec {
        self.store.get_vec(&self.domain, v.0).unwrap().unwrap()
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
        let domain = self.domain.name();
        // How do we get this value?
        let size = 2_000_000;
        let comparator = ComparatorMeta {
            domain: domain.to_string(),
            size,
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
        let ComparatorMeta { domain, size } = serde_json::from_str(&contents)?;
        let domain = store.get_domain(&domain)?;
        Ok(OpenAIComparator {
            domain,
            store: store.into(),
        })
    }
}

#[derive(Clone, Default)]
pub struct Centroid32Comparator {
    centroids: Arc<RwLock<Vec<Centroid32>>>,
}

impl Comparator for Centroid32Comparator {
    type T = Centroid32;

    type Borrowable<'a> = ReadLockedVec<'a, Centroid32>;

    fn lookup(&self, v: VectorId) -> Self::Borrowable<'_> {
        ReadLockedVec {
            lock: self.centroids.read().unwrap(),
            id: v,
        }
    }

    fn compare_raw(&self, v1: &Self::T, v2: &Self::T) -> f32 {
        vecmath::euclidean_distance_32(v1, v2)
    }
}

impl Serializable for Centroid32Comparator {
    type Params = ();

    fn serialize<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError> {
        let centroids = self.centroids.read().unwrap();
        let len = centroids.len();
        let buf: &[u8] = unsafe {
            std::slice::from_raw_parts(
                centroids.as_ptr() as *const u8,
                len * std::mem::size_of::<Centroid32>(),
            )
        };
        std::fs::write(path, buf)?;
        Ok(())
    }

    fn deserialize<P: AsRef<Path>>(
        path: P,
        _params: Self::Params,
    ) -> Result<Self, SerializationError> {
        let size = std::fs::metadata(&path)?.size() as usize;
        assert_eq!(0, size % CENTROID_32_BYTE_LENGTH);
        let number_of_centroids = size / CENTROID_32_BYTE_LENGTH;
        let mut vec = vec![Centroid32::default(); number_of_centroids];
        let mut file = std::fs::File::open(&path)?;
        let buf = unsafe { std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, size) };
        file.read_exact(buf)?;

        Ok(Self {
            centroids: Arc::new(RwLock::new(vec)),
        })
    }
}

impl parallel_hnsw::pq::VectorStore for Centroid32Comparator {
    type T = <Centroid32Comparator as Comparator>::T;

    fn store(&self, i: Box<dyn Iterator<Item = Self::T>>) -> Vec<VectorId> {
        let mut data = self.centroids.write().unwrap();
        let vid = data.len();
        let mut vectors: Vec<VectorId> = Vec::new();
        data.extend(i.enumerate().map(|(i, v)| {
            vectors.push(VectorId(vid + i));
            v
        }));
        vectors
    }
}

#[derive(Clone)]
pub struct QuantizedComparator {
    pub cc: Centroid32Comparator,
    pub data: Arc<RwLock<Vec<QuantizedEmbedding>>>,
}

pub struct ReadLockedVec<'a, T> {
    lock: RwLockReadGuard<'a, Vec<T>>,
    id: VectorId,
}

impl<'a, T> Deref for ReadLockedVec<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.lock[self.id.0]
    }
}

impl Comparator for QuantizedComparator {
    type T = QuantizedEmbedding;

    type Borrowable<'a> = ReadLockedVec<'a, Self::T>;

    fn lookup(&self, v: VectorId) -> Self::Borrowable<'_> {
        ReadLockedVec {
            lock: self.data.read().unwrap(),
            id: v,
        }
    }

    fn compare_raw(&self, v1: &Self::T, v2: &Self::T) -> f32 {
        let v_reconstruct1: Vec<f32> = v1
            .iter()
            .flat_map(|i| self.cc.lookup(VectorId(*i as usize)).into_iter())
            .collect();
        let v_reconstruct2: Vec<f32> = v2
            .iter()
            .flat_map(|i| self.cc.lookup(VectorId(*i as usize)).into_iter())
            .collect();
        let mut ar1 = [0.0_f32; 1536];
        let mut ar2 = [0.0_f32; 1536];
        ar1.copy_from_slice(&v_reconstruct1);
        ar2.copy_from_slice(&v_reconstruct2);
        normalized_cosine_distance(&ar1, &ar2)
    }
}

impl Serializable for QuantizedComparator {
    type Params = ();

    fn serialize<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError> {
        let path_buf: PathBuf = path.as_ref().into();
        let index_path = path_buf.join("index");
        self.cc.serialize(index_path)?;

        let vector_path = path_buf.join("vectors");
        let vec_lock = self.data.read().unwrap();
        let size = vec_lock.len() * std::mem::size_of::<QuantizedEmbedding>();
        let buf: &[u8] =
            unsafe { std::slice::from_raw_parts(vec_lock.as_ptr() as *const u8, size) };
        std::fs::write(vector_path, buf)?;
        Ok(())
    }

    fn deserialize<P: AsRef<Path>>(
        path: P,
        params: Self::Params,
    ) -> Result<Self, SerializationError> {
        let path_buf: PathBuf = path.as_ref().into();
        let index_path = path_buf.join("index");
        let cc = Centroid32Comparator::deserialize(index_path, ())?;

        let vector_path = path_buf.join("vectors");

        let size = std::fs::metadata(&path)?.size() as usize;
        assert_eq!(0, size % std::mem::size_of::<QuantizedEmbedding>());
        let number_of_quantized = size / std::mem::size_of::<QuantizedEmbedding>();
        let mut vec = vec![[0_u16; QUANTIZED_EMBEDDING_LENGTH]; number_of_quantized];
        let mut file = std::fs::File::open(&vector_path)?;
        let buf = unsafe { std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, size) };
        file.read_exact(buf)?;
        let data = Arc::new(RwLock::new(vec));
        Ok(Self { cc, data })
    }
}

impl pq::VectorStore for QuantizedComparator {
    type T = <QuantizedComparator as Comparator>::T;

    fn store(&self, i: Box<dyn Iterator<Item = Self::T>>) -> Vec<VectorId> {
        let mut data = self.data.write().unwrap();
        let vid = data.len();
        let mut vectors: Vec<VectorId> = Vec::new();
        data.extend(i.enumerate().map(|(i, v)| {
            vectors.push(VectorId(vid + i));
            v
        }));
        vectors
    }
}

impl pq::VectorSelector for OpenAIComparator {
    type T = Embedding;

    fn selection(&self, size: usize) -> Vec<Self::T> {
        self.store.get_random_vectors(&self.domain, size).unwrap()
    }

    fn vector_chunks(&self) -> impl Iterator<Item = Vec<Self::T>> {
        // low quality make better
        let iter = (0..self.domain.num_vecs())
            .map(|index| *self.store.get_vec(&self.domain, index).unwrap().unwrap());

        ChunkedVecIterator {
            iter,
            _x: PhantomData,
        }
    }
}

pub struct ChunkedVecIterator<T, I: Iterator<Item = T>> {
    iter: I,
    _x: PhantomData<T>,
}

impl<T, I: Iterator<Item = T>> Iterator for ChunkedVecIterator<T, I> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut chunk = Vec::with_capacity(1_000_000);
        while let Some(item) = self.iter.next() {
            chunk.push(item);
            if chunk.len() == 16_384 {
                break;
            }
        }

        if chunk.is_empty() {
            None
        } else {
            Some(chunk)
        }
    }
}
