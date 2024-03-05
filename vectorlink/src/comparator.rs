use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::fs::MetadataExt;
use std::{path::Path, sync::Arc};

use parallel_hnsw::{AbstractVector, Comparator, Serializable, SerializationError, VectorId};

use crate::vecmath::{self, Centroid32, CENTROID_32_BYTE_LENGTH};
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

#[derive(Clone)]
pub struct Centroid32Comparator {
    centroids: Arc<Vec<Centroid32>>,
}

impl Comparator for Centroid32Comparator {
    type T = Centroid32;

    type Borrowable<'a> = &'a Centroid32;

    fn lookup(&self, v: VectorId) -> Self::Borrowable<'_> {
        &self.centroids[v.0]
    }

    fn compare_raw(&self, v1: &Self::T, v2: &Self::T) -> f32 {
        vecmath::normalized_cosine_distance_32(v1, v2)
    }
}

impl Serializable for Centroid32Comparator {
    type Params = ();

    fn serialize<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError> {
        let buf: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.centroids.as_ptr() as *const u8,
                self.centroids.len() * std::mem::size_of::<Centroid32>(),
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
            centroids: Arc::new(vec),
        })
    }
}
