use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::{path::Path, sync::Arc};

use parallel_hnsw::{AbstractVector, Comparator, Serializable, SerializationError, VectorId};

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
