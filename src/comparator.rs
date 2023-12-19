use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::{path::Path, sync::Arc};

use parallel_hnsw::{AbstractVector, Comparator, SerializationError};

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

impl Comparator<Embedding> for OpenAIComparator {
    type Params = Arc<VectorStore>;
    fn compare_vec(&self, v1: AbstractVector<Embedding>, v2: AbstractVector<Embedding>) -> f32 {
        #[allow(unused_assignments)]
        let mut v1_opt = None;
        #[allow(unused_assignments)]
        let mut v2_opt = None;
        let v1 = match v1 {
            AbstractVector::Stored(vid) => {
                let loaded = self.store.get_vec(&self.domain, vid.0).unwrap().unwrap();
                v1_opt = Some(loaded);
                &**v1_opt.as_ref().unwrap()
            }
            AbstractVector::Unstored(v) => v,
        };
        let v2 = match v2 {
            AbstractVector::Stored(vid) => {
                let loaded = self.store.get_vec(&self.domain, vid.0).unwrap().unwrap();
                v2_opt = Some(loaded);
                &**v2_opt.as_ref().unwrap()
            }
            AbstractVector::Unstored(v) => v,
        };
        normalized_cosine_distance(v1, v2)
    }

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
