use std::sync::Arc;

use parallel_hnsw::{AbstractVector, Comparator};

use crate::{
    vecmath::{normalized_cosine_distance, Embedding},
    vectors::{Domain, VectorStore},
};

#[derive(Clone)]
pub struct OpenAIComparator {
    pub domain: Arc<Domain>,
    pub store: Arc<VectorStore>,
}

impl Comparator<Embedding> for OpenAIComparator {
    fn compare_vec(&self, v1: AbstractVector<Embedding>, v2: AbstractVector<Embedding>) -> f32 {
        let v1 = match v1 {
            AbstractVector::Stored(vid) => {
                &*self.store.get_vec(&self.domain, vid.0).unwrap().unwrap()
            }
            AbstractVector::Unstored(v) => v,
        };
        let v2 = match v2 {
            AbstractVector::Stored(vid) => {
                &*self.store.get_vec(&self.domain, vid.0).unwrap().unwrap()
            }
            AbstractVector::Unstored(v) => v,
        };
        normalized_cosine_distance(v1, v2)
    }
}
