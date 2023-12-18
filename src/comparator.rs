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
}
