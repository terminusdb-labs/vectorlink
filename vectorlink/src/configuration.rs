use std::{fs::OpenOptions, path::PathBuf, sync::Arc};

use itertools::Either;
use parallel_hnsw::{
    pq::{HnswQuantizer, QuantizedHnsw},
    AbstractVector, Hnsw, Serializable, VectorId,
};
use rayon::iter::IndexedParallelIterator;
use serde::{Deserialize, Serialize};

use crate::{
    comparator::{
        Centroid16Comparator, Centroid32Comparator, DiskOpenAIComparator, DomainQuantizer,
        OpenAIComparator, Quantized16Comparator, Quantized32Comparator,
    },
    openai::Model,
    vecmath::{
        Embedding, EuclideanDistance16, EuclideanDistance32, CENTROID_16_LENGTH,
        CENTROID_32_LENGTH, EMBEDDING_LENGTH, QUANTIZED_16_EMBEDDING_LENGTH,
        QUANTIZED_32_EMBEDDING_LENGTH,
    },
    vectors::VectorStore,
};

pub type OpenAIHnsw = Hnsw<OpenAIComparator>;

#[derive(Serialize, Deserialize)]
pub enum HnswConfigurationType {
    QuantizedOpenAi,
    SmallQuantizedOpenAi,
    UnquantizedOpenAi,
}

#[derive(Serialize, Deserialize)]
pub struct HnswConfigurationState {
    version: usize,
    #[serde(rename = "type")]
    typ: HnswConfigurationType,
    model: Model,
}

pub enum HnswConfiguration {
    QuantizedOpenAi(
        Model,
        QuantizedHnsw<
            EMBEDDING_LENGTH,
            CENTROID_32_LENGTH,
            QUANTIZED_32_EMBEDDING_LENGTH,
            Quantized32Comparator,
            DiskOpenAIComparator,
            DomainQuantizer<
                EMBEDDING_LENGTH,
                CENTROID_32_LENGTH,
                QUANTIZED_32_EMBEDDING_LENGTH,
                EuclideanDistance32,
            >,
        >,
    ),
    SmallQuantizedOpenAi(
        Model,
        QuantizedHnsw<
            EMBEDDING_LENGTH,
            CENTROID_16_LENGTH,
            QUANTIZED_16_EMBEDDING_LENGTH,
            Quantized16Comparator,
            DiskOpenAIComparator,
            DomainQuantizer<
                EMBEDDING_LENGTH,
                CENTROID_16_LENGTH,
                QUANTIZED_16_EMBEDDING_LENGTH,
                EuclideanDistance16,
            >,
        >,
    ),
    UnquantizedOpenAi(Model, OpenAIHnsw),
}

impl HnswConfiguration {
    fn state(&self) -> HnswConfigurationState {
        let (typ, model) = match self {
            HnswConfiguration::QuantizedOpenAi(model, _) => {
                (HnswConfigurationType::QuantizedOpenAi, model)
            }
            HnswConfiguration::SmallQuantizedOpenAi(model, _) => {
                (HnswConfigurationType::SmallQuantizedOpenAi, model)
            }
            HnswConfiguration::UnquantizedOpenAi(model, _) => {
                (HnswConfigurationType::UnquantizedOpenAi, model)
            }
        };
        let version = 1;

        HnswConfigurationState {
            version,
            typ,
            model: *model,
        }
    }

    pub fn model(&self) -> Model {
        match self {
            HnswConfiguration::QuantizedOpenAi(m, _) => *m,
            HnswConfiguration::SmallQuantizedOpenAi(m, _) => *m,
            HnswConfiguration::UnquantizedOpenAi(m, _) => *m,
        }
    }

    #[allow(dead_code)]
    pub fn vector_count(&self) -> usize {
        match self {
            HnswConfiguration::QuantizedOpenAi(_model, q) => q.vector_count(),
            HnswConfiguration::SmallQuantizedOpenAi(_model, q) => q.vector_count(),
            HnswConfiguration::UnquantizedOpenAi(_model, h) => h.vector_count(),
        }
    }

    pub fn search(
        &self,
        v: AbstractVector<Embedding>,
        number_of_candidates: usize,
        probe_depth: usize,
    ) -> Vec<(VectorId, f32)> {
        match self {
            HnswConfiguration::QuantizedOpenAi(_model, q) => {
                q.search(v, number_of_candidates, probe_depth)
            }
            HnswConfiguration::SmallQuantizedOpenAi(_model, q) => {
                q.search(v, number_of_candidates, probe_depth)
            }
            HnswConfiguration::UnquantizedOpenAi(_model, h) => {
                h.search(v, number_of_candidates, probe_depth)
            }
        }
    }

    pub fn improve_neighbors(&mut self, threshold: f32, recall: f32) {
        match self {
            HnswConfiguration::QuantizedOpenAi(_model, q) => q.improve_neighbors(threshold, recall),
            HnswConfiguration::SmallQuantizedOpenAi(_model, q) => {
                q.improve_neighbors(threshold, recall)
            }
            HnswConfiguration::UnquantizedOpenAi(_model, h) => {
                h.improve_neighbors(threshold, recall)
            }
        }
    }

    pub fn zero_neighborhood_size(&self) -> usize {
        match self {
            HnswConfiguration::QuantizedOpenAi(_model, q) => q.zero_neighborhood_size(),
            HnswConfiguration::SmallQuantizedOpenAi(_model, q) => q.zero_neighborhood_size(),
            HnswConfiguration::UnquantizedOpenAi(_model, h) => h.zero_neighborhood_size(),
        }
    }
    pub fn threshold_nn(
        &self,
        threshold: f32,
        probe_depth: usize,
        initial_search_depth: usize,
    ) -> impl IndexedParallelIterator<Item = (VectorId, Vec<(VectorId, f32)>)> + '_ {
        match self {
            HnswConfiguration::QuantizedOpenAi(_model, q) => {
                Either::Left(q.threshold_nn(threshold, probe_depth, initial_search_depth))
            }
            HnswConfiguration::SmallQuantizedOpenAi(_model, q) => Either::Right(Either::Left(
                q.threshold_nn(threshold, probe_depth, initial_search_depth),
            )),
            HnswConfiguration::UnquantizedOpenAi(_model, h) => Either::Right(Either::Right(
                h.threshold_nn(threshold, probe_depth, initial_search_depth),
            )),
        }
    }
    pub fn stochastic_recall(&self, recall_proportion: f32) -> f32 {
        match self {
            HnswConfiguration::QuantizedOpenAi(_, q) => q.stochastic_recall(recall_proportion),
            HnswConfiguration::SmallQuantizedOpenAi(_, q) => q.stochastic_recall(recall_proportion),
            HnswConfiguration::UnquantizedOpenAi(_, h) => h.stochastic_recall(recall_proportion),
        }
    }
}

impl Serializable for HnswConfiguration {
    type Params = Arc<VectorStore>;

    fn serialize<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), parallel_hnsw::SerializationError> {
        match self {
            HnswConfiguration::QuantizedOpenAi(_, qhnsw) => {
                qhnsw.serialize(&path)?;
            }
            HnswConfiguration::UnquantizedOpenAi(_, hnsw) => {
                hnsw.serialize(&path)?;
            }
            HnswConfiguration::SmallQuantizedOpenAi(_, qhnsw) => {
                qhnsw.serialize(&path)?;
            }
        }
        let state_path: PathBuf = path.as_ref().join("state.json");
        let mut state_file = OpenOptions::new()
            .create(true)
            .write(true)
            .open(state_path)?;
        serde_json::to_writer(&mut state_file, &self.state())?;
        state_file.sync_data()?;

        Ok(())
    }

    fn deserialize<P: AsRef<std::path::Path>>(
        path: P,
        params: &Self::Params,
    ) -> Result<Self, parallel_hnsw::SerializationError> {
        let state_path: PathBuf = path.as_ref().join("state.json");
        let mut state_file = OpenOptions::new()
            .create(false)
            .read(true)
            .open(state_path)?;

        let state: HnswConfigurationState = serde_json::from_reader(&mut state_file)?;

        Ok(match state.typ {
            HnswConfigurationType::QuantizedOpenAi => HnswConfiguration::QuantizedOpenAi(
                state.model,
                QuantizedHnsw::deserialize(path, &(params.clone(), params.clone()))?,
            ),
            HnswConfigurationType::UnquantizedOpenAi => {
                HnswConfiguration::UnquantizedOpenAi(state.model, Hnsw::deserialize(path, params)?)
            }
            HnswConfigurationType::SmallQuantizedOpenAi => HnswConfiguration::SmallQuantizedOpenAi(
                state.model,
                QuantizedHnsw::deserialize(path, &(params.clone(), params.clone()))?,
            ),
        })
    }
}
