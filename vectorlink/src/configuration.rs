use std::{fs::OpenOptions, path::PathBuf, sync::Arc};

use itertools::Either;
use parallel_hnsw::{pq::QuantizedHnsw, AbstractVector, Hnsw, Serializable, VectorId};
use rayon::iter::IndexedParallelIterator;
use serde::{Deserialize, Serialize};

use crate::{
    comparator::{
        Centroid16Comparator, Centroid32Comparator, Centroid4Comparator, Centroid8Comparator,
        DiskOpenAIComparator, OpenAIComparator, Quantized16Comparator, Quantized32Comparator,
        Quantized4Comparator, Quantized8Comparator,
    },
    openai::Model,
    vecmath::{
        Embedding, CENTROID_16_LENGTH, CENTROID_32_LENGTH, CENTROID_4_LENGTH, CENTROID_8_LENGTH,
        EMBEDDING_LENGTH, QUANTIZED_16_EMBEDDING_LENGTH, QUANTIZED_32_EMBEDDING_LENGTH,
        QUANTIZED_4_EMBEDDING_LENGTH, QUANTIZED_8_EMBEDDING_LENGTH,
    },
    vectors::VectorStore,
};

pub type OpenAIHnsw = Hnsw<OpenAIComparator>;

#[derive(Serialize, Deserialize)]
pub enum HnswConfigurationType {
    QuantizedOpenAi,
    SmallQuantizedOpenAi,
    SmallQuantizedOpenAi8,
    SmallQuantizedOpenAi4,
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
            Centroid32Comparator,
            Quantized32Comparator,
            DiskOpenAIComparator,
        >,
    ),
    SmallQuantizedOpenAi(
        Model,
        QuantizedHnsw<
            EMBEDDING_LENGTH,
            CENTROID_16_LENGTH,
            QUANTIZED_16_EMBEDDING_LENGTH,
            Centroid16Comparator,
            Quantized16Comparator,
            DiskOpenAIComparator,
        >,
    ),
    SmallQuantizedOpenAi8(
        Model,
        QuantizedHnsw<
            EMBEDDING_LENGTH,
            CENTROID_8_LENGTH,
            QUANTIZED_8_EMBEDDING_LENGTH,
            Centroid8Comparator,
            Quantized8Comparator,
            DiskOpenAIComparator,
        >,
    ),
    SmallQuantizedOpenAi4(
        Model,
        QuantizedHnsw<
            EMBEDDING_LENGTH,
            CENTROID_4_LENGTH,
            QUANTIZED_4_EMBEDDING_LENGTH,
            Centroid4Comparator,
            Quantized4Comparator,
            DiskOpenAIComparator,
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
            HnswConfiguration::SmallQuantizedOpenAi8(model, _) => {
                (HnswConfigurationType::SmallQuantizedOpenAi8, model)
            }
            HnswConfiguration::SmallQuantizedOpenAi4(model, _) => {
                (HnswConfigurationType::SmallQuantizedOpenAi4, model)
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
            HnswConfiguration::SmallQuantizedOpenAi8(m, _) => *m,
            HnswConfiguration::SmallQuantizedOpenAi4(m, _) => *m,
        }
    }

    #[allow(dead_code)]
    pub fn vector_count(&self) -> usize {
        match self {
            HnswConfiguration::QuantizedOpenAi(_model, q) => q.vector_count(),
            HnswConfiguration::SmallQuantizedOpenAi(_model, q) => q.vector_count(),
            HnswConfiguration::UnquantizedOpenAi(_model, h) => h.vector_count(),
            HnswConfiguration::SmallQuantizedOpenAi8(_model, q) => q.vector_count(),
            HnswConfiguration::SmallQuantizedOpenAi4(_model, q) => q.vector_count(),
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
            HnswConfiguration::SmallQuantizedOpenAi8(_, q) => {
                q.search(v, number_of_candidates, probe_depth)
            }
            HnswConfiguration::SmallQuantizedOpenAi4(_, q) => {
                q.search(v, number_of_candidates, probe_depth)
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
            HnswConfiguration::SmallQuantizedOpenAi8(_, q) => {
                q.improve_neighbors(threshold, recall)
            }
            HnswConfiguration::SmallQuantizedOpenAi4(_, q) => {
                q.improve_neighbors(threshold, recall)
            }
        }
    }

    pub fn zero_neighborhood_size(&self) -> usize {
        match self {
            HnswConfiguration::QuantizedOpenAi(_model, q) => q.zero_neighborhood_size(),
            HnswConfiguration::SmallQuantizedOpenAi(_model, q) => q.zero_neighborhood_size(),
            HnswConfiguration::UnquantizedOpenAi(_model, h) => h.zero_neighborhood_size(),
            HnswConfiguration::SmallQuantizedOpenAi8(_model, q) => q.zero_neighborhood_size(),
            HnswConfiguration::SmallQuantizedOpenAi4(_model, q) => q.zero_neighborhood_size(),
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
                Either::Left(h.threshold_nn(threshold, probe_depth, initial_search_depth)),
            )),
            HnswConfiguration::SmallQuantizedOpenAi8(_model, q) => {
                Either::Right(Either::Right(Either::Right(Either::Left(q.threshold_nn(
                    threshold,
                    probe_depth,
                    initial_search_depth,
                )))))
            }
            HnswConfiguration::SmallQuantizedOpenAi4(_model, q) => {
                Either::Right(Either::Right(Either::Right(Either::Right(q.threshold_nn(
                    threshold,
                    probe_depth,
                    initial_search_depth,
                )))))
            }
        }
    }

    pub fn stochastic_recall(&self, recall_proportion: f32) -> f32 {
        match self {
            HnswConfiguration::QuantizedOpenAi(_, q) => q.stochastic_recall(recall_proportion),
            HnswConfiguration::SmallQuantizedOpenAi(_, q) => q.stochastic_recall(recall_proportion),
            HnswConfiguration::UnquantizedOpenAi(_, h) => h.stochastic_recall(recall_proportion),
            HnswConfiguration::SmallQuantizedOpenAi8(_, q) => {
                q.stochastic_recall(recall_proportion)
            }
            HnswConfiguration::SmallQuantizedOpenAi4(_, q) => {
                q.stochastic_recall(recall_proportion)
            }
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
            HnswConfiguration::UnquantizedOpenAi(_, hhnsw) => hhnsw.serialize(&path)?,
            HnswConfiguration::QuantizedOpenAi(_, qnsw) => qnsw.serialize(&path)?,
            HnswConfiguration::SmallQuantizedOpenAi(_, qhnsw) => qhnsw.serialize(&path)?,
            HnswConfiguration::SmallQuantizedOpenAi8(_, qhnsw) => qhnsw.serialize(&path)?,
            HnswConfiguration::SmallQuantizedOpenAi4(_, qhnsw) => qhnsw.serialize(&path)?,
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
        params: Self::Params,
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
                QuantizedHnsw::deserialize(path, params)?,
            ),
            HnswConfigurationType::UnquantizedOpenAi => {
                HnswConfiguration::UnquantizedOpenAi(state.model, Hnsw::deserialize(path, params)?)
            }
            HnswConfigurationType::SmallQuantizedOpenAi => HnswConfiguration::SmallQuantizedOpenAi(
                state.model,
                QuantizedHnsw::deserialize(path, params)?,
            ),
            HnswConfigurationType::SmallQuantizedOpenAi8 => {
                HnswConfiguration::SmallQuantizedOpenAi8(
                    state.model,
                    QuantizedHnsw::deserialize(path, params)?,
                )
            }
            HnswConfigurationType::SmallQuantizedOpenAi4 => {
                HnswConfiguration::SmallQuantizedOpenAi4(
                    state.model,
                    QuantizedHnsw::deserialize(path, params)?,
                )
            }
        })
    }
}
