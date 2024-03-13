use parallel_hnsw::{pq::QuantizedHnsw, Serializable};
use serde::{Deserialize, Serialize};

use crate::{
    comparator::{Centroid32Comparator, OpenAIComparator, QuantizedComparator},
    vecmath::{CENTROID_32_LENGTH, EMBEDDING_LENGTH, QUANTIZED_EMBEDDING_LENGTH},
};

pub type OpenAIHnsw = Hnsw<OpenAIComparator>;

#[derive(Serialize, Deserialize)]
pub enum HnswConfigurationType {
    QuantizedOpenAi,
    UnquantizedOpenAi,
}

pub enum HnswConfiguration {
    QuantizedOpenAI(
        QuantizedHnsw<
            EMBEDDING_LENGTH,
            CENTROID_32_LENGTH,
            QUANTIZED_EMBEDDING_LENGTH,
            Centroid32Comparator,
            QuantizedComparator,
            OpenAIComparator,
        >,
    ),
    UnquantizedOpenAi(OpenAIHnsw),
}

impl Serializable for HnswConfiguration {
    type Params = Arc<VectorStore>;

    fn serialize<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), parallel_hnsw::SerializationError> {
        match self {
            HnswConfiguration::QuantizedOpenAI(hnsw) => {
                let mut type_meta: PathBuf = path.as_ref().into().join("type");

                hnsw.serialize(path);
            }
            HnswConfiguration::UnquantizedOpenAi(_) => todo!(),
        }
    }

    fn deserialize<P: AsRef<std::path::Path>>(
        path: P,
        params: Self::Params,
    ) -> Result<Self, parallel_hnsw::SerializationError> {
        todo!()
    }
}
