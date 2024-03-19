#![allow(unused, dead_code)]
use crate::{
    comparator::OpenAIComparator,
    configuration::OpenAIHnsw,
    openai::{embeddings_for, EmbeddingError, Model},
    server::Operation,
    vecmath::{self, Embedding},
    vectors::{Domain, LoadedVec, VectorStore},
};
use parallel_hnsw::{AbstractVector, Hnsw, SerializationError, VectorId};
use rand_pcg::Lcg128Xsl64;
use rayon::prelude::*;
use space::{Metric, Neighbor};
use std::{fs::File, path::Path, sync::Arc};
use std::{
    io,
    iter::{self, zip},
    path::PathBuf,
};
use thiserror::Error;
use tokio::task::JoinError;
use urlencoding::{decode, encode};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
pub enum Point {
    Stored { id: String, vec: LoadedVec },
    Mem { vec: Box<Embedding> },
}

pub struct IndexPoint {
    id: String,
    index: usize,
}

impl Point {
    pub fn id(&self) -> &str {
        match self {
            Point::Stored { id, vec } => id,
            Point::Mem { vec } => panic!("You can not get the external id of a memory point"),
        }
    }

    fn vec_id(&self) -> usize {
        match self {
            Point::Stored { id, vec } => vec.id(),
            Point::Mem { vec } => panic!("You can not get the vector id of a memory point"),
        }
    }

    fn vec(&self) -> &Embedding {
        match self {
            Point::Stored { id: _, vec } => vec,
            Point::Mem { vec } => vec,
        }
    }

    fn abstract_vector(&self) -> AbstractVector<Embedding> {
        match self {
            Point::Stored { id, vec } => AbstractVector::Stored(VectorId(vec.id())),
            Point::Mem { vec } => AbstractVector::Unstored(vec),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct OpenAI;

impl Metric<Point> for OpenAI {
    type Unit = u32;
    fn distance(&self, p1: &Point, p2: &Point) -> u32 {
        let a = p1.vec();
        let b = p2.vec();
        let f = vecmath::normalized_cosine_distance(a, b);
        f.to_bits()
    }
}

impl Metric<IndexPoint> for OpenAI {
    type Unit = u32;
    fn distance(&self, _p1: &IndexPoint, _p2: &IndexPoint) -> u32 {
        unimplemented!()
    }
}

#[derive(Clone, Debug)]
pub enum PointOperation {
    Insert { point: Point },
    Replace { point: Point },
    Delete { id: String },
}

enum Op {
    Insert,
    Changed,
}

pub struct IndexIdentifier {
    pub previous: Option<String>,
    pub commit: String,
    pub domain: String,
}

#[derive(Debug, Error)]
pub enum IndexError {
    #[error("Indexing failed")]
    Failed,
    #[error("Indexing failed with io error: {0:?}")]
    IoError(#[from] std::io::Error),
    #[error("Embedding error: {0:?}")]
    EmbeddingError(#[from] EmbeddingError),
    #[error("Join error: {0:?}")]
    JoinError(#[from] JoinError),
}

pub fn start_indexing_from_operations(
    hnsw: Arc<OpenAIHnsw>,
    operations: Vec<PointOperation>,
) -> Result<Arc<OpenAIHnsw>, io::Error> {
    todo!()
    /*
    let mut searcher = Searcher::default();
    for operation in operations {
        match operation {
            PointOperation::Insert { point } => {
                hnsw.insert(point.clone(), &mut searcher);
            }
            PointOperation::Replace { point: _ } => todo!(),
            PointOperation::Delete { id: _ } => todo!(),
        }
    }
    // Put this index somewhere!
    //todo!()
    Ok(hnsw)
    */
}

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("Search failed for unknown reason")]
    SearchFailed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PointQuery {
    id: usize,
    distance: f32,
}

impl PointQuery {
    pub fn internal_id(&self) -> usize {
        self.id
    }

    pub fn distance(&self) -> f32 {
        self.distance
    }
}

pub fn search(p: &Point, mut num: usize, hnsw: &OpenAIHnsw) -> Vec<PointQuery> {
    let ef = num.max(100);
    let output = hnsw.search(p.abstract_vector(), ef, 2);
    let points = output
        .into_iter()
        .map(|elt| PointQuery {
            id: elt.0 .0,
            distance: elt.1,
        })
        .collect();
    points
}

pub fn index_serialization_path<P: AsRef<Path>>(path: P, name: &str) -> PathBuf {
    let mut path: PathBuf = path.as_ref().into();
    path.push(format!("{name}.hnsw"));
    path
}

pub fn create_index_name(domain: &str, commit: &str) -> String {
    let domain = encode(domain);
    format!("{}@{}", domain, commit)
}

pub fn parse_index_name(name: &str) -> (String, String) {
    let (domain, commit) = name.split_once('@').unwrap();
    let domain = decode(domain).unwrap();
    (domain.to_string(), commit.to_string())
}

#[cfg(test)]
mod tests {
    use crate::vectors::VectorStore;

    use super::*;

    /*
    #[test]
    fn low_dimensional_search() {
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path();
        let store = VectorStore::new(path, 2);

        let mut vector_block: Vec<Embedding> = [[0.0; 1536], [0.0; 1536], [0.0; 1536], [0.0; 1536]]
            .into_iter()
            .collect();
        vector_block[0][0] = 1.0;
        vector_block[1][1] = 1.0;
        vector_block[2][0] = -1.0;
        vector_block[3][1] = -1.0;

        let domain = store.get_domain("foo").unwrap();
        let ids = store.add_vecs(&domain, vector_block.iter()).unwrap();
        let e1 = store.get_vec(&domain, ids[0]).unwrap().unwrap();
        let e2 = store.get_vec(&domain, ids[1]).unwrap().unwrap();
        let e3 = store.get_vec(&domain, ids[2]).unwrap().unwrap();
        let e4 = store.get_vec(&domain, ids[3]).unwrap().unwrap();

        let operations: Vec<_> = [
            PointOperation::Insert {
                point: Point::Stored {
                    id: "Point/1".to_string(),
                    vec: e1.clone(),
                },
            },
            PointOperation::Insert {
                point: Point::Stored {
                    id: "Point/2".to_string(),
                    vec: e2.clone(),
                },
            },
            PointOperation::Insert {
                point: Point::Stored {
                    id: "Point/3".to_string(),
                    vec: e3,
                },
            },
            PointOperation::Insert {
                point: Point::Stored {
                    id: "Point/4".to_string(),
                    vec: e4,
                },
            },
        ]
        .into_iter()
        .collect();
        let hnsw = start_indexing_from_operations(Hnsw::new(OpenAI), operations).unwrap();
        let mut candidate_vec: Embedding = [0.0; 1536];
        candidate_vec[03] = 0.8;
        candidate_vec[1] = 0.6;

        let p = Point::Mem {
            vec: Box::new(candidate_vec),
        };
        let points = search(&p, 4, &hnsw);
        let p1 = &points[0];
        let p2 = &points[1];
        assert_eq!(*p1.point.vec(), *e1);
        assert_eq!(*p2.point.vec(), *e2);
    }
    */
}
