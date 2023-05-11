use std::{
    io,
    iter::{self, zip},
    sync::Arc,
};

use hnsw::{Hnsw, Searcher};
use rand_pcg::Lcg128Xsl64;

use crate::{
    openai::{embeddings_for, Embedding},
    server::Operation,
    vectors::{LoadedVec, VectorStore},
};
use space::{Metric, Neighbor};

pub type HnswIndex = Hnsw<OpenAI, Point, Lcg128Xsl64, 12, 24>;

#[derive(Clone, Debug, PartialEq)]
pub enum Point {
    Stored { id: String, vec: LoadedVec },
    Mem { vec: Box<Embedding> },
}

impl Point {
    fn vec(&self) -> &Embedding {
        match self {
            Point::Stored { id: _, vec } => vec,
            Point::Mem { vec } => vec,
        }
    }
}

#[derive(Clone)]
pub struct OpenAI;

impl Metric<Point> for OpenAI {
    type Unit = u32;
    fn distance(&self, p1: &Point, p2: &Point) -> u32 {
        let a = p1.vec();
        let b = p2.vec();
        let f = a.iter().zip(b.iter()).map(|(&a, &b)| (a - b)).sum::<f32>();
        (1.0 - f).to_bits()
    }
}

#[derive(Clone, Debug)]
pub enum PointOperation {
    Insert { point: Point },
    Replace { point: Point },
    Delete { id: String },
}

const API_KEY: &str = "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";

enum Op {
    Insert,
    Changed,
}

pub async fn operations_to_point_operations(
    domain: &str,
    vector_store: &VectorStore,
    structs: Vec<Result<Operation, std::io::Error>>,
) -> Vec<PointOperation> {
    let ops: Vec<Operation> = structs.into_iter().map(|ro| ro.unwrap()).collect();
    let tuples: Vec<(Op, String, String)> = ops
        .iter()
        .flat_map(|o| match o {
            Operation::Inserted { string, id } => Some((Op::Insert, string.into(), id.into())),
            Operation::Changed { string, id } => Some((Op::Changed, string.into(), id.into())),
            Operation::Deleted { id: _ } => None,
        })
        .collect();
    let strings: Vec<String> = tuples.iter().map(|(_, s, _)| s.to_string()).collect();
    let vecs: Vec<Embedding> = embeddings_for(API_KEY, &strings).await.unwrap();
    let domain = vector_store.get_domain(domain).unwrap();
    let loaded_vecs = vector_store
        .add_and_load_vecs(&domain, vecs.iter())
        .unwrap();
    let mut new_ops: Vec<PointOperation> = zip(tuples, loaded_vecs)
        .map(|((op, _, id), vec)| match op {
            Op::Insert => PointOperation::Insert {
                point: Point::Stored { vec, id },
            },
            Op::Changed => PointOperation::Replace {
                point: Point::Stored { vec, id },
            },
        })
        .collect();
    let mut delete_ops: Vec<_> = ops
        .into_iter()
        .flat_map(|o| match o {
            Operation::Deleted { id } => Some(PointOperation::Delete { id }),
            _ => None,
        })
        .collect();
    new_ops.append(&mut delete_ops);
    new_ops
}

pub struct IndexIdentifier {
    pub previous: Option<String>,
    pub commit: String,
    pub domain: String,
}

#[derive(Debug)]
enum IndexError {
    Failed,
}

/*
pub fn serialize_index(domain: Domain, hnsw: HnswIndex) -> io::Result<()> {}
 */

pub fn start_indexing_from_operations(
    mut hnsw: HnswIndex,
    operations: Vec<PointOperation>,
) -> Result<HnswIndex, io::Error> {
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
}

#[derive(Debug)]
enum SearchError {
    SearchFailed,
}

#[derive(Clone, Debug, PartialEq)]
struct PointQuery {
    point: Point,
    distance: u32,
}

fn search(
    p: &Point,
    num: usize,
    hnsw: Hnsw<OpenAI, Point, Lcg128Xsl64, 12, 24>,
) -> Result<Vec<PointQuery>, SearchError> {
    let mut output: Vec<_> = iter::repeat(Neighbor {
        index: !0,
        distance: !0,
    })
    .take(num)
    .collect();
    let mut searcher = Searcher::default();
    hnsw.nearest(p, 24, &mut searcher, &mut output);
    let mut points = Vec::with_capacity(num);
    for elt in output {
        points.push(PointQuery {
            point: hnsw.feature(elt.index).clone(),
            distance: elt.distance,
        })
    }
    Ok(points)
}

#[cfg(test)]
mod tests {
    use crate::vectors::VectorStore;

    use super::*;

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
        let e1 = store.get_vec(&domain, 0).unwrap().unwrap();
        let e2 = store.get_vec(&domain, 1).unwrap().unwrap();
        let e3 = store.get_vec(&domain, 2).unwrap().unwrap();
        let e4 = store.get_vec(&domain, 3).unwrap().unwrap();

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
        candidate_vec[0] = 0.707;
        candidate_vec[1] = 0.707;

        let p = Point::Mem {
            vec: Box::new(candidate_vec),
        };
        let points = search(&p, 4, hnsw).unwrap();
        let p1 = &points[0];
        let p2 = &points[1];
        assert_eq!(*p1.point.vec(), *e1);
        assert_eq!(*p2.point.vec(), *e2);
    }
}
