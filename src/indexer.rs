use std::{io, iter, sync::Arc};

use hnsw::{Hnsw, Searcher};
use rand_pcg::Lcg128Xsl64;

use crate::{
    openai::{embeddings_for, Embedding},
    server::Operation,
};
use space::{Metric, Neighbor};

pub type HnswIndex = Hnsw<OpenAI, Point, Lcg128Xsl64, 12, 24>;

#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    id: String,
    vec: Arc<Embedding>,
}

#[derive(Clone)]
pub struct OpenAI;

impl Metric<Point> for OpenAI {
    type Unit = u32;
    fn distance(&self, p1: &Point, p2: &Point) -> u32 {
        let a = &p1.vec;
        let b = &p2.vec;
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

pub async fn operations_to_point_operations(
    structs: Vec<Result<Operation, std::io::Error>>,
) -> Vec<PointOperation> {
    let ops: Vec<Operation> = structs.into_iter().map(|ro| ro.unwrap()).collect();
    let strings: Vec<String> = ops
        .iter()
        .map(|o| match o {
            Operation::Inserted { string, id: _ } => string.into(),
            Operation::Changed { string, id: _ } => string.into(),
            Operation::Deleted { id: _ } => "".to_string(),
        })
        .collect();
    let vecs: Vec<Embedding> = embeddings_for(API_KEY, &strings).await.unwrap();
    let new_ops: Vec<PointOperation> = ops
        .into_iter()
        .enumerate()
        .map(|(i, o)| match o {
            Operation::Inserted { string: _, id } => PointOperation::Insert {
                point: Point {
                    vec: Arc::new(vecs[i]),
                    id,
                },
            },
            Operation::Changed { string: _, id } => PointOperation::Replace {
                point: Point {
                    vec: Arc::new(vecs[i]),
                    id,
                },
            },
            Operation::Deleted { id } => PointOperation::Delete { id },
        })
        .collect();
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
    use super::*;

    #[test]
    fn low_dimensional_search() {
        let mut vector_block: Vec<Embedding> = [[0.0; 1536], [0.0; 1536], [0.0; 1536], [0.0; 1536]]
            .into_iter()
            .collect();
        vector_block[0][0] = 1.0;
        vector_block[1][1] = 1.0;
        vector_block[2][0] = -1.0;
        vector_block[3][1] = -1.0;
        let operations: Vec<_> = [
            PointOperation::Insert {
                point: Point {
                    id: "Point/1".to_string(),
                    vec: Arc::new(vector_block[0]),
                },
            },
            PointOperation::Insert {
                point: Point {
                    id: "Point/2".to_string(),
                    vec: Arc::new(vector_block[1]),
                },
            },
            PointOperation::Insert {
                point: Point {
                    id: "Point/3".to_string(),
                    vec: Arc::new(vector_block[2]),
                },
            },
            PointOperation::Insert {
                point: Point {
                    id: "Point/4".to_string(),
                    vec: Arc::new(vector_block[3]),
                },
            },
        ]
        .into_iter()
        .collect();
        let hnsw = start_indexing_from_operations(Hnsw::new(OpenAI), operations).unwrap();
        let mut candidate_vec: Embedding = [0.0; 1536];
        candidate_vec[0] = 0.707;
        candidate_vec[1] = 0.707;
        let p = Point {
            id: "unknown".to_string(),
            vec: Arc::new(candidate_vec),
        };
        let points = search(&p, 4, hnsw).unwrap();
        let p1 = &points[0];
        let p2 = &points[1];
        assert_eq!(p1.point.vec[0], 1.0);
        assert_eq!(p1.point.vec[1], 0.0);
        assert_eq!(p2.point.vec[0], 0.0);
        assert_eq!(p2.point.vec[1], 1.0);
    }
}
