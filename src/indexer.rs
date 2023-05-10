use std::{collections::HashMap, io, iter, sync::Arc};

use futures::Stream;
use hnsw::{Hnsw, Searcher};
use rand_pcg::{Lcg128Xsl64, Pcg64};

use crate::{
    openai::Embedding,
    server::{Operation, Service},
};
use space::{Metric, Neighbor};

use tokio_stream::StreamExt;

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
enum PointOperation {
    Insert { point: Point },
    //    Replace { point: Point },
    //    Delete { point: Point },
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

fn index_points(
    operations: Vec<PointOperation>,
    domain: IndexIdentifier,
) -> Result<Hnsw<OpenAI, Point, Lcg128Xsl64, 12, 24>, IndexError> {
    todo!()
    /*
    let mut hnsw = load_hnsw(domain);
    let mut searcher = Searcher::default();
    for operation in &operations {
        match operation {
            PointOperation::Insert { point } => {
                hnsw.insert(point.clone(), &mut searcher);
            }
        }
    }
        Ok(hnsw)
     */
}

pub async fn start_indexing_from_operations(
    hnsw: Hnsw<OpenAI, Point, Lcg128Xsl64, 12, 24>,
    operations: impl Stream<Item = io::Result<Operation>> + Unpin,
) -> Result<(), io::Error> {
    todo!();
    /*
    let mut searcher = Searcher::default();
    while let Some(operation) = operations.try_next().await? {
        match operation {
            Operation::Insert { point } => {
                hnsw.insert(point.clone(), &mut searcher);
            }
        }
    }
    // Put this index somewhere!
    todo!();
    */
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
        let hnsw = index_points(
            operations,
            IndexIdentifier {
                previous: None,
                commit: "commit".to_string(),
                domain: "here".to_string(),
            },
        )
        .unwrap();
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
