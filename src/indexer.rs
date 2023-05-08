use std::{iter, sync::Arc};

use hnsw::{Hnsw, Searcher};
use rand_pcg::{Lcg128Xsl64, Pcg64};
use space::{Metric, Neighbor};

use crate::openai::Embedding;

#[derive(Clone, Debug, PartialEq)]
struct Point {
    id: String,
    vec: Arc<Embedding>,
}

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
enum Operation {
    Insert { point: Point },
    //    Replace { point: Point },
    //    Delete { point: Point },
}

struct Domain {
    previous: String,
    commit: String,
    domain: String,
}

fn load_hnsw(domain: Domain) -> Hnsw<OpenAI, Point, Pcg64, 12, 24> {
    Hnsw::new(OpenAI)
}

#[derive(Debug)]
enum IndexError {
    Failed,
}

fn index_points(
    operations: Vec<Operation>,
    domain: Domain,
) -> Result<Hnsw<OpenAI, Point, Lcg128Xsl64, 12, 24>, IndexError> {
    let mut hnsw = load_hnsw(domain);
    let mut searcher = Searcher::default();
    for operation in &operations {
        match operation {
            Operation::Insert { point } => {
                hnsw.insert(point.clone(), &mut searcher);
            }
        }
    }
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

fn search<'a, 'b>(
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
            Operation::Insert {
                point: Point {
                    id: "Point/1".to_string(),
                    vec: Arc::new(vector_block[0]),
                },
            },
            Operation::Insert {
                point: Point {
                    id: "Point/2".to_string(),
                    vec: Arc::new(vector_block[1]),
                },
            },
            Operation::Insert {
                point: Point {
                    id: "Point/3".to_string(),
                    vec: Arc::new(vector_block[2]),
                },
            },
            Operation::Insert {
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
            Domain {
                previous: "previous".to_string(),
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
