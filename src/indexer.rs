use std::{iter, sync::Arc};

use hnsw::{Hnsw, Searcher};
use rand_pcg::{Lcg128Xsl64, Pcg64};
use space::{Metric, Neighbor};

#[derive(Clone, Debug, PartialEq)]
struct Point<'a> {
    id: String,
    vec: &'a [f32],
}

pub struct OpenAI;

impl<'a> Metric<Point<'a>> for OpenAI {
    type Unit = u32;
    fn distance(&self, p1: &Point, p2: &Point) -> u32 {
        let a = &p1.vec;
        let b = &p2.vec;
        let f = a.iter().zip(b.iter()).map(|(&a, &b)| (a - b)).sum::<f32>();
        (1.0 - f).to_bits()
    }
}

#[derive(Clone, Debug)]
enum Operation<'a> {
    Insert { point: Point<'a> },
    //    Replace { point: Point },
    //    Delete { point: Point },
}

struct Domain {
    previous: String,
    commit: String,
    domain: String,
}

fn load_hnsw<'a>(domain: Domain) -> Hnsw<OpenAI, Point<'a>, Pcg64, 12, 24> {
    Hnsw::new(OpenAI)
}

#[derive(Debug)]
enum IndexError {
    Failed,
}

fn index_points<'a>(
    operations: Vec<Operation<'a>>,
    domain: Domain,
) -> Result<Hnsw<OpenAI, Point<'a>, Lcg128Xsl64, 12, 24>, IndexError> {
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
struct PointQuery<'a> {
    point: Point<'a>,
    distance: u32,
}

fn search<'a, 'b>(
    p: &Point<'a>,
    num: usize,
    hnsw: Hnsw<OpenAI, Point<'b>, Lcg128Xsl64, 12, 24>,
) -> Result<Vec<PointQuery<'b>>, SearchError> {
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
        let vector_block: Vec<Vec<f32>> = [
            [0.0_f32, -1.0],
            [1.0_f32, 0.0],
            [0.0_f32, 1.0],
            [-1.0_f32, 0.0],
        ]
        .map(|a| a.into_iter().collect())
        .into_iter()
        .collect();
        let operations: Vec<_> = [
            Operation::Insert {
                point: Point {
                    id: "Point/1".to_string(),
                    vec: &vector_block[0],
                },
            },
            Operation::Insert {
                point: Point {
                    id: "Point/2".to_string(),
                    vec: &vector_block[1],
                },
            },
            Operation::Insert {
                point: Point {
                    id: "Point/3".to_string(),
                    vec: &vector_block[2],
                },
            },
            Operation::Insert {
                point: Point {
                    id: "Point/4".to_string(),
                    vec: &vector_block[3],
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
        let candidate_vec: Vec<f32> = [0.707_f32, 0.707].into_iter().collect();
        let p = Point {
            id: "unknown".to_string(),
            vec: &candidate_vec,
        };
        let points = search(&p, 4, hnsw);
        assert_eq!(
            points.unwrap(),
            [
                PointQuery {
                    point: Point {
                        id: "Point/2".to_string(),
                        vec: &[1.0, 0.0]
                    },
                    distance: 1058407448
                },
                PointQuery {
                    point: Point {
                        id: "Point/3".to_string(),
                        vec: &[0.0, 1.0]
                    },
                    distance: 1058407448
                },
                PointQuery {
                    point: Point {
                        id: "Point/1".to_string(),
                        vec: &[0.0, -1.0]
                    },
                    distance: 3216309748
                },
                PointQuery {
                    point: Point {
                        id: "Point/4".to_string(),
                        vec: &[-1.0, 0.0]
                    },
                    distance: 3216309748
                }
            ]
        );
    }
}
