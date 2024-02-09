#![feature(portable_simd)]

use std::collections::HashSet;
use std::io::ErrorKind;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
mod batch;
mod comparator;
mod indexer;
mod openai;
mod server;
mod vecmath;
mod vectors;

use batch::index_from_operations_file;
use clap::CommandFactory;
use clap::{Parser, Subcommand, ValueEnum};
//use hnsw::Hnsw;
use indexer::start_indexing_from_operations;
use indexer::Point;
use indexer::{index_serialization_path, serialize_index};
use indexer::{operations_to_point_operations, OpenAI};
use itertools::Itertools;
use openai::Model;
use parallel_hnsw::{AbstractVector, AllVectorIterator, Hnsw, NodeDistance, NodeId, VectorId};
use rand::thread_rng;
use rand::*;
use server::Operation;
use space::Metric;
use std::fs::File;
use std::io::{self, BufRead};

use rayon::iter::Either;
use rayon::prelude::*;

use crate::indexer::deserialize_index;

use {
    indexer::{create_index_name, HnswIndex},
    vecmath::empty_embedding,
    vectors::VectorStore,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Serve {
        #[arg(short, long)]
        content_endpoint: Option<String>,
        #[arg(short, long)]
        user_forward_header: Option<String>,
        #[arg(short, long)]
        directory: String,
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
        #[arg(short, long, default_value_t = 10000)]
        size: usize,
    },
    Load {
        #[arg(short, long)]
        key: Option<String>,
        #[arg(short, long)]
        commit: String,
        #[arg(long)]
        domain: String,
        #[arg(short, long)]
        directory: String,
        #[arg(short, long)]
        input: String,
        #[arg(short, long, default_value_t = 10000)]
        size: usize,
    },
    Load2 {
        #[arg(short, long)]
        key: Option<String>,
        #[arg(short, long)]
        commit: String,
        #[arg(long)]
        domain: String,
        #[arg(short, long)]
        directory: String,
        #[arg(short, long)]
        input: String,
        #[arg(short, long, default_value_t = 10000)]
        size: usize,
        #[arg(short, long, value_enum, default_value_t = Model::Ada2)]
        model: Model,
    },
    Embed {
        #[arg(short, long)]
        key: Option<String>,
        #[arg(short, long)]
        string: String,
        #[arg(short, long, value_enum, default_value_t = Model::Ada2)]
        model: Model,
    },
    Compare {
        #[arg(short, long)]
        key: Option<String>,
        #[arg(long)]
        s1: String,
        #[arg(long)]
        s2: String,
        #[arg(short, long, value_enum, default_value_t = Model::Ada2)]
        model: Model,
    },
    Compare2 {
        #[arg(short, long)]
        key: Option<String>,
        #[arg(long)]
        s1: String,
        #[arg(long)]
        s2: String,
        #[arg(short, long, value_enum, default_value_t=DistanceVariant::Default)]
        variant: DistanceVariant,
        #[arg(short, long, value_enum, default_value_t = Model::Ada2)]
        model: Model,
    },
    CompareModels {
        #[arg(short, long)]
        key: Option<String>,
        #[arg(long)]
        word: String,
        #[arg(long)]
        near1: String,
        #[arg(long)]
        near2: String,
    },
    TestRecall {
        #[arg(short, long)]
        commit: String,
        #[arg(long)]
        domain: String,
        #[arg(short, long)]
        directory: String,
        #[arg(short, long, default_value_t = 10000)]
        size: usize,
    },
    Diagnostics {
        #[arg(short, long)]
        commit: String,
        #[arg(long)]
        domain: String,
        #[arg(short, long)]
        directory: String,
        #[arg(short, long, default_value_t = 10000)]
        size: usize,
        #[arg(short, long, default_value_t = 0)]
        layer: usize,
    },
    Duplicates {
        #[arg(short, long)]
        commit: String,
        #[arg(long)]
        domain: String,
        #[arg(short, long)]
        directory: String,
        #[arg(short, long)]
        take: Option<usize>,
        #[arg(short, long, default_value_t = 10000)]
        size: usize,
        #[arg(short, long, default_value_t = 1.0_f32)]
        threshold: f32,
    },
    Test {
        #[arg(short, long)]
        key: Option<String>,
        #[arg(short, long, value_enum, default_value_t = Model::Ada2)]
        model: Model,
    },
    ImproveIndex {
        #[arg(short, long)]
        commit: String,
        #[arg(long)]
        domain: String,
        #[arg(short, long)]
        directory: String,
        #[arg(short, long, default_value_t = 10000)]
        size: usize,
        #[arg(short, long)]
        improve_neighbors: Option<f32>,
        #[arg(short, long)]
        promote: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DistanceVariant {
    Default,
    Simd,
    Scalar,
}

fn key_or_env(k: Option<String>) -> String {
    let result = k.or_else(|| std::env::var("OPENAI_KEY").ok());
    if result.is_none() {
        let mut app = Args::command();
        eprintln!("Error: no OpenAI key given. Configure it with the OPENAI_KEY environment variable, or by passing in the --key argument");
        app.print_help().unwrap();
        std::process::exit(2);
    }

    result.unwrap()
}

fn content_endpoint_or_env(c: Option<String>) -> Option<String> {
    c.or_else(|| std::env::var("TERMINUSDB_CONTENT_ENDPOINT").ok())
}

fn user_forward_header_or_env(c: Option<String>) -> String {
    c.unwrap_or_else(|| std::env::var("TERMINUSDB_USER_FORWARD_HEADER").unwrap())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    match args.command {
        Commands::Serve {
            content_endpoint,
            user_forward_header,
            directory,
            port,
            size,
        } => {
            server::serve(
                directory,
                user_forward_header_or_env(user_forward_header),
                port,
                size,
                content_endpoint_or_env(content_endpoint),
            )
            .await?
        }
        Commands::Embed { key, string, model } => {
            let v: Vec<[f32; 1536]> = openai::embeddings_for(&key_or_env(key), &[string], model)
                .await?
                .0;
            eprintln!("{:?}", v);
        }
        Commands::Compare { key, s1, s2, model } => {
            let v = openai::embeddings_for(&key_or_env(key), &[s1, s2], model)
                .await?
                .0;
            let p1 = Point::Mem {
                vec: Box::new(v[0]),
            };
            let p2 = Point::Mem {
                vec: Box::new(v[1]),
            };
            println!(
                "same? {}, distance: {}",
                p1 == p2,
                f32::from_bits(OpenAI.distance(&p1, &p2))
            );
        }
        Commands::Compare2 {
            key,
            s1,
            s2,
            variant,
            model,
        } => {
            let v = openai::embeddings_for(&key_or_env(key), &[s1, s2], model)
                .await?
                .0;
            let p1 = &v[0];
            let p2 = &v[1];
            let distance = match variant {
                DistanceVariant::Default => vecmath::normalized_cosine_distance(p1, p2),
                DistanceVariant::Scalar => vecmath::normalized_cosine_distance_scalar(p1, p2),
                DistanceVariant::Simd => vecmath::normalized_cosine_distance_simd(p1, p2),
            };
            println!("distance: {}", distance);
        }
        Commands::CompareModels {
            key,
            word,
            near1,
            near2,
        } => {
            let strings = [word, near1, near2];
            for model in [Model::Ada2, Model::Small3] {
                let v = openai::embeddings_for(&key_or_env(key.clone()), &strings, model)
                    .await?
                    .0;
                let embedding_word = &v[0];
                let embedding_n1 = &v[1];
                let embedding_n2 = &v[2];
                let distance1 = vecmath::normalized_cosine_distance(embedding_word, embedding_n1);
                let distance2 = vecmath::normalized_cosine_distance(embedding_word, embedding_n2);
                println!("{model:?}: {distance1} {distance2}");
            }
        }
        Commands::Test { key, model } => {
            let v = openai::embeddings_for(
                &key_or_env(key),
                &[
                    "king".to_string(),
                    "man".to_string(),
                    "woman".to_string(),
                    "queen".to_string(),
                ],
                model,
            )
            .await?
            .0;
            let mut calculated = empty_embedding();
            for (i, calculated) in calculated.iter_mut().enumerate() {
                *calculated = v[0][i] - v[1][i] + v[2][i];
            }
            let distance = vecmath::normalized_cosine_distance(&v[3], &calculated);
            eprintln!("{}", distance);
        }
        Commands::Load {
            key,
            domain,
            commit,
            directory,
            input,
            size,
        } => {
            let path = Path::new(&input);
            let dirpath = Path::new(&directory);
            panic!("yikes!");
            /*
            let mut hnsw: HnswIndex<OpenAIComparator, = Hnsw::new(OpenAI);
            let store = VectorStore::new(dirpath, size);
            let resolved_domain = store.get_domain(&domain)?;

            let f = File::options().read(true).open(path)?;

            let lines = io::BufReader::new(f).lines();
            let opstream = &lines
                .map(|l| {
                    let ro: io::Result<Operation> = serde_json::from_str(&l.unwrap())
                        .map_err(|e| std::io::Error::new(ErrorKind::Other, e));
                    ro
                })
                .chunks(100);

            let key = key_or_env(key);
            for structs in opstream {
                let structs: Vec<_> = structs.collect();
                let new_ops =
                    operations_to_point_operations(&resolved_domain, &store, structs, &key)
                        .await?
                        .0;
                hnsw = start_indexing_from_operations(hnsw, new_ops).unwrap();
            }
            let index_id = create_index_name(&domain, &commit);
            let filename = index_serialization_path(dirpath, &index_id);
            serialize_index(filename, hnsw.clone()).unwrap();
            */
        }
        Commands::Load2 {
            key,
            domain,
            directory,
            input,
            size,
            commit,
            model,
        } => {
            eprintln!("starting load");
            let key = key_or_env(key);
            index_from_operations_file(&key, model, input, directory, &domain, &commit, size)
                .await
                .unwrap()
        }
        Commands::TestRecall {
            domain,
            directory,
            size,
            commit,
        } => {
            eprintln!("Testing recall");
            let dirpath = Path::new(&directory);
            let hnsw_index_path = dbg!(format!(
                "{}/{}.hnsw",
                directory,
                create_index_name(&domain, &commit)
            ));
            let store = VectorStore::new(dirpath, size);
            let hnsw: HnswIndex = deserialize_index(hnsw_index_path, Arc::new(store))
                .unwrap()
                .unwrap();
            let mut rng = thread_rng();
            let num = hnsw.vector_count();
            let max = (0.001 * num as f32) as usize;
            let mut seen = HashSet::new();
            let vecs_to_find: Vec<VectorId> = (0..max)
                .map(|_| {
                    let vid: VectorId;
                    loop {
                        let v = VectorId(rng.gen_range(0..num));
                        if seen.insert(v) {
                            vid = v;
                            break;
                        }
                    }
                    vid
                })
                .collect();
            let relevant: usize = vecs_to_find
                .par_iter()
                .filter(|vid| {
                    let res = hnsw.search(AbstractVector::Stored(**vid), 3200, 2);
                    res.iter().map(|(v, _)| v).any(|v| v == *vid)
                })
                .count();
            let recall = relevant as f32 / max as f32;
            eprintln!("Recall: {recall}");
        }
        Commands::Diagnostics {
            domain,
            directory,
            size,
            commit,
            layer,
        } => {
            let dirpath = Path::new(&directory);
            let hnsw_index_path = dbg!(format!(
                "{}/{}.hnsw",
                directory,
                create_index_name(&domain, &commit)
            ));
            let store = VectorStore::new(dirpath, size);
            let hnsw: HnswIndex = deserialize_index(hnsw_index_path, Arc::new(store))
                .unwrap()
                .unwrap();

            let bottom_distances: Vec<NodeDistance> =
                hnsw.node_distances_for_layer(hnsw.layer_count() - 1 - layer);

            let mut bottom_distances: Vec<(NodeId, usize)> = bottom_distances
                .into_iter()
                .enumerate()
                .map(|(ix, d)| (NodeId(ix), d.index_sum))
                .collect();

            bottom_distances.sort_by_key(|(_, d)| usize::MAX - d);

            let unreachables: Vec<NodeId> = bottom_distances
                .iter()
                .take_while(|(_, d)| *d == !0)
                .map(|(n, _)| *n)
                .collect();

            eprintln!("unreachables: {}", unreachables.len());

            let mean = bottom_distances
                .iter()
                .skip(unreachables.len())
                .map(|(_, d)| d)
                .sum::<usize>() as f32
                / (bottom_distances.len() - unreachables.len()) as f32;
            eprintln!("mean: {mean}");

            let variance = bottom_distances
                .iter()
                .skip(unreachables.len())
                .map(|(_, d)| {
                    let d = *d as f32;
                    let diff = if d > mean { d - mean } else { mean - d };
                    diff * diff
                })
                .sum::<f32>()
                / (bottom_distances.len() - unreachables.len()) as f32;

            eprintln!("variance: {variance}");

            for x in bottom_distances.iter().skip(unreachables.len()).take(250) {
                eprintln!(" {} has distance {}", x.0 .0, x.1);
            }

            let mut clusters: Vec<(NodeId, Vec<(NodeId, usize)>)> = unreachables
                .par_iter()
                .map(|node| {
                    (
                        *node,
                        hnsw.reachables_from_node_for_layer(
                            hnsw.layer_count() - 1 - layer,
                            *node,
                            &unreachables[..],
                        ),
                    )
                })
                .collect();

            clusters.sort_by_key(|c| usize::MAX - c.1.len());

            for x in clusters.iter().take(250) {
                eprintln!(
                    " from {} we reach {} previously unreachables",
                    x.0 .0,
                    x.1.len()
                );
            }

            // take first from unreachables
            // figure out its neighbors and if they are also unreachables
            let mut cluster_queue: Vec<_> = clusters.iter().map(Some).collect();
            cluster_queue.reverse();
            let mut nodes_to_promote: Vec<NodeId> = Vec::new();
            while let Some(next) = cluster_queue.pop() {
                if let Some((nodeid, _)) = next {
                    nodes_to_promote.push(*nodeid);
                    for other in cluster_queue.iter_mut() {
                        if let Some((_, other_distances)) = other {
                            if other_distances.iter().any(|(n, _)| nodeid == n) {
                                *other = None
                            }
                        }
                    }
                }
            }
            eprintln!("Nodes to promote: {nodes_to_promote:?}");
        }
        Commands::Duplicates {
            commit,
            domain,
            size,
            take,
            directory,
            threshold,
        } => {
            let dirpath = Path::new(&directory);
            let hnsw_index_path = dbg!(format!(
                "{}/{}.hnsw",
                directory,
                create_index_name(&domain, &commit)
            ));
            let store = VectorStore::new(dirpath, size);
            let hnsw: HnswIndex = deserialize_index(hnsw_index_path, Arc::new(store))
                .unwrap()
                .unwrap();

            let elts = if let Some(take) = take {
                Either::Left(hnsw.threshold_nn(threshold, 2).take_any(take))
            } else {
                Either::Right(hnsw.threshold_nn(threshold, 2))
            };
            let stdout = std::io::stdout();
            elts.for_each(|(v, results)| {
                let mut cluster = Vec::new();
                for result in results.iter() {
                    let distance = result.1;
                    if distance < threshold {
                        cluster.push((result.0 .0, distance))
                    }
                }
                let cluster = serde_json::to_string(&cluster).unwrap();
                let mut lock = stdout.lock();
                writeln!(lock, "[{}, {}]", v.0, cluster).unwrap();
            });
        }
        Commands::ImproveIndex {
            commit,
            domain,
            directory,
            size,
            improve_neighbors,
            promote,
        } => {
            let dirpath = Path::new(&directory);
            let hnsw_index_path = dbg!(format!(
                "{}/{}.hnsw",
                directory,
                create_index_name(&domain, &commit)
            ));
            let store = VectorStore::new(dirpath, size);
            let mut hnsw: HnswIndex = deserialize_index(&hnsw_index_path, Arc::new(store))
                .unwrap()
                .unwrap();

            if let Some(threshold) = improve_neighbors {
                hnsw.improve_neighbors(threshold);
                // TODO should write to staging first
                hnsw.serialize(hnsw_index_path);
            } else {
                todo!();
            }
        }
    }

    Ok(())
}
