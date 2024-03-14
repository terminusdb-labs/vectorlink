#![feature(portable_simd)]

use std::collections::HashSet;
use std::io::ErrorKind;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
mod batch;
mod comparator;
mod configuration;
mod indexer;
mod openai;
mod server;
mod vecmath;
mod vectors;

mod yale_er;

use batch::index_from_operations_file;
use clap::CommandFactory;
use clap::{Parser, Subcommand, ValueEnum};
use comparator::Centroid32Comparator;
use comparator::OpenAIComparator;
use comparator::QuantizedComparator;
use configuration::HnswConfiguration;
//use hnsw::Hnsw;
use indexer::index_serialization_path;
use indexer::start_indexing_from_operations;
use indexer::Point;
use indexer::{operations_to_point_operations, OpenAI};
use itertools::Itertools;
use openai::Model;
use parallel_hnsw::pq::QuantizedHnsw;
use parallel_hnsw::pq::VectorSelector;
use parallel_hnsw::pq::{HnswQuantizer, Quantizer};
use parallel_hnsw::Comparator;
use parallel_hnsw::Serializable;
use parallel_hnsw::{AbstractVector, AllVectorIterator, Hnsw, NodeDistance, NodeId, VectorId};
use rand::thread_rng;
use rand::*;
use serde_json::json;
use server::Operation;
use space::Metric;
use std::fs::File;
use std::io::{self, BufRead};
use vecmath::Embedding;
use vecmath::QuantizedEmbedding;
use vecmath::CENTROID_32_LENGTH;
use vecmath::EMBEDDING_BYTE_LENGTH;
use vecmath::EMBEDDING_LENGTH;
use vecmath::QUANTIZED_EMBEDDING_LENGTH;

use rayon::iter::Either;
use rayon::prelude::*;

use crate::configuration::OpenAIHnsw;

use {indexer::create_index_name, vecmath::empty_embedding, vectors::VectorStore};

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
        #[arg(long)]
        build_index: Option<bool>,
        #[arg(short, long)]
        quantize_hnsw: bool,
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
        #[arg(short, long, default_value_t = 1.0)]
        proportion: f32,
    },
    ScanNeighbors {
        #[arg(short, long)]
        commit: String,
        #[arg(long)]
        domain: String,
        #[arg(long)]
        sequence_domain: String,
        #[arg(short, long)]
        directory: String,
        #[arg(short, long, default_value_t = 10000)]
        size: usize,
        #[arg(short, long, default_value_t = 1.0_f32)]
        threshold: f32,
    },
    TestQuantization {
        #[arg(short, long)]
        directory: String,
        #[arg(short, long)]
        commit: String,
        #[arg(long)]
        domain: String,
        #[arg(short, long, default_value_t = 10000)]
        size: usize,
    },
    YaleEr {
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
        #[arg(short, long)]
        operations_file: String,
        #[arg(short, long)]
        directory: String,
        #[arg(short, long)]
        commit: String,
        #[arg(long)]
        domain: String,
        #[arg(short, long, default_value_t = 10000)]
        size: usize,
        #[arg(short, long)]
        key: Option<String>,
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

impl HnswConfiguration {
    fn vector_count(&self) -> usize {
        match self {
            HnswConfiguration::QuantizedOpenAi(_model, q) => q.vector_count(),
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
            HnswConfiguration::UnquantizedOpenAi(_model, h) => {
                h.search(v, number_of_candidates, probe_depth)
            }
        }
    }

    pub fn improve_neighbors(&mut self, threshold: f32, recall: f32) {
        match self {
            HnswConfiguration::QuantizedOpenAi(_model, q) => q.improve_neighbors(threshold, recall),
            HnswConfiguration::UnquantizedOpenAi(_model, h) => {
                h.improve_neighbors(threshold, recall)
            }
        }
    }

    pub fn zero_neighborhood_size(&self) -> usize {
        match self {
            HnswConfiguration::QuantizedOpenAi(_model, q) => q.zero_neighborhood_size(),
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
            HnswConfiguration::UnquantizedOpenAi(_model, h) => {
                Either::Right(h.threshold_nn(threshold, probe_depth, initial_search_depth))
            }
        }
    }
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
            build_index,
            quantize_hnsw,
        } => {
            eprintln!("starting load");
            let key = key_or_env(key);
            index_from_operations_file(
                &key,
                model,
                input,
                directory,
                &domain,
                &commit,
                size,
                build_index.unwrap_or(true),
                quantize_hnsw,
            )
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
            let hnsw = HnswConfiguration::deserialize(hnsw_index_path, Arc::new(store)).unwrap();
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
            let hnsw = HnswConfiguration::deserialize(hnsw_index_path, Arc::new(store)).unwrap();

            let initial_search_depth = 3 * hnsw.zero_neighborhood_size();
            let elts = if let Some(take) = take {
                Either::Left(
                    hnsw.threshold_nn(threshold, 2, initial_search_depth)
                        .take_any(take),
                )
            } else {
                Either::Right(hnsw.threshold_nn(threshold, 2, initial_search_depth))
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
            proportion,
        } => {
            let dirpath = Path::new(&directory);
            let hnsw_index_path = dbg!(format!(
                "{}/{}.hnsw",
                directory,
                create_index_name(&domain, &commit)
            ));
            let store = VectorStore::new(dirpath, size);

            if let Some(threshold) = improve_neighbors {
                let mut hnsw: HnswConfiguration =
                    HnswConfiguration::deserialize(&hnsw_index_path, Arc::new(store)).unwrap();

                hnsw.improve_neighbors(threshold, proportion);

                // TODO should write to staging first
                hnsw.serialize(hnsw_index_path)?;
            } else {
                todo!();
            }
        }
        Commands::ScanNeighbors {
            commit,
            domain,
            sequence_domain,
            directory,
            size,
            threshold,
        } => {
            let dirpath = Path::new(&directory);
            let hnsw_index_path = dbg!(format!(
                "{}/{}.hnsw",
                directory,
                create_index_name(&domain, &commit)
            ));
            let store = VectorStore::new(dirpath, size);
            let hnsw = HnswConfiguration::deserialize(&hnsw_index_path, Arc::new(store)).unwrap();

            let mut sequence_path = PathBuf::from(directory);
            sequence_path.push(format!("{sequence_domain}.vecs"));
            let mut embedding = [0; EMBEDDING_BYTE_LENGTH];
            let mut sequence_file = File::open(sequence_path).unwrap();
            let mut sequence_index = 0; // todo file offsetting etc
            let output = std::io::stdout();
            loop {
                match sequence_file.read_exact(&mut embedding) {
                    Ok(()) => {
                        let converted_embedding: &[f32; EMBEDDING_LENGTH] =
                            unsafe { std::mem::transmute(&embedding) };
                        let search_result: Vec<_> = hnsw
                            .search(AbstractVector::Unstored(converted_embedding), 300, 1)
                            .into_iter()
                            .filter(|r| r.1 < threshold)
                            .map(|r| (r.0 .0, r.1))
                            .collect();
                        let result_tuple = (sequence_index, search_result);
                        {
                            let mut lock = output.lock();
                            serde_json::to_writer(&mut lock, &result_tuple).unwrap();
                            write!(&mut lock, "\n").unwrap();
                        }

                        // do index lookup stuff
                        sequence_index += 1;
                    }
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                        break;
                    }
                    Err(e) => {
                        panic!("error occured while processing sequence vector file: {}", e);
                    }
                }
            }
        }
        Commands::TestQuantization {
            commit,
            domain,
            directory,
            size,
        } => {
            let dirpath = Path::new(&directory);
            let hnsw_index_path = dbg!(format!(
                "{}/{}.hnsw",
                directory,
                create_index_name(&domain, &commit)
            ));
            let store = VectorStore::new(dirpath, size);
            let hnsw = HnswConfiguration::deserialize(&hnsw_index_path, Arc::new(store)).unwrap();
            if let HnswConfiguration::QuantizedOpenAi(_, hnsw) = hnsw {
                let c = hnsw.quantized_comparator();
                let quantized_vecs = c.data.read().unwrap();
                let mut cursor: &[QuantizedEmbedding] = &quantized_vecs;
                let quantizer = hnsw.quantizer();
                // sample_avg = sum(errors)/|errors|
                // sample_var = sum((error - sample_avg)^2)/|errors|

                let fc = hnsw.full_comparator();

                let mut errors = vec![0.0_f32; hnsw.vector_count()];

                let mut offset = 0;
                for chunk in fc.vector_chunks() {
                    let len = chunk.len();
                    let quantized_chunk = &cursor[..len];
                    cursor = &cursor[len..];

                    chunk
                        .into_par_iter()
                        .zip(quantized_chunk.into_par_iter())
                        .map(|(full_vec, quantized_vec)| {
                            let reconstructed = quantizer.reconstruct(&quantized_vec);

                            fc.compare_raw(&full_vec, &reconstructed)
                        })
                        .enumerate()
                        .for_each(|(ix, distance)| unsafe {
                            let ptr = errors.as_ptr().offset((offset + ix) as isize) as *mut f32;
                            *ptr = distance;
                        });

                    offset += len;
                }

                let sample_avg: f32 = errors.iter().sum::<f32>() / errors.len() as f32;
                let sample_var = errors
                    .iter()
                    .map(|e| (e - sample_avg))
                    .map(|x| x * x)
                    .sum::<f32>()
                    / errors.len() as f32;
                let sample_deviation = sample_var.sqrt();

                eprintln!("sample avg: {sample_avg}\nsample var: {sample_var}\nsample deviation: {sample_deviation}");
            } else {
                panic!("not a pq hnsw index");
            }
        }
        Commands::YaleEr {
            port,
            operations_file,
            directory,
            commit,
            domain,
            size,
            key,
        } => {
            let key = key_or_env(key);
            yale_er::serve(
                port,
                &operations_file,
                &directory,
                &commit,
                &domain,
                size,
                &key,
            )
            .await
            .unwrap()
        }
    }

    Ok(())
}
