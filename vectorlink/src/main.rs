#![feature(portable_simd)]
#![feature(trait_upcasting)]

use std::io::Read;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
mod batch;
mod comparator;
mod configuration;
mod domain;
mod indexer;
mod openai;
mod server;
mod store;
mod utils;
mod vecmath;
mod vectors;

mod search_server;

use batch::index_from_operations_file;
use clap::CommandFactory;
use clap::{Parser, Subcommand, ValueEnum};
use configuration::HnswConfiguration;
//use hnsw::Hnsw;
use openai::Model;
use parallel_hnsw::pq::Quantizer;
use parallel_hnsw::pq::VectorSelector;
use parallel_hnsw::AbstractVector;
use parallel_hnsw::Comparator;
use parallel_hnsw::Serializable;
use rand::prelude::*;
use std::fs::File;
use std::io;
use store::VectorFile;
use vecmath::Quantized32Embedding;
use vecmath::EMBEDDING_BYTE_LENGTH;
use vecmath::EMBEDDING_LENGTH;

use rayon::iter::Either;
use rayon::prelude::*;

use crate::utils::{test_quantization, QuantizationStatistics};
use crate::vecmath::Quantized16Embedding;

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
        #[arg(short, long, default_value_t = 0.001)]
        recall_proportion: f32,
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
    SearchServer {
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
    Scramble {
        #[arg(short, long)]
        vec_file: String,
        #[arg(short, long)]
        output_vecs: String,
        #[arg(short, long)]
        output_map: String,
        #[arg(short, long)]
        vector_size: usize,
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
        Commands::Compare { .. } => {
            todo!();
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
            recall_proportion,
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
            let recall = hnsw.stochastic_recall(recall_proportion);
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
            promote: _promote,
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
                            writeln!(&mut lock).unwrap();
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
            let hnsw = HnswConfiguration::deserialize(hnsw_index_path, Arc::new(store)).unwrap();
            let QuantizationStatistics {
                sample_avg,
                sample_var,
                sample_deviation,
            } = match hnsw {
                HnswConfiguration::QuantizedOpenAi(_, q) => test_quantization(&q),
                HnswConfiguration::SmallQuantizedOpenAi(_, q) => test_quantization(&q),
                HnswConfiguration::SmallQuantizedOpenAi8(_, q) => test_quantization(&q),
                HnswConfiguration::SmallQuantizedOpenAi4(_, q) => test_quantization(&q),
                HnswConfiguration::UnquantizedOpenAi(_, _) => panic!("not a quantized hnsw"),
            };
            eprintln!("sample avg: {sample_avg}\nsample var: {sample_var}\nsample deviation: {sample_deviation}");
        }
        Commands::SearchServer {
            port,
            operations_file,
            directory,
            commit,
            domain,
            size,
            key,
        } => {
            let key = key_or_env(key);
            search_server::serve(
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
        Commands::Scramble {
            vec_file,
            output_vecs,
            output_map,
            vector_size,
        } => {
            let vec_file = File::open(vec_file).unwrap();
            let mut output_vecs = File::create(output_vecs).unwrap();

            let byte_size = vec_file.metadata().unwrap().size() as usize;
            assert!(byte_size % vector_size == 0);
            let vector_byte_size = vector_size * std::mem::size_of::<f32>();
            let number_of_vecs = byte_size / vector_byte_size;

            let mut remap: Vec<usize> = (0..number_of_vecs).collect();
            remap.shuffle(&mut thread_rng());

            let mut buf = vec![0; vector_byte_size];
            for mapping in remap.iter() {
                let byte_offset = mapping * vector_size;
                vec_file.read_at(&mut buf, byte_offset as u64).unwrap();
                output_vecs.write_all(&buf).unwrap();
            }

            let remap_buf = unsafe {
                std::slice::from_raw_parts(
                    remap.as_ptr() as *const u8,
                    number_of_vecs * std::mem::size_of::<usize>(),
                )
            };

            std::fs::write(output_map, &remap_buf).unwrap();
        }
    }

    Ok(())
}
