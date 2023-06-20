use std::io::ErrorKind;
use std::path::Path;

use clap::CommandFactory;
use clap::{Parser, Subcommand, ValueEnum};
use hnsw::Hnsw;
use indexer::serialize_index;
use indexer::start_indexing_from_operations;
use indexer::Point;
use indexer::{operations_to_point_operations, OpenAI};
use server::Operation;
use space::Metric;
use std::fs::File;
use std::io::{self, BufRead};
use {
    indexer::{create_index_name, HnswIndex},
    vecmath::empty_embedding,
    vectors::VectorStore,
};
mod indexer;
mod openai;
mod server;
mod vecmath;
mod vectors;
use itertools::Itertools;

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
    Embed {
        #[arg(short, long)]
        key: Option<String>,
        #[arg(short, long)]
        string: String,
    },
    Compare {
        #[arg(short, long)]
        key: Option<String>,
        #[arg(long)]
        s1: String,
        #[arg(long)]
        s2: String,
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
    },
    Test {
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
        Commands::Embed { key, string } => {
            let v = openai::embeddings_for(&key_or_env(key), &[string]).await?;
            eprintln!("{:?}", v);
        }
        Commands::Compare { key, s1, s2 } => {
            let v = openai::embeddings_for(&key_or_env(key), &[s1, s2]).await?;
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
        } => {
            let v = openai::embeddings_for(&key_or_env(key), &[s1, s2]).await?;
            let p1 = &v[0];
            let p2 = &v[1];
            let distance = match variant {
                DistanceVariant::Default => vecmath::normalized_cosine_distance(p1, p2),
                DistanceVariant::Scalar => vecmath::normalized_cosine_distance_scalar(p1, p2),
                DistanceVariant::Simd => vecmath::normalized_cosine_distance_simd(p1, p2),
            };
            println!("distance: {}", distance);
        }
        Commands::Test { key } => {
            let v = openai::embeddings_for(
                &key_or_env(key),
                &[
                    "king".to_string(),
                    "man".to_string(),
                    "woman".to_string(),
                    "queen".to_string(),
                ],
            )
            .await?;
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
            let mut hnsw: HnswIndex = Hnsw::new(OpenAI);
            let store = VectorStore::new(dirpath, size);

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
                    operations_to_point_operations(&domain.clone(), &store, structs, &key).await;
                hnsw = start_indexing_from_operations(hnsw, new_ops).unwrap();
            }
            let index_id = create_index_name(&domain, &commit);
            serialize_index(dirpath.to_path_buf(), &index_id, hnsw.clone()).unwrap();
        }
    }

    Ok(())
}
