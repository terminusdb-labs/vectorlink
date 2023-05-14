use clap::{Parser, Subcommand, ValueEnum};
use indexer::Point;
use space::Metric;

use crate::indexer::OpenAI;

mod indexer;
mod openai;
mod server;
mod vectors;
mod vecmath;

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
        directory: String,
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
        #[arg(short, long, default_value_t = 10000)]
        size: usize,
    },
    Embed {
        #[arg(short, long)]
        key: String,
        #[arg(short, long)]
        string: String,
    },
    Compare {
        #[arg(short, long)]
        key: String,
        #[arg(long)]
        s1: String,
        #[arg(long)]
        s2: String,
    },
    Compare2 {
        #[arg(short, long)]
        key: String,
        #[arg(long)]
        s1: String,
        #[arg(long)]
        s2: String,
        #[arg(short, long, value_enum, default_value_t=DistanceVariant::Default)]
        variant: DistanceVariant
    },
    Test,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DistanceVariant {
    Default,
    Simd,
    Cpu
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    match args.command {
        Commands::Serve {
            directory,
            port,
            size,
        } => server::serve(directory, port, size).await?,
        Commands::Embed { key, string } => {
            let v = openai::embeddings_for(&key, &[string]).await?;
            eprintln!("{:?}", v);
        }
        Commands::Compare { key, s1, s2 } => {
            let v = openai::embeddings_for(&key, &[s1, s2]).await?;
            let p1 = Point::Mem {
                vec: Box::new(v[0]),
            };
            let p2 = Point::Mem {
                vec: Box::new(v[1]),
            };
            println!(
                "same? {}, distance: {}",
                p1 == p2,
                OpenAI.distance(&p1, &p2)
            );
        },
        Commands::Compare2 { key, s1, s2, variant } => {
            let v = openai::embeddings_for(&key, &[s1, s2]).await?;
            let p1 = &v[0];
            let p2 = &v[1];
            let distance = match variant {
                DistanceVariant::Default => vecmath::normalized_cosine_distance(&p1, &p2),
                DistanceVariant::Cpu => vecmath::normalized_cosine_distance_cpu(&p1, &p2),
                DistanceVariant::Simd => vecmath::normalized_cosine_distance_simd(&p1, &p2),
            };
            println!(
                "distance: {}",
                distance
            );
        }
        Commands::Test => {
            eprintln!("{}", 1.0_f32.to_bits());
        }
    }

    Ok(())
}
