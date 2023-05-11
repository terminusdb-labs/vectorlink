use clap::{Parser, Subcommand};

mod indexer;
mod openai;
mod server;
mod vectors;

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
        #[arg(short, long, default_value_t = 100)]
        size: usize,
    },
    Embed {
        #[arg(short, long)]
        key: String,
        #[arg(short, long)]
        string: String,
    },
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
    }

    Ok(())
}
