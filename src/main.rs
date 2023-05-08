use clap::{Parser, Subcommand};

mod server;
mod openai;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands
}

#[derive(Subcommand, Debug)]
enum Commands {
    Serve {
        #[arg(short, long)]
        directory: String,
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
    },
    Embed {
        #[arg(short, long)]
        key: String,
        #[arg(short, long)]
        string: String
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    match args.command {
        Commands::Serve{ directory, port } => server::serve(directory, port).await?,
        Commands::Embed { key, string } => {
            openai::embeddings_for(&key, &[&string]).await?;
        }
    }

    Ok(())
}
