use std::{
    io::{self, SeekFrom},
    os::unix::prelude::MetadataExt,
    path::{Path, PathBuf},
    pin::pin,
};

use futures::{future, Stream, StreamExt, TryStreamExt};
use thiserror::Error;
use tokio::{
    fs::{File, OpenOptions},
    io::{AsyncBufReadExt, AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader},
};
use tokio_stream::wrappers::LinesStream;

use crate::{
    openai::{embeddings_for, EmbeddingError},
    server::Operation,
    vecmath::Embedding,
};

#[derive(Error, Debug)]
pub enum VectorizationError {
    #[error(transparent)]
    EmbeddingError(#[from] EmbeddingError),
    #[error(transparent)]
    Io(#[from] io::Error),
}

async fn save_embeddings(
    vec_file: &mut File,
    offset: usize,
    embeddings: &[Embedding],
) -> Result<(), VectorizationError> {
    let transmuted = unsafe {
        std::slice::from_raw_parts(
            embeddings.as_ptr() as *const u8,
            std::mem::size_of_val(embeddings),
        )
    };
    vec_file
        .seek(SeekFrom::Start(
            (offset * std::mem::size_of::<Embedding>()) as u64,
        ))
        .await?;
    vec_file.write_all(transmuted).await?;
    vec_file.flush().await?;
    vec_file.sync_data().await?;

    Ok(())
}

pub async fn vectorize_from_operations<
    S: Stream<Item = io::Result<Operation>>,
    P: AsRef<Path> + Unpin,
>(
    api_key: &str,
    vec_file: &mut File,
    op_stream: S,
    progress_file_path: P,
) -> Result<usize, VectorizationError> {
    let mut progress_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(progress_file_path)
        .await?;
    let mut offset;
    if progress_file.metadata().await?.size() != 8 {
        // assume we have to start from scratch
        progress_file.write_u64(0).await?;
        offset = 0;
    } else {
        offset = progress_file.read_u64().await?;
    }

    let filtered_op_stream = pin!(op_stream
        .try_filter(|o| future::ready(o.has_string()))
        .skip(offset as usize)
        .chunks(100));
    let mut taskstream = filtered_op_stream
        .map(|chunk| {
            let inner_api_key = api_key.to_string();
            tokio::spawn(async move { chunk_to_embeds(inner_api_key, chunk).await })
        })
        .buffered(10);

    let mut failures = 0;
    eprintln!("starting indexing at {offset}");
    while let Some(embeds) = taskstream.next().await {
        eprintln!("start of loop");
        let (embeddings, chunk_failures) = embeds.unwrap()?;
        eprintln!("retrieved embeddings");

        save_embeddings(vec_file, offset as usize, &embeddings).await?;
        eprintln!("saved embeddings");
        failures += chunk_failures;
        offset += embeddings.len() as u64;
        progress_file.seek(SeekFrom::Start(0)).await?;
        progress_file.write_u64(offset).await?;
        progress_file.flush().await?;
        progress_file.sync_data().await?;
        eprintln!("indexed {offset}");
    }

    Ok(failures)
}

async fn chunk_to_embeds(
    api_key: String,
    chunk: Vec<Result<Operation, io::Error>>,
) -> Result<(Vec<Embedding>, usize), VectorizationError> {
    let chunk: Result<Vec<String>, _> = chunk
        .into_iter()
        .map(|o| o.map(|o| o.string().unwrap()))
        .collect();
    let chunk = chunk?;

    Ok(embeddings_for(&api_key, &chunk).await?)
}

async fn get_operations_from_file(
    file: &mut File,
) -> io::Result<impl Stream<Item = io::Result<Operation>> + '_> {
    file.seek(SeekFrom::Start(0)).await?;

    let buf_reader = BufReader::new(file);
    let lines = buf_reader.lines();
    let lines_stream = LinesStream::new(lines);
    let stream = lines_stream.and_then(|l| {
        future::ready(serde_json::from_str(&l).map_err(|e| io::Error::new(io::ErrorKind::Other, e)))
    });

    Ok(stream)
}

pub async fn index_from_operations_file<P: AsRef<Path>>(
    api_key: &str,
    op_file_path: P,
    vectorlink_path: P,
    domain: &str,
) -> Result<(), VectorizationError> {
    let mut staging_path: PathBuf = vectorlink_path.as_ref().into();
    staging_path.push(".staging");
    staging_path.push(domain);
    tokio::fs::create_dir_all(&staging_path).await?;

    let mut vector_path = staging_path.clone();
    vector_path.push("vectors");
    let mut vec_file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(&vector_path)
        .await?;
    let mut progress_file_path = staging_path.clone();
    progress_file_path.push("progress");

    let mut op_file = File::open(op_file_path).await?;
    let op_stream = get_operations_from_file(&mut op_file).await?;

    vectorize_from_operations(api_key, &mut vec_file, op_stream, progress_file_path).await?;

    Ok(())
}
