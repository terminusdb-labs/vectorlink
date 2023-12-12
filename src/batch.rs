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
};

#[derive(Error, Debug)]
pub enum VectorizationError {
    #[error(transparent)]
    EmbeddingError(#[from] EmbeddingError),
    #[error(transparent)]
    Io(#[from] io::Error),
}

async fn strings_to_vecs(
    api_key: &str,
    vec_file: &mut File,
    offset: usize,
    strings: &[String],
) -> Result<usize, VectorizationError> {
    vec_file.seek(SeekFrom::Start(offset as u64)).await?;

    let (embeddings, failures) = embeddings_for(api_key, strings).await?;
    let transmuted = unsafe {
        std::slice::from_raw_parts(embeddings.as_ptr() as *const u8, embeddings.len() * 4)
    };
    vec_file.write_all(transmuted).await?;
    vec_file.flush().await?;
    vec_file.sync_data().await?;

    Ok(failures)
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

    let mut filtered_op_stream = pin!(op_stream
        .try_filter(|o| future::ready(o.has_string()))
        .skip(offset as usize)
        .chunks(100));

    let mut failures = 0;
    eprintln!("starting indexing at {offset}");
    #[allow(for_loops_over_fallibles)]
    for chunk in filtered_op_stream.next().await {
        let chunk: Result<Vec<String>, _> = chunk
            .into_iter()
            .map(|o| o.map(|o| o.string().unwrap()))
            .collect();
        let chunk = chunk?;

        failures += strings_to_vecs(api_key, vec_file, offset as usize, &chunk).await?;
        offset += chunk.len() as u64;
        progress_file.seek(SeekFrom::Start(0)).await?;
        progress_file.write_u64(offset).await?;
        progress_file.flush().await?;
        progress_file.sync_data().await?;
        eprintln!("indexed {offset}");
    }

    Ok(failures)
}

async fn get_operations_from_file<'a>(
    file: &'a mut File,
) -> io::Result<impl Stream<Item = io::Result<Operation>> + 'a> {
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
