use std::{
    io::{self, SeekFrom},
    os::unix::prelude::MetadataExt,
    path::{Path, PathBuf},
    pin::pin,
    sync::Arc,
};

use futures::{future, Stream, StreamExt, TryStreamExt};
use parallel_hnsw::Serializable;
use parallel_hnsw::{pq::QuantizedHnsw, SerializationError};
use parallel_hnsw::{Hnsw, VectorId};
use thiserror::Error;
use tokio::{
    fs::{File, OpenOptions},
    io::{AsyncBufReadExt, AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader},
};
use tokio_stream::wrappers::LinesStream;
use urlencoding::encode;

use crate::{
    comparator::{
        Centroid16Comparator, DiskOpenAIComparator, OpenAIComparator, Quantized16Comparator,
    },
    configuration::HnswConfiguration,
    indexer::{create_index_name, index_serialization_path},
    openai::{embeddings_for, EmbeddingError, Model},
    server::Operation,
    vecmath::{Embedding, CENTROID_16_LENGTH, EMBEDDING_LENGTH, QUANTIZED_16_EMBEDDING_LENGTH},
    vectors::VectorStore,
};

#[derive(Error, Debug)]
pub enum BatchError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    VectorizationError(#[from] VectorizationError),
    #[error(transparent)]
    IndexingError(#[from] IndexingError),
}

#[derive(Error, Debug)]
pub enum IndexingError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    SerializationError(#[from] SerializationError),
}

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
    model: Model,
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
            tokio::spawn(async move { chunk_to_embeds(inner_api_key, chunk, model).await })
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
    model: Model,
) -> Result<(Vec<Embedding>, usize), VectorizationError> {
    let chunk: Result<Vec<String>, _> = chunk
        .into_iter()
        .map(|o| o.map(|o| o.string().unwrap()))
        .collect();
    let chunk = chunk?;

    Ok(embeddings_for(&api_key, &chunk, model).await?)
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

pub async fn extend_vector_store<P0: AsRef<Path>, P1: AsRef<Path>>(
    domain: &str,
    vectorlink_path: P0,
    vec_path: P1,
    size: usize,
) -> Result<usize, io::Error> {
    let vs_path: PathBuf = vectorlink_path.as_ref().into();
    let vs: VectorStore = VectorStore::new(vs_path, size);
    let domain = vs.get_domain(domain)?;
    domain.concatenate_file(&vec_path)
}

const NUMBER_OF_CENTROIDS: usize = 10_000;
pub async fn index_using_operations_and_vectors<
    P0: AsRef<Path>,
    P1: AsRef<Path>,
    P2: AsRef<Path>,
>(
    domain: &str,
    commit: &str,
    vectorlink_path: P0,
    staging_path: P1,
    op_file_path: P2,
    size: usize,
    id_offset: u64,
    quantize_hnsw: bool,
    model: Model,
) -> Result<(), IndexingError> {
    // Start at last hnsw offset
    let mut progress_file_path: PathBuf = staging_path.as_ref().into();
    progress_file_path.push("index_progress");

    let offset: u64;
    let mut progress_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(progress_file_path)
        .await?;
    if progress_file.metadata().await?.size() != 8 {
        // assume we have to start from scratch
        progress_file.write_u64(id_offset).await?;
        offset = id_offset;
    } else {
        offset = progress_file.read_u64().await?;
    }

    // Start filling the HNSW
    let vs_path_buf: PathBuf = vectorlink_path.as_ref().into();
    let vs: VectorStore = VectorStore::new(&vs_path_buf, size);
    //    let index_id = create_index_name(domain, commit);
    let domain_obj = vs.get_domain(domain)?;
    let mut op_file = File::open(&op_file_path).await?;
    let mut op_stream = get_operations_from_file(&mut op_file).await?;
    let mut i: usize = 0;

    let index_file_name = "index";
    //    let temp_file = index_serialization_path(&staging_path, temp_file_name);
    let staging_file = index_serialization_path(&staging_path, index_file_name);
    let index_name = create_index_name(domain, commit);
    let final_file = index_serialization_path(&vectorlink_path, &index_name);
    /*
    let mut hnsw: HnswIndex;
    if let Some(index) = deserialize_index(&staging_file, &domain_obj, &index_id, &vs)? {
        hnsw = index;
    } else {
        hnsw = deserialize_index(&final_file, &domain_obj, &index_id, &vs)?
            .unwrap_or_else(|| HnswIndex::new(OpenAI));
    }*/
    while let Some(op) = op_stream.next().await {
        match op.unwrap() {
            Operation::Inserted { .. } => i += 1,
            Operation::Changed { .. } => {
                todo!()
            }
            Operation::Deleted { .. } => {
                todo!()
            }
            Operation::Error { message } => {
                panic!("Error in indexing {message}");
            }
        }
    }
    let comparator = OpenAIComparator::new(
        domain_obj.name().to_string(),
        Arc::new(domain_obj.all_vecs()?),
    );
    let vecs: Vec<_> = (offset as usize..(offset as usize + i))
        .map(VectorId)
        .collect();

    eprintln!("ready to generate hnsw");
    let hnsw = if quantize_hnsw {
        let number_of_vectors = NUMBER_OF_CENTROIDS / 10;
        let c = DiskOpenAIComparator::new(domain_obj);
        let hnsw: QuantizedHnsw<
            EMBEDDING_LENGTH,
            CENTROID_16_LENGTH,
            QUANTIZED_16_EMBEDDING_LENGTH,
            Centroid16Comparator,
            Quantized16Comparator,
            DiskOpenAIComparator,
        > = QuantizedHnsw::new(number_of_vectors, c);
        HnswConfiguration::SmallQuantizedOpenAi(model, hnsw)
    } else {
        let hnsw = Hnsw::generate(comparator, vecs, 24, 48, 12);
        HnswConfiguration::UnquantizedOpenAi(model, hnsw)
    };
    eprintln!("done generating hnsw");
    hnsw.serialize(&staging_file)?;
    eprintln!("done serializing hnsw");
    eprintln!("renaming {staging_file:?} to {final_file:?}");
    tokio::fs::rename(&staging_file, &final_file).await?;
    eprintln!("renamed hnsw");
    Ok(())
}

pub async fn index_from_operations_file<P: AsRef<Path>>(
    api_key: &str,
    model: Model,
    op_file_path: P,
    vectorlink_path: P,
    domain: &str,
    commit: &str,
    size: usize,
    build_index: bool,
    quantize_hnsw: bool,
) -> Result<(), BatchError> {
    let mut staging_path: PathBuf = vectorlink_path.as_ref().into();
    staging_path.push(".staging");
    staging_path.push(&*encode(domain));
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

    let mut op_file = File::open(&op_file_path).await?;
    let op_stream = get_operations_from_file(&mut op_file).await?;

    vectorize_from_operations(api_key, model, &mut vec_file, op_stream, progress_file_path).await?;

    // first append vectors in bulk
    let mut extended_path: PathBuf = staging_path.clone();
    extended_path.push("vectors_extended");
    let mut extended_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(extended_path)
        .await?;
    let id_offset: u64;
    if extended_file.metadata().await?.size() != 8 {
        eprintln!("Concatenating to vector store");
        id_offset = extend_vector_store(domain, &vectorlink_path, vector_path, size).await? as u64;
        extended_file.write_u64(id_offset).await?;
    } else {
        eprintln!("Already concatenated");
        id_offset = extended_file.read_u64().await?;
    }

    if build_index {
        index_using_operations_and_vectors(
            domain,
            commit,
            vectorlink_path,
            staging_path,
            op_file_path,
            size,
            id_offset,
            quantize_hnsw,
            model,
        )
        .await?;
    } else {
        eprintln!("No index built");
    }
    Ok(())
}
