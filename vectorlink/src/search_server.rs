use std::{
    convert::Infallible,
    error::Error,
    fs::File,
    io::{self, BufRead, BufReader},
    net::{IpAddr, Ipv6Addr, SocketAddr},
    path::Path,
    sync::Arc,
};

use hyper::{
    service::{make_service_fn, service_fn},
    Body, Request, Response, Server,
};
use parallel_hnsw::{AbstractVector, Serializable};

use serde::Serialize;
use thiserror::Error;

use crate::{
    configuration::HnswConfiguration,
    indexer::create_index_name,
    openai::{self, EmbeddingError},
    server::{query_map, Operation},
    vectors::VectorStore,
};

#[derive(Debug, Error)]
pub enum SearchServerError {
    #[error("Embedding error")]
    EmbeddingError(#[from] EmbeddingError),
}

pub struct State {
    key: String,
    hnsw: HnswConfiguration,
    dict: Vec<String>,
}

fn ids_from_file<P: AsRef<Path>>(path: P) -> io::Result<Vec<String>> {
    let file = File::open(path)?;
    eprintln!("opened");
    let reader = BufReader::new(file);

    let mut result = Vec::new();
    for s in reader.lines() {
        let s = s?;
        let op: Operation = serde_json::from_str(&s)?;
        result.push(op.id().unwrap());
    }

    Ok(result)
}

#[derive(Serialize)]
struct MatchResult {
    id: String,
    distance: f32,
}

pub async fn handle_request(
    state: Arc<State>,
    request: Request<Body>,
) -> Result<Response<Body>, SearchServerError> {
    let query = query_map(request.uri());
    let Some(embedding_string) = query.get("string") else {
        return Ok(Response::builder()
            .status(400)
            .body("No embedding string specified".to_string().into())
            .unwrap());
    };

    let (embeddings, _) = openai::embeddings_for(
        &state.key,
        &[embedding_string.to_owned()],
        state.hnsw.model(),
    )
    .await?;
    let embedding = embeddings[0];

    let vec = AbstractVector::Unstored(&embedding);
    let results = state.hnsw.search(vec, 300, 2);
    let result: Vec<_> = results
        .into_iter()
        .map(|(v, d)| MatchResult {
            id: state.dict[v.0].to_string(),
            distance: d,
        })
        .collect();

    let response_string =
        serde_json::to_string(&result).expect("json serialization failed for MatchResult vec");

    Ok(Response::builder()
        .status(200)
        .header("Content-Type", "application/json")
        .body(response_string.into())
        .unwrap())
}

pub async fn serve(
    port: u16,
    operations_file: &str,
    directory: &str,
    commit: &str,
    domain: &str,
    size: usize,
    key: &str,
) -> Result<(), Box<dyn Error>> {
    let dirpath = Path::new(&directory);
    let hnsw_index_path = dbg!(format!(
        "{}/{}.hnsw",
        directory,
        create_index_name(&domain, &commit)
    ));
    let store = VectorStore::new(dirpath, size);
    let hnsw = HnswConfiguration::deserialize(&hnsw_index_path, Arc::new(store))?;

    println!("about to get ids from file {operations_file}");
    let dict = ids_from_file(operations_file)?;
    println!("got ids from file");

    let state = Arc::new(State {
        key: key.to_owned(),
        hnsw,
        dict,
    });

    let make_svc = make_service_fn(move |_connection| {
        let state2 = state.clone();
        async {
            Ok::<_, Infallible>(service_fn(move |req| {
                // this probably should respond with a response?
                handle_request(state2.clone(), req)
            }))
        }
    });

    let addr = SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), port);
    let server = Server::bind(&addr).serve(make_svc);
    server.await?;

    Ok(())
}
