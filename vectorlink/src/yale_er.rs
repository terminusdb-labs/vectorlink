use std::{
    convert::Infallible,
    error::Error,
    io,
    net::{IpAddr, Ipv6Addr, SocketAddr},
    path::Path,
    sync::Arc,
};

use hyper::{
    service::{make_service_fn, service_fn},
    Body, Request, Response, Server,
};
use reqwest::ResponseBuilderExt;
use thiserror::Error;

use crate::{
    configuration::OpenAIHnsw,
    indexer::create_index_name,
    openai::{self, EmbeddingError},
    server::query_map,
    vectors::VectorStore,
};

#[derive(Debug, Error)]
pub enum YaleError {
    #[error("No embedding string specified")]
    NoEmbeddingStringSpecified,
    #[error("Embedding error")]
    EmbeddingError(#[from] EmbeddingError),
}

struct State {
    key: String,
    hnsw: OpenAIHnsw,
}

pub async fn handle_request(
    state: Arc<State>,
    request: Request<Body>,
) -> Result<Response<Body>, YaleError> {
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
        openai::Model::Small3,
    )
    .await?;
    let embedding = embeddings[0];

    //state.hnsw.search(v, number_of_candidates, probe_depth)

    Ok(Response::builder()
        .status(200)
        .body("Hello".to_string().into())
        .unwrap())
}

pub async fn serve(
    port: u16,
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
    let hnsw = OpenAIHnsw::deserialize(&hnsw_index_path, Arc::new(store))?;

    let state = Arc::new(State {
        key: key.to_owned(),
        hnsw,
    });

    let make_svc = make_service_fn(move |connection| {
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
