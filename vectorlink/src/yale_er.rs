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

use crate::{configuration::OpenAIHnsw, indexer::create_index_name, vectors::VectorStore};

pub async fn handle_request(
    request: Request<Body>,
) -> Result<Response<Body>, Box<dyn Error + Send + Sync>> {
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

    let make_svc = make_service_fn(move |connection| {
        async {
            Ok::<_, Infallible>(service_fn(move |req| {
                // this probably should respond with a response?
                handle_request(req)
            }))
        }
    });

    let addr = SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), port);
    let server = Server::bind(&addr).serve(make_svc);
    server.await?;

    Ok(())
}
