use bytes::Bytes;
use hyper::{
    service::{make_service_fn, service_fn},
    Body, Method, Request, Response, Server, Uri,
};
use lazy_static::lazy_static;
use regex::Regex;
use serde::{self, Deserialize};
use std::{
    convert::Infallible,
    net::{IpAddr, Ipv6Addr, SocketAddr},
    path::PathBuf,
    sync::Arc,
};
//use serde_json;

#[derive(Deserialize, Debug)]
#[serde(tag = "op")]
enum Operation {
    Inserted { string: String, id: String },
    Changed { string: String, id: String },
    Deleted { id: String },
}

#[derive(Deserialize, Debug)]
struct IndexRequest {
    commit_id: String,
    operations: Vec<Operation>,
}

enum ResourceSpec {
    IndexRequest,
}

enum SpecParseError {
    UnknownPath,
}

fn uri_to_spec(uri: &Uri) -> Result<ResourceSpec, SpecParseError> {
    lazy_static! {
        static ref RE_INDEX: Regex = Regex::new(r"^/index/$").unwrap();
    }
    let path = uri.path();

    if RE_INDEX.is_match(path) {
        Ok(ResourceSpec::IndexRequest)
    } else {
        Err(SpecParseError::UnknownPath)
    }
}

#[derive(Clone)]
struct Service {}

async fn extract_body(req: Request<Body>) -> Bytes {
    hyper::body::to_bytes(req.into_body()).await.unwrap()
}

impl Service {
    fn new<P: Into<PathBuf>>(_: P) -> Self {
        Service {}
    }

    async fn serve(&self, req: Request<Body>) -> Result<Response<Body>, Infallible> {
        match req.method() {
            &Method::POST => self.post(req).await,
            _ => todo!(),
        }
    }

    async fn post(&self, req: Request<Body>) -> Result<Response<Body>, Infallible> {
        let uri = req.uri();
        match uri_to_spec(uri) {
            Ok(ResourceSpec::IndexRequest) => {
                let body = extract_body(req).await;
                let operations: Result<IndexRequest, _> = serde_json::from_slice(&body);
                match operations {
                    Ok(indexrequest) => Ok(Response::builder()
                        .body(format!("Hello {:?}!", indexrequest).into())
                        .unwrap()),
                    Err(_error) => todo!(),
                }
            }
            Err(_) => todo!(),
        }
    }
}

pub async fn serve<P: Into<PathBuf>>(
    directory: P,
    port: u16,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), port);
    let service = Arc::new(Service::new(directory));

    let make_svc = make_service_fn(move |_conn| {
        let s = service.clone();
        async {
            Ok::<_, Infallible>(service_fn(move |req| {
                let s = s.clone();
                async move { s.serve(req).await }
            }))
        }
    });

    let server = Server::bind(&addr).serve(make_svc);
    server.await?;

    Ok(())
}
