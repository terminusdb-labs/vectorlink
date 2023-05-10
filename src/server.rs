use bytes::Bytes;
use futures::TryStreamExt;
use hnsw::Hnsw;
use hyper::{
    service::{make_service_fn, service_fn},
    Body, Method, Request, Response, Server, Uri,
};
use lazy_static::lazy_static;
use rand::distributions::Alphanumeric;
use rand::Rng;
use regex::Regex;
use reqwest::Url;
use serde::{self, Deserialize};
use std::{
    collections::HashMap,
    convert::Infallible,
    net::{IpAddr, Ipv6Addr, SocketAddr},
    path::PathBuf,
    sync::Arc,
};
use std::{
    future,
    io::{self, ErrorKind},
};
use tokio::io::AsyncBufReadExt;
use tokio_stream::{wrappers::LinesStream, Stream};
use tokio_util::io::StreamReader;

use crate::indexer::{start_indexing_from_operations, HnswIndex, IndexIdentifier, OpenAI};

#[derive(Deserialize, Debug)]
#[serde(tag = "op")]
pub enum Operation {
    Inserted { string: String, id: String },
    Changed { string: String, id: String },
    Deleted { id: String },
}

#[derive(Deserialize, Debug)]
struct IndexRequest {
    domain: String,
    commit: String,
    previous: Option<String>,
    operations: Vec<Operation>,
}

enum ResourceSpec {
    IndexRequest,
    StartIndex {
        domain: String,
        commit: String,
        previous: Option<String>,
    },
    CheckTask {
        task_id: String,
    },
}

enum SpecParseError {
    UnknownPath,
    NoTaskId,
    NoDomain,
    NoTaskIdOrDomain,
}

fn uri_to_spec(uri: &Uri) -> Result<ResourceSpec, SpecParseError> {
    lazy_static! {
        static ref RE_INDEX: Regex = Regex::new(r"^/index(/?)$").unwrap();
        static ref RE_START: Regex = Regex::new(r"^/start(/?)$").unwrap();
        static ref RE_CHECK: Regex = Regex::new(r"^/check(/?)$").unwrap();
    }
    let path = uri.path();

    if RE_INDEX.is_match(path) {
        Ok(ResourceSpec::IndexRequest)
    } else if RE_START.is_match(path) {
        let uri_string = uri.to_string();
        let request_url = Url::parse(&uri_string).unwrap();
        let params = request_url.query_pairs();
        let mut commit = None;
        let mut domain = None;
        let mut previous = None;
        for (key, value) in params {
            if key == "commit" {
                commit = Some(value.to_string())
            } else if key == "domain" {
                domain = Some(value.to_string())
            } else if key == "previous" {
                previous = Some(value.to_string())
            }
        }
        match (domain, commit) {
            (Some(domain), Some(commit)) => Ok(ResourceSpec::StartIndex {
                domain,
                commit,
                previous,
            }),
            (Some(_domain), None) => Err(SpecParseError::NoTaskId),
            (None, Some(_commit)) => Err(SpecParseError::NoDomain),
            (None, None) => Err(SpecParseError::NoTaskIdOrDomain),
        }
    } else if RE_CHECK.is_match(path) {
        let uri_string = uri.to_string();
        let request_url = Url::parse(&uri_string).unwrap();
        let params = request_url.query_pairs();
        for (key, task_id) in params {
            if key == "task_id" {
                return Ok(ResourceSpec::CheckTask {
                    task_id: task_id.to_string(),
                });
            }
        }
        Err(SpecParseError::NoTaskId)
    } else {
        Err(SpecParseError::UnknownPath)
    }
}

#[derive(Clone, Debug)]
pub enum TaskStatus {
    Pending,
    Error,
    Completed,
}

#[derive(Clone)]
pub struct Service {
    pub tasks: HashMap<String, TaskStatus>,
    pub indexes: HashMap<String, HnswIndex>,
}

async fn extract_body(req: Request<Body>) -> Bytes {
    hyper::body::to_bytes(req.into_body()).await.unwrap()
}

enum TerminusIndexOperationError {}

const TERMINUSDB_INDEX_ENDPOINT: &str = "http://localhost:6363/api/index";
async fn get_operations_from_terminusdb(
    domain: String,
    commit: String,
    previous: Option<String>,
) -> Result<impl Stream<Item = io::Result<Operation>> + Unpin, io::Error> {
    let mut params: Vec<_> = [("domain", domain), ("commit", commit)]
        .into_iter()
        .collect();
    if let Some(previous) = previous {
        params.push(("previous", previous))
    }
    let url = reqwest::Url::parse_with_params(TERMINUSDB_INDEX_ENDPOINT, &params).unwrap();
    let res = reqwest::get(url)
        .await
        .unwrap()
        .bytes_stream()
        .map_err(|e| std::io::Error::new(ErrorKind::Other, e));
    let lines = StreamReader::new(res).lines();
    let lines_stream = LinesStream::new(lines);
    let fp = lines_stream.and_then(|l| {
        future::ready(
            serde_json::from_str(&l).map_err(|e| std::io::Error::new(ErrorKind::Other, e)),
        )
    });
    Ok(fp)
}

impl Service {
    fn generate_task() -> String {
        let s: String = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(8)
            .map(char::from)
            .collect();
        s
    }

    fn new<P: Into<PathBuf>>(_: P) -> Self {
        Service {
            tasks: HashMap::new(),
            indexes: HashMap::new(),
        }
    }

    async fn serve(&mut self, req: Request<Body>) -> Result<Response<Body>, Infallible> {
        match req.method() {
            &Method::POST => self.post(req).await,
            &Method::GET => self.get(req).await,
            _ => todo!(),
        }
    }

    async fn load_hnsw(&mut self, idxid: IndexIdentifier) -> HnswIndex {
        if let Some(previous) = idxid.previous {
            // load previous index
            Hnsw::new(OpenAI)
        } else {
            Hnsw::new(OpenAI)
        }
    }

    async fn start_indexing(&mut self, domain: String, commit: String, previous: Option<String>) {
        let opstream =
            get_operations_from_terminusdb(domain.clone(), commit.clone(), previous.clone())
                .await
                .unwrap();
        let hnsw = self
            .load_hnsw(IndexIdentifier {
                domain,
                commit,
                previous,
            })
            .await;
        start_indexing_from_operations(hnsw, opstream);
    }

    async fn get(&mut self, req: Request<Body>) -> Result<Response<Body>, Infallible> {
        let uri = req.uri();
        match uri_to_spec(uri) {
            Ok(ResourceSpec::StartIndex {
                domain,
                commit,
                previous,
            }) => {
                let task_id = Service::generate_task();
                self.start_indexing(domain, commit, previous);
                Ok(Response::builder()
                    .body(format!("{}", task_id).into())
                    .unwrap())
            }
            Ok(ResourceSpec::CheckTask { task_id }) => {
                if let Some(state) = self.tasks.get(&task_id) {
                    Ok(Response::builder()
                        .body(format!("{:?}", state).into())
                        .unwrap())
                } else {
                    Ok(Response::builder().body("Completed".into()).unwrap())
                }
            }
            Ok(_) => todo!(),
            Err(_) => todo!(),
        }
    }

    async fn post(&self, req: Request<Body>) -> Result<Response<Body>, Infallible> {
        let uri = req.uri();
        match uri_to_spec(uri) {
            Ok(_) => todo!(),
            Err(_) => todo!(),
        }
    }
}

pub async fn serve<P: Into<PathBuf>>(
    directory: P,
    port: u16,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), port);
    let service = Service::new(directory);
    let make_svc = make_service_fn(move |_conn| {
        let s = service.clone();
        async {
            Ok::<_, Infallible>(service_fn(move |req| {
                let mut s = s.clone();
                async move { s.serve(req).await }
            }))
        }
    });

    let server = Server::bind(&addr).serve(make_svc);
    server.await?;

    Ok(())
}
