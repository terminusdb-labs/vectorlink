use bytes::Bytes;
use futures::StreamExt;
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
use std::collections::HashSet;
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
use tokio::sync::Mutex;
use tokio::{io::AsyncBufReadExt, sync::RwLock};
use tokio_stream::{wrappers::LinesStream, Stream};
use tokio_util::io::StreamReader;

use crate::indexer::operations_to_point_operations;
use crate::indexer::PointOperation;
use crate::indexer::{start_indexing_from_operations, HnswIndex, IndexIdentifier, OpenAI};
use crate::vectors::VectorStore;

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
    Search,
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
        static ref RE_SEARCH: Regex = Regex::new(r"^/search(/?)$").unwrap();
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
    } else if RE_SEARCH.is_match(path) {
        Ok(ResourceSpec::Search)
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

pub struct Service {
    vector_store: VectorStore,
    pending: Mutex<HashSet<String>>,
    tasks: RwLock<HashMap<String, TaskStatus>>,
    indexes: RwLock<HashMap<String, Arc<HnswIndex>>>,
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

fn create_index_id(commit_id: &str, domain: &str) -> String {
    format!("{}_{}", commit_id, domain)
}

impl Service {
    async fn get_task_status(&self, task_id: &str) -> Option<TaskStatus> {
        self.tasks.read().await.get(task_id).cloned()
    }

    async fn set_task_status(&self, task_id: String, status: TaskStatus) {
        self.tasks.write().await.insert(task_id, status);
    }

    async fn get_index(&self, commit_id: &str) -> Option<Arc<HnswIndex>> {
        self.indexes.read().await.get(commit_id).cloned()
    }

    async fn set_index(&self, commit_id: String, hnsw: Arc<HnswIndex>) {
        self.indexes.write().await.insert(commit_id, hnsw);
    }

    async fn test_and_set_pending(&self, index_id: String) -> bool {
        let mut lock = self.pending.lock().await;
        if lock.contains(&index_id) {
            false
        } else {
            lock.insert(index_id);
            true
        }
    }

    async fn clear_pending(&self, index_id: &str) {
        self.pending.lock().await.remove(index_id);
    }

    fn generate_task() -> String {
        let s: String = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(8)
            .map(char::from)
            .collect();
        s
    }

    fn new<P: Into<PathBuf>>(path: P, num_bufs: usize) -> Self {
        Service {
            vector_store: VectorStore::new(path, num_bufs),
            pending: Mutex::new(HashSet::new()),
            tasks: RwLock::new(HashMap::new()),
            indexes: RwLock::new(HashMap::new()),
        }
    }

    async fn serve(self: Arc<Self>, req: Request<Body>) -> Result<Response<Body>, Infallible> {
        match req.method() {
            &Method::POST => self.post(req).await,
            &Method::GET => self.get(req).await,
            _ => todo!(),
        }
    }

    async fn load_hnsw_for_indexing(&self, idxid: IndexIdentifier) -> HnswIndex {
        if let Some(previous_id) = idxid.previous {
            //let commit_id = idxid.commit;
            let domain = idxid.domain;
            let previous_id = create_index_id(&previous_id, &domain);
            //let current_id = create_index_id(&commit_id, &domain);
            let hnsw = self.get_index(&previous_id).await.unwrap();
            (*hnsw).clone()
        } else {
            Hnsw::new(OpenAI)
        }
    }

    fn start_indexing(self: Arc<Self>, domain: String, commit: String, previous: Option<String>) {
        tokio::spawn(async move {
            let index_id = create_index_id(&domain, &commit);
            if self.test_and_set_pending(index_id.clone()).await {
                let mut opstream = get_operations_from_terminusdb(
                    domain.clone(),
                    commit.clone(),
                    previous.clone(),
                )
                .await
                .unwrap()
                .chunks(100);
                let mut point_ops: Vec<PointOperation> = Vec::new();
                while let Some(structs) = opstream.next().await {
                    let mut new_ops =
                        operations_to_point_operations(&domain, &self.vector_store, structs).await;
                    point_ops.append(&mut new_ops)
                }
                let id = create_index_id(&commit, &domain);
                let hnsw = self
                    .load_hnsw_for_indexing(IndexIdentifier {
                        domain,
                        commit,
                        previous,
                    })
                    .await;
                let hnsw = start_indexing_from_operations(hnsw, point_ops).unwrap();
                self.set_index(id, hnsw.into()).await;
                self.clear_pending(&index_id).await;
            }
        });
    }

    async fn get(self: Arc<Self>, req: Request<Body>) -> Result<Response<Body>, Infallible> {
        let uri = req.uri();
        match uri_to_spec(uri) {
            Ok(ResourceSpec::StartIndex {
                domain,
                commit,
                previous,
            }) => {
                let task_id = Service::generate_task();
                self.start_indexing(domain, commit, previous);
                Ok(Response::builder().body(task_id.into()).unwrap())
            }
            Ok(ResourceSpec::CheckTask { task_id }) => {
                if let Some(state) = self.get_task_status(&task_id).await {
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
        todo!()
        /*
        match uri_to_spec(uri) {
            Ok(ResourceSpec::Search) => {
                let search: String = req.body().bytes()  to_text();
                let search_strings = vec![search];
                let vec: Vec<Embedding> = embedings_for(API_KEY, &search_strings);
                todo!()
            }
            Ok(_) => panic!(),
            Err(_) => panic!(),
         }
         */
    }
}

pub async fn serve<P: Into<PathBuf>>(
    directory: P,
    port: u16,
    num_bufs: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), port);
    let service = Arc::new(Service::new(directory, num_bufs));
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
