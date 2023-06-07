#![allow(unused, dead_code)]
use bytes::Bytes;
use futures::StreamExt;
use futures::TryStreamExt;
use hnsw::Hnsw;
use hyper::StatusCode;
use hyper::{
    service::{make_service_fn, service_fn},
    Body, Method, Request, Response, Server, Uri,
};
use lazy_static::lazy_static;
use rand::distributions::Alphanumeric;
use rand::Rng;
use regex::Regex;
use serde::Serialize;
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
use thiserror::Error;
use tokio::sync::Mutex;
use tokio::{io::AsyncBufReadExt, sync::RwLock};
use tokio_stream::{wrappers::LinesStream, Stream};
use tokio_util::io::StreamReader;

use crate::indexer::create_index_name;
use crate::indexer::deserialize_index;
use crate::indexer::operations_to_point_operations;
use crate::indexer::search;
use crate::indexer::serialize_index;
use crate::indexer::Point;
use crate::indexer::PointOperation;
use crate::indexer::{start_indexing_from_operations, HnswIndex, IndexIdentifier, OpenAI};
use crate::openai::embeddings_for;
use crate::vectors::VectorStore;

#[derive(Clone, Deserialize, Debug)]
#[serde(tag = "op")]
pub enum Operation {
    Inserted { string: String, id: String },
    Changed { string: String, id: String },
    Deleted { id: String },
    Error { message: String },
}

#[derive(Deserialize, Debug)]
struct IndexRequest {
    domain: String,
    commit: String,
    previous: Option<String>,
    operations: Vec<Operation>,
}

#[derive(Debug)]
enum ResourceSpec {
    Search {
        domain: String,
        commit: String,
        count: usize,
    },
    StartIndex {
        domain: String,
        commit: String,
        previous: Option<String>,
    },
    AssignIndex {
        domain: String,
        source_commit: String,
        target_commit: String,
    },
    CheckTask {
        task_id: String,
    },
    Similar {
        domain: String,
        commit: String,
        id: String,
        count: usize,
    },
    DuplicateCandidates {
        domain: String,
        commit: String,
        threshold: f32,
    },
}

#[derive(Debug, Error)]
enum SpecParseError {
    #[error("Unknown URL Path")]
    UnknownPath,
    #[error("No task id")]
    NoTaskId,
    #[error("No commit id or domain id given")]
    NoCommitIdOrDomain,
}

fn query_map(uri: &Uri) -> HashMap<String, String> {
    uri.query()
        .map(|v| {
            url::form_urlencoded::parse(v.as_bytes())
                .into_owned()
                .collect()
        })
        .unwrap_or_else(|| HashMap::with_capacity(0))
}

fn uri_to_spec(uri: &Uri) -> Result<ResourceSpec, SpecParseError> {
    lazy_static! {
        static ref RE_INDEX: Regex = Regex::new(r"^/index(/?)$").unwrap();
        static ref RE_ASSIGN: Regex = Regex::new(r"^/assign(/?)$").unwrap();
        static ref RE_CHECK: Regex = Regex::new(r"^/check(/?)$").unwrap();
        static ref RE_SEARCH: Regex = Regex::new(r"^/search(/?)$").unwrap();
        static ref RE_SIMILAR: Regex = Regex::new(r"^/similar(/?)$").unwrap();
        static ref RE_DUPLICATES: Regex = Regex::new(r"^/duplicates(/?)$").unwrap();
    }
    let path = uri.path();

    if RE_INDEX.is_match(path) {
        let query = dbg!(query_map(uri));
        let commit = query.get("commit").map(|v| v.to_string());
        let domain = query.get("domain").map(|v| v.to_string());
        let previous = query.get("previous").map(|v| v.to_string());
        match (domain, commit) {
            (Some(domain), Some(commit)) => Ok(ResourceSpec::StartIndex {
                domain,
                commit,
                previous,
            }),
            _ => Err(SpecParseError::NoCommitIdOrDomain),
        }
    } else if RE_ASSIGN.is_match(path) {
        let query = query_map(uri);
        let domain = query.get("domain").map(|v| v.to_string());
        let source_commit = query.get("source_commit").map(|v| v.to_string());
        let target_commit = query.get("target_commit").map(|v| v.to_string());
        match (domain, source_commit, target_commit) {
            (Some(domain), Some(source_commit), Some(target_commit)) => {
                Ok(ResourceSpec::AssignIndex {
                    domain,
                    source_commit,
                    target_commit,
                })
            }
            _ => Err(SpecParseError::NoCommitIdOrDomain),
        }
    } else if RE_CHECK.is_match(path) {
        let query = query_map(uri);
        if let Some(task_id) = query.get("task_id") {
            Ok(ResourceSpec::CheckTask {
                task_id: task_id.to_string(),
            })
        } else {
            Err(SpecParseError::NoTaskId)
        }
    } else if RE_SEARCH.is_match(path) {
        let query = query_map(uri);
        let domain = query.get("domain").map(|v| v.to_string());
        let commit = query.get("commit").map(|v| v.to_string());
        let count = query.get("count").map(|v| v.parse::<usize>().unwrap());
        match (domain, commit) {
            (Some(domain), Some(commit)) => {
                let count = count.unwrap_or(10);
                Ok(ResourceSpec::Search {
                    domain,
                    commit,
                    count,
                })
            }
            _ => Err(SpecParseError::NoCommitIdOrDomain),
        }
    } else if RE_SIMILAR.is_match(path) {
        let query = query_map(uri);
        let domain = query.get("domain").map(|v| v.to_string());
        let commit = query.get("commit").map(|v| v.to_string());
        let id = query.get("id").map(|v| v.to_string());
        let count = query.get("count").map(|v| v.parse::<usize>().unwrap());
        match (domain, commit, id) {
            (Some(domain), Some(commit), Some(id)) => {
                let count = count.unwrap_or(10);
                Ok(ResourceSpec::Similar {
                    domain,
                    commit,
                    id,
                    count,
                })
            }
            _ => Err(SpecParseError::NoCommitIdOrDomain),
        }
    } else if RE_DUPLICATES.is_match(path) {
        let query = query_map(uri);
        let domain = query.get("domain").map(|v| v.to_string());
        let commit = query.get("commit").map(|v| v.to_string());
        let threshold = query.get("threshold").map(|v| v.parse::<f32>().unwrap());
        match (domain, commit) {
            (Some(domain), Some(commit)) => {
                let threshold = threshold.unwrap_or(0.0);
                Ok(ResourceSpec::DuplicateCandidates {
                    domain,
                    commit,
                    threshold,
                })
            }
            _ => Err(SpecParseError::NoCommitIdOrDomain),
        }
    } else {
        Err(SpecParseError::UnknownPath)
    }
}

#[derive(Clone, Debug)]
pub enum TaskStatus {
    Pending(f32),
    Error,
    Completed,
}

#[derive(Clone, Debug, Serialize)]
pub struct QueryResult {
    id: String,
    distance: f32,
}

pub struct Service {
    content_endpoint: Option<String>,
    api_key: String,
    path: PathBuf,
    vector_store: VectorStore,
    pending: Mutex<HashSet<String>>,
    tasks: RwLock<HashMap<String, TaskStatus>>,
    indexes: RwLock<HashMap<String, Arc<HnswIndex>>>,
}

#[derive(Debug, Error)]
enum StartIndexError {
    #[error("No content endpoint found: specify at server startup or supply indexing data from the command line")]
    NoContentEndpoint,
}

async fn extract_body(req: Request<Body>) -> Bytes {
    hyper::body::to_bytes(req.into_body()).await.unwrap()
}

enum TerminusIndexOperationError {}

async fn get_operations_from_content_endpoint(
    content_endpoint: String,
    domain: String,
    commit: String,
    previous: Option<String>,
) -> Result<impl Stream<Item = io::Result<Operation>> + Unpin, io::Error> {
    let mut params: Vec<_> = [("commit_id", commit)].into_iter().collect();
    if let Some(previous) = previous {
        params.push(("previous", previous))
    }
    let endpoint = format!("{}/{}", content_endpoint, &domain);
    let url = reqwest::Url::parse_with_params(&endpoint, &params).unwrap();
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

fn add_to_duplicates(duplicates: &mut HashMap<usize, usize>, id1: usize, id2: usize) {
    if id1 < id2 {
        duplicates.insert(id1, id2);
    }
}

impl Service {
    async fn get_task_status(&self, task_id: &str) -> Option<TaskStatus> {
        self.tasks.read().await.get(task_id).cloned()
    }

    async fn set_task_status(&self, task_id: String, status: TaskStatus) {
        self.tasks.write().await.insert(task_id, status);
    }

    async fn get_index(&self, index_id: &str) -> Option<Arc<HnswIndex>> {
        if let Some(hnsw) = self.indexes.read().await.get(index_id) {
            Some(hnsw).cloned()
        } else {
            let mut path = self.path.clone();
            match deserialize_index(&mut path, index_id, &self.vector_store) {
                Ok(res) => Some(res.into()),
                Err(_) => None,
            }
        }
    }

    async fn set_index(&self, index_id: String, hnsw: Arc<HnswIndex>) {
        self.indexes.write().await.insert(index_id, hnsw);
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

    fn new<P: Into<PathBuf>>(
        path: P,
        num_bufs: usize,
        key: String,
        content_endpoint: Option<String>,
    ) -> Self {
        let path = path.into();
        Service {
            content_endpoint,
            api_key: key,
            path: path.clone(),
            vector_store: VectorStore::new(path, num_bufs),
            pending: Mutex::new(HashSet::new()),
            tasks: RwLock::new(HashMap::new()),
            indexes: RwLock::new(HashMap::new()),
        }
    }

    async fn serve(self: Arc<Self>, req: Request<Body>) -> Result<Response<Body>, Infallible> {
        match *req.method() {
            Method::POST => self.post(req).await,
            Method::GET => self.get(req).await,
            _ => todo!(),
        }
    }

    async fn load_hnsw_for_indexing(&self, idxid: IndexIdentifier) -> HnswIndex {
        if let Some(previous_id) = idxid.previous {
            //let commit = idxid.commit;
            let domain = idxid.domain;
            let previous_id = create_index_name(&domain, &previous_id);
            let hnsw = self.get_index(&previous_id).await.unwrap();
            (*hnsw).clone()
        } else {
            Hnsw::new(OpenAI)
        }
    }

    fn start_indexing(
        self: Arc<Self>,
        domain: String,
        commit: String,
        previous: Option<String>,
        task_id: String,
        api_key: String,
    ) -> Result<(), StartIndexError> {
        let content_endpoint = self.content_endpoint.clone();
        if let Some(content_endpoint) = content_endpoint {
            tokio::spawn(async move {
                let index_id = create_index_name(&domain, &commit);
                if self.test_and_set_pending(index_id.clone()).await {
                    let opstream = get_operations_from_content_endpoint(
                        content_endpoint.to_string(),
                        domain.clone(),
                        commit.clone(),
                        previous.clone(),
                    )
                    .await
                    .unwrap()
                    .chunks(100);
                    let (id, hnsw) = self
                        .process_operation_chunks(
                            opstream, domain, commit, previous, &index_id, &task_id, &api_key,
                        )
                        .await;
                    self.set_index(id, hnsw.into()).await;
                    self.clear_pending(&index_id).await;
                }
                self.set_task_status(task_id, TaskStatus::Completed).await;
            });
            Ok(())
        } else {
            Err(StartIndexError::NoContentEndpoint)
        }
    }

    async fn assign_index(
        self: Arc<Self>,
        domain: String,
        source_commit: String,
        target_commit: String,
    ) -> Result<(), AssignIndexError> {
        let source_name = create_index_name(&domain, &source_commit);
        let target_name = create_index_name(&domain, &target_commit);

        if self.get_index(&target_name).await.is_some() {
            return Err(AssignIndexError::TargetCommitAlreadyHasIndex);
        }
        if let Some(index) = self.get_index(&source_name).await {
            let mut indexes = self.indexes.write().await;
            indexes.insert(target_name.clone(), index.clone());

            std::mem::drop(indexes);
            tokio::task::block_in_place(move || {
                let path = self.path.clone();
                serialize_index(path, &target_name, (*index).clone()).unwrap();
            });

            Ok(())
        } else {
            Err(AssignIndexError::SourceCommitNotFound)
        }
    }

    async fn process_operation_chunks(
        self: &Arc<Self>,
        mut opstream: futures::stream::Chunks<
            impl Stream<Item = Result<Operation, io::Error>> + Unpin,
        >,
        domain: String,
        commit: String,
        previous: Option<String>,
        index_id: &str,
        task_id: &str,
        api_key: &str,
    ) -> (String, HnswIndex) {
        let id = create_index_name(&domain, &commit);
        let mut hnsw = self
            .load_hnsw_for_indexing(IndexIdentifier {
                domain: domain.clone(),
                commit,
                previous,
            })
            .await;
        self.set_task_status(task_id.to_string(), TaskStatus::Pending(0.3))
            .await;
        while let Some(structs) = opstream.next().await {
            let new_ops = operations_to_point_operations(
                &domain.clone(),
                &self.vector_store,
                structs,
                api_key,
            )
            .await;
            hnsw = start_indexing_from_operations(hnsw, new_ops).unwrap();
        }
        self.set_task_status(task_id.to_string(), TaskStatus::Pending(0.8))
            .await;
        let path = self.path.clone();
        serialize_index(path, index_id, hnsw.clone()).unwrap();
        (id, hnsw)
    }

    async fn get(self: Arc<Self>, req: Request<Body>) -> Result<Response<Body>, Infallible> {
        let uri = req.uri();
        match dbg!(uri_to_spec(uri)) {
            Ok(ResourceSpec::StartIndex {
                domain,
                commit,
                previous,
            }) => {
                let task_id = Service::generate_task();
                let headers = req.headers();
                let openai_key = headers.get("TERMINUSDB_VECTOR_API_KEY");
                match openai_key {
                    Some(openai_key) => {
                        let openai_key = String::from_utf8(openai_key.as_bytes().to_vec()).unwrap();
                        self.set_task_status(task_id.clone(), TaskStatus::Pending(0.0))
                            .await;
                        match self.start_indexing(
                            domain,
                            commit,
                            previous,
                            task_id.clone(),
                            openai_key,
                        ) {
                            Ok(()) => Ok(Response::builder().body(task_id.into()).unwrap()),
                            Err(e) => Ok(Response::builder()
                                .status(400)
                                .body(e.to_string().into())
                                .unwrap()),
                        }
                    }
                    None => Ok(Response::builder()
                        .status(400)
                        .body(
                            "No API key supplied in header (TERMINUSDB_VECTOR_API_KEY)"
                                .to_string()
                                .into(),
                        )
                        .unwrap()),
                }
            }
            Ok(ResourceSpec::AssignIndex {
                domain,
                source_commit,
                target_commit,
            }) => {
                let result = self
                    .assign_index(domain, source_commit, target_commit)
                    .await;
                match result {
                    Ok(()) => Ok(Response::builder().status(204).body(Body::empty()).unwrap()),
                    Err(e) => Ok(Response::builder()
                        .status(400)
                        .body(e.to_string().into())
                        .unwrap()),
                }
            }
            Ok(ResourceSpec::CheckTask { task_id }) => {
                if let Some(state) = self.get_task_status(&task_id).await {
                    match state {
                        TaskStatus::Pending(f) => {
                            Ok(Response::builder().body(format!("{}", f).into()).unwrap())
                        }
                        TaskStatus::Error => Ok(Response::builder()
                            .body(format!("{:?}", state).into())
                            .unwrap()),
                        TaskStatus::Completed => {
                            Ok(Response::builder().body(format!("{}", 1.0).into()).unwrap())
                        }
                    }
                } else {
                    Ok(Response::builder().body(format!("{}", 1.0).into()).unwrap())
                }
            }
            Ok(ResourceSpec::DuplicateCandidates {
                domain,
                commit,
                threshold,
            }) => {
                let index_id = create_index_name(&domain, &commit);
                // if None, then return 404
                let hnsw = self.get_index(&index_id).await.unwrap();
                let mut duplicates: HashMap<usize, usize> = HashMap::new();
                let elts = hnsw.layer_len(0);
                for i in 0..elts {
                    let current_point = &hnsw.feature(i);
                    let results = search(current_point, 2, &hnsw).unwrap();
                    for result in results.iter() {
                        if f32::from_bits(result.distance()) < threshold {
                            add_to_duplicates(&mut duplicates, i, result.internal_id())
                        }
                    }
                }
                let mut v: Vec<(&str, &str)> = duplicates
                    .into_iter()
                    .map(|(i, j)| (hnsw.feature(i).id(), hnsw.feature(j).id()))
                    .collect();
                Ok(Response::builder()
                    .body(serde_json::to_string(&v).unwrap().into())
                    .unwrap())
            }
            Ok(ResourceSpec::Similar {
                domain,
                commit,
                count,
                id,
            }) => {
                let index_id = create_index_name(&domain, &commit);
                // if None, then return 404
                let hnsw = self.get_index(&index_id).await.unwrap();
                let elts = hnsw.layer_len(0);
                let mut qp = None;
                for i in 0..elts {
                    if *hnsw.feature(i).id() == id {
                        qp = Some(hnsw.feature(i))
                    }
                }
                match qp {
                    Some(qp) => {
                        let res = search(qp, count, &hnsw).unwrap();
                        let ids: Vec<QueryResult> = res
                            .iter()
                            .map(|p| QueryResult {
                                id: p.id().to_string(),
                                distance: f32::from_bits(p.distance()),
                            })
                            .collect();
                        let s = serde_json::to_string(&ids).unwrap();
                        Ok(Response::builder().body(s.into()).unwrap())
                    }
                    None => Ok(Response::builder()
                        .status(StatusCode::NOT_FOUND)
                        .body("id not found".into())
                        .unwrap()),
                }
            }
            Ok(_) => todo!(),
            Err(e) => Ok(Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(e.to_string().into())
                .unwrap()),
        }
    }

    async fn post(&self, req: Request<Body>) -> Result<Response<Body>, Infallible> {
        let uri = req.uri();
        match uri_to_spec(uri) {
            Ok(ResourceSpec::Search {
                domain,
                commit,
                count,
            }) => {
                let body_bytes = hyper::body::to_bytes(req.into_body()).await.unwrap();
                let q = String::from_utf8(body_bytes.to_vec()).unwrap();
                let vec = Box::new((embeddings_for(&self.api_key, &[q]).await.unwrap())[0]);
                let qp = Point::Mem { vec };
                let index_id = create_index_name(&domain, &commit);
                // if None, then return 404
                let hnsw = self.get_index(&index_id).await.unwrap();
                let res = search(&qp, count, &hnsw).unwrap();
                let ids: Vec<QueryResult> = res
                    .iter()
                    .map(|p| QueryResult {
                        id: p.id().to_string(),
                        distance: f32::from_bits(p.distance()),
                    })
                    .collect();
                let s = serde_json::to_string(&ids).unwrap();
                Ok(Response::builder().body(s.into()).unwrap())
            }
            Ok(_) => todo!(),
            Err(_) => todo!(),
        }
    }
}

#[derive(Debug, Error)]
enum AssignIndexError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("source commit not found")]
    SourceCommitNotFound,
    #[error("target commit already has an index")]
    TargetCommitAlreadyHasIndex,
}

pub async fn serve<P: Into<PathBuf>>(
    directory: P,
    port: u16,
    num_bufs: usize,
    key: String,
    content_endpoint: Option<String>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), port);
    let service = Arc::new(Service::new(directory, num_bufs, key, content_endpoint));
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
