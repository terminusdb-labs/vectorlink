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
    CheckTask {
        task_id: String,
    },
    Similar {
        domain: String,
        commit: String,
        id: String,
        count: usize,
    },
}

#[derive(Debug)]
enum SpecParseError {
    UnknownPath,
    NoTaskId,
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
        static ref RE_CHECK: Regex = Regex::new(r"^/check(/?)$").unwrap();
        static ref RE_SEARCH: Regex = Regex::new(r"^/search(/?)$").unwrap();
        static ref RE_SIMILAR: Regex = Regex::new(r"^/similar(/?)$").unwrap();
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

#[derive(Clone, Debug, Serialize)]
pub struct QueryResult {
    id: String,
    distance: f32,
}

pub struct Service {
    api_key: String,
    path: PathBuf,
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
    let mut params: Vec<_> = [("commit_id", commit)].into_iter().collect();
    if let Some(previous) = previous {
        params.push(("previous", previous))
    }
    let endpoint = format!("{}/{}", TERMINUSDB_INDEX_ENDPOINT, &domain);
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

    fn new<P: Into<PathBuf>>(path: P, num_bufs: usize, key: String) -> Self {
        let path = path.into();
        Service {
            api_key: key,
            path: path.clone(),
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
    ) {
        tokio::spawn(async move {
            let index_id = create_index_name(&domain, &commit);
            if self.test_and_set_pending(index_id.clone()).await {
                let opstream = get_operations_from_terminusdb(
                    domain.clone(),
                    commit.clone(),
                    previous.clone(),
                )
                .await
                .unwrap()
                .chunks(100);
                let (id, hnsw) = self
                    .process_operation_chunks(opstream, domain, commit, previous, &index_id)
                    .await;
                self.set_index(id, hnsw.into()).await;
                self.clear_pending(&index_id).await;
            }
            self.set_task_status(task_id, TaskStatus::Completed).await;
        });
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
    ) -> (String, HnswIndex) {
        let id = create_index_name(&domain, &commit);
        let mut hnsw = self
            .load_hnsw_for_indexing(IndexIdentifier {
                domain: domain.clone(),
                commit,
                previous,
            })
            .await;
        while let Some(structs) = opstream.next().await {
            let new_ops = operations_to_point_operations(
                &domain.clone(),
                &self.vector_store,
                structs,
                &self.api_key,
            )
            .await;
            hnsw = start_indexing_from_operations(hnsw, new_ops).unwrap();
        }
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
                self.set_task_status(task_id.clone(), TaskStatus::Pending)
                    .await;
                self.start_indexing(domain, commit, previous, task_id.clone());
                Ok(Response::builder().body(task_id.into()).unwrap())
            }
            Ok(ResourceSpec::CheckTask { task_id }) => {
                if let Some(state) = self.get_task_status(&task_id).await {
                    Ok(Response::builder()
                        .body(format!("{:?}", state).into())
                        .unwrap())
                } else {
                    Ok(Response::builder()
                        .body(format!("{:?}", TaskStatus::Completed).into())
                        .unwrap())
                }
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
            Err(_) => todo!(),
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

pub async fn serve<P: Into<PathBuf>>(
    directory: P,
    port: u16,
    num_bufs: usize,
    key: String,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), port);
    let service = Arc::new(Service::new(directory, num_bufs, key));
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
