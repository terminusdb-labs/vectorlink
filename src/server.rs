use bytes::Bytes;
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
enum TaskStatus {
    Pending,
    Error,
    Completed,
}

#[derive(Clone)]
struct Service {
    tasks: HashMap<String, TaskStatus>,
}

async fn extract_body(req: Request<Body>) -> Bytes {
    hyper::body::to_bytes(req.into_body()).await.unwrap()
}

impl Service {
    fn generate_task() -> String {
        let s: String = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(7)
            .map(char::from)
            .collect();
        s
    }

    fn new<P: Into<PathBuf>>(_: P) -> Self {
        Service {
            tasks: HashMap::new(),
        }
    }

    async fn serve(&self, req: Request<Body>) -> Result<Response<Body>, Infallible> {
        match req.method() {
            &Method::POST => self.post(req).await,
            &Method::GET => self.get(req).await,
            _ => todo!(),
        }
    }

    fn start_indexing(&self, domain: String, commit: String, previous: Option<String>) {
        todo!();
        //return Ok(());
    }

    async fn get(&self, req: Request<Body>) -> Result<Response<Body>, Infallible> {
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
