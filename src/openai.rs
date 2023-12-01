#![allow(unused, dead_code)]
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use lazy_static::lazy_static;
use reqwest::{header::HeaderValue, Body, Client, Method, Request, StatusCode, Url};
use serde::{
    de::{SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use thiserror::Error;
use tiktoken_rs::{cl100k_base, CoreBPE};
use tokio::sync::{Mutex, Notify, RwLock};

use crate::vecmath::Embedding;

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [Vec<usize>],
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<&'a str>,
}

#[derive(Deserialize, Debug)]
struct EmbeddingResponse {
    object: String,
    data: Vec<EmbeddingData>,
    model: String,
    usage: EmbeddingUsage,
}

#[derive(Deserialize, Debug)]
struct EmbeddingData {
    object: String,
    index: usize,
    #[serde(deserialize_with = "deserialize_single_embedding")]
    embedding: Embedding,
}

fn deserialize_single_embedding<'de, D>(deserializer: D) -> Result<Embedding, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_seq(SingleEmbeddingVisitor)
}

struct SingleEmbeddingVisitor;

impl<'de> Visitor<'de> for SingleEmbeddingVisitor {
    type Value = Embedding;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "a list of 1536 floats")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut result = [0.0; 1536];
        let mut index = 0;
        while let Some(next) = seq.next_element()? {
            if index >= result.len() {
                // should not really happen but let's not panic
                break;
            }
            result[index] = next;
            index += 1;
        }

        Ok(result)
    }
}

#[derive(Deserialize, Debug)]
struct EmbeddingUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("error while doing openai request: {0:?}")]
    ReqwestError(#[from] reqwest::Error),
    #[error("response had bad status code: {}", .0)]
    BadStatus(StatusCode, String),

    #[error("error while parsing json: {0:?}")]
    BadJson(#[from] serde_json::Error),
}

lazy_static! {
    static ref ENCODER: CoreBPE = cl100k_base().unwrap();
}

fn tokens_for(s: &str) -> Vec<usize> {
    ENCODER.encode_with_special_tokens(s)
}

const MAX_TOKEN_COUNT: usize = 8191;
fn truncated_tokens_for(s: &str) -> Vec<usize> {
    let mut tokens = tokens_for(s);
    if tokens.len() > MAX_TOKEN_COUNT {
        tokens.truncate(MAX_TOKEN_COUNT);
        let decoded = ENCODER.decode(tokens.clone()).unwrap();
        eprintln!("truncating to {decoded}");
    }

    tokens
}

struct RateLimiter {
    budget: Arc<Mutex<usize>>,
    waiters: Arc<Mutex<VecDeque<(usize, Arc<Notify>)>>>,
}

impl RateLimiter {
    fn new(budget: usize) -> Self {
        Self {
            budget: Arc::new(Mutex::new(budget)),
            waiters: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    async fn wakeup_existing(mut budget: usize, waiters: &mut VecDeque<(usize, Arc<Notify>)>) {
        while waiters
            .front()
            .map(|(requested_budget, _)| *requested_budget < budget)
            .unwrap_or(false)
        {
            eprintln!("wake up time!");
            let (requested_budget, wakeup) = waiters.pop_front().unwrap();
            wakeup.notify_one();
            budget -= requested_budget;
        }
    }

    async fn budget_tokens(&self, requested_budget: usize) {
        loop {
            let mut budget = self.budget.lock().await;
            if requested_budget <= *budget {
                *budget -= requested_budget;
                eprintln!("requested {}. budget now {}", requested_budget, *budget);
                let inner_budget = self.budget.clone();
                let inner_waiters = self.waiters.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                    let mut budget = inner_budget.lock().await;
                    *budget += requested_budget;
                    let budget_copy = *budget;
                    std::mem::drop(budget);
                    eprintln!("minute passed. budget now {}", budget_copy);
                    Self::wakeup_existing(budget_copy, &mut *inner_waiters.lock().await);
                });
                return;
            } else {
                eprintln!("rate limit time!");
                std::mem::drop(budget);
                let notify = Arc::new(Notify::new());
                {
                    let mut waiters = self.waiters.lock().await;
                    waiters.push_back((requested_budget, notify.clone()));
                }
                notify.notified().await;
            }
        }
    }
}

pub async fn embeddings_for(
    api_key: &str,
    strings: &[String],
) -> Result<Vec<Embedding>, EmbeddingError> {
    const RATE_LIMIT: usize = 1_000_000;
    lazy_static! {
        static ref ENDPOINT: Url = Url::parse("https://api.openai.com/v1/embeddings").unwrap();
        static ref CLIENT: Client = Client::new();
        static ref LIMITERS: Arc<RwLock<HashMap<String, RateLimiter>>> =
            Arc::new(RwLock::new(HashMap::new()));
    }

    {
        let read_guard = LIMITERS.read().await;
        if !read_guard.contains_key(api_key) {
            std::mem::drop(read_guard);
            let mut write_guard = LIMITERS.write().await;
            let limiter = RateLimiter::new(RATE_LIMIT);
            write_guard.insert(api_key.to_owned(), limiter);
        }
    }
    let read_guard = LIMITERS.read().await;
    let limiter = &read_guard[api_key];

    let token_lists: Vec<_> = strings.iter().map(|s| truncated_tokens_for(s)).collect();
    limiter
        .budget_tokens(token_lists.iter().map(|ts| ts.len()).sum())
        .await;

    let mut req = Request::new(Method::POST, ENDPOINT.clone());
    let headers = req.headers_mut();
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));
    headers.insert(
        "Authorization",
        HeaderValue::from_str(&format!("Bearer {api_key}")).unwrap(),
    );

    let body = EmbeddingRequest {
        model: "text-embedding-ada-002",
        input: &token_lists,
        user: None,
    };
    let body_vec = serde_json::to_vec(&body).unwrap();
    let body: Body = body_vec.into();

    *req.body_mut() = Some(body); // once told me the world is gonna roll me

    let response = CLIENT.execute(req).await?;
    let status = response.status();
    let response_bytes = response.bytes().await?;
    if status != StatusCode::OK {
        let body = String::from_utf8_lossy(&response_bytes).to_string();
        return Err(EmbeddingError::BadStatus(status, body));
    }
    let response: EmbeddingResponse = serde_json::from_slice(&response_bytes)?;
    let mut result = Vec::with_capacity(strings.len());
    for embedding in response.data {
        result.push(embedding.embedding);
    }

    Ok(result)
}
