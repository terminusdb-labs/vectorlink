use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::prelude::FileExt;
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::ops::Deref;

use lru::LruCache;

use crate::openai::{Embedding, EMBEDDING_LENGTH};

pub struct VectorDomain {
    name: Arc<String>
}


#[derive(Clone)]
pub struct LoadableVector {
    domain: Arc<String>,
    id: u64
}

pub struct LoadedVector {
    loadable: LoadableVector,
    counter: Arc<()>,
    vector: *const Embedding
}

impl LoadableVector {
    pub fn load(&self) -> LoadedVector {
        load_vec(&self.domain, self.id)
    }
}

impl LoadedVector {
    pub fn as_loadable(&self) -> LoadableVector {
        self.loadable.clone()
    }
}

impl Deref for LoadedVector {
    type Target = Embedding;

    fn deref(&self) -> &Self::Target {
        // We know this is allowed cause we hold an arc counter
        unsafe { &*self.vector}
    }
}

// 3 memory pages of 4K each hold a round number of embedding vectors
const VECTOR_PAGE_SIZE: usize = 3*4096*1024/(EMBEDDING_LENGTH*4);

type VectorPage = [f32;VECTOR_PAGE_SIZE];
type VectorPageBytes = [u8;VECTOR_PAGE_SIZE*4];

struct LoadedVectorPage {
    index: usize,
    counter: Arc<()>, // for pinning
    page: Box<VectorPage>
}

struct Domain {
    name: Arc<String>,
    index: usize,
    file: File
}

impl Domain {
    fn open(dir: &PathBuf, name: &str, index: usize) -> io::Result<Self> {
        let mut path = dir.clone();
        path.push(format!("{name}.vecs"));
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        Ok(Domain {
            name: Arc::new(name.to_string()),
            index,
            file,
        })
    }

    fn add_vecs(&mut self, vecs: &[Embedding]) -> io::Result<()> {
        todo!();
    }

    fn load_page(&mut self, index: usize, data: &mut VectorPage) -> io::Result<bool> {
        let offset = (index * std::mem::size_of::<VectorPage>()) as u64;
        let data: &mut VectorPageBytes = unsafe { std::mem::transmute(data) };
        let result = self.file.read_exact_at(data, offset);
        if result.is_err() && result.as_ref().err().unwrap().kind() == io::ErrorKind::UnexpectedEof {
            // this page doesn't exist.
            return Ok(false);
        }
        // any other error can propagate
        result?;

        Ok(true)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone)]
struct PageSpec {
    domain: Arc<String>,
    index: usize,
}

struct VectorStore {
    dir: PathBuf,
    free: Vec<Box<VectorPage>>,
    loaded: HashMap<PageSpec, LoadedVectorPage>,
    cache: LruCache<PageSpec, LoadedVectorPage>,
    domains: HashMap<String, Domain>
}

impl VectorStore {
    pub fn add_vecs(&mut self, domain: &str, vecs: &[Embedding]) -> io::Result<()> {
        let domain = self.get_or_add_domain(domain)?;
        domain.add_vecs(vecs)
    }
}

impl VectorStore {
    fn get_or_add_domain(&mut self, name: &str) -> io::Result<&mut Domain> {
        if !self.domains.contains_key(name) {
            let domain = Domain::open(&self.dir, name, self.domains.len())?;
            self.domains.insert(name.to_string(), domain);
        }

        Ok(self.domains.get_mut(name).unwrap())
    }

    fn get_page(&mut self, domain: &mut Domain, index: usize) -> io::Result<Option<&mut LoadedVectorPage>> {
        let pageSpec = PageSpec {
            domain: domain.name.clone(),
            index,
        };
        let mut found = self.loaded.contains_key(&pageSpec);
        if !found {
            // page is not currently loaded. it might still be in cache.
            if let Some(page) = self.cache.pop(&pageSpec) {
                found = true;
                // we'll have to move it into loaded.
                self.loaded.insert(pageSpec.clone(), page);
            } else {
                // turns out it is not in cache. time to go to disk and get it from there.
                // We'll need a free page first to handle this load
                let free = self.free.pop()
                    .or_else(|| self.cache.pop_lru().map(|x|x.1.page));
                if let Some(mut free) = free {
                    let found = domain.load_page(index, &mut free)?;
                    if found {
                        // great, we managed to load this page. let's move it into loaded
                        self.loaded.insert(pageSpec.clone(), LoadedVectorPage {
                            index,
                            counter: Arc::new(()), // TODO
                            page: free
                        });
                    }
                } else {
                    // TODO dynamically grow free pool up to an upper limit
                    return Err(io::Error::new(io::ErrorKind::Other, "no available buffer to load vector page"));
                }
            }
        }

        if found {
            Ok(Some(self.loaded.get_mut(&pageSpec).unwrap()))
        } else {
            Ok(None)
        }
    }
}

pub fn load_vec(domain: &str, id: u64) -> LoadedVector {
    todo!();
}
