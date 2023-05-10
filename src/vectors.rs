use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::prelude::FileExt;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Weak, Condvar, Mutex, RwLock};
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
const VECTOR_PAGE_FLOAT_SIZE: usize = 3*4096*1024/(EMBEDDING_LENGTH*4);
const VECTOR_PAGE_BYTE_SIZE: usize = VECTOR_PAGE_FLOAT_SIZE * 4;

type VectorPage = [f32;VECTOR_PAGE_FLOAT_SIZE];
type VectorPageBytes = [u8;VECTOR_PAGE_BYTE_SIZE];

struct LoadedVectorPage {
    index: usize,
    page: Pin<Box<VectorPage>>
}

struct PinnedVectorPage {
    page: LoadedVectorPage,
    handle: Weak<PageHandle>
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
            .truncate(false)
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

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone, Copy)]
struct PageSpec {
    domain: usize,
    index: usize,
}

struct PageArena {
    free: Mutex<Vec<Pin<Box<VectorPage>>>>,
    loading: Mutex<HashMap<PageSpec, Arc<(Condvar, Mutex<LoadState>)>>>,
    loaded: RwLock<HashMap<PageSpec, PinnedVectorPage>>,
    cache: RwLock<LruCache<PageSpec, LoadedVectorPage>>,
}

#[derive(Default, PartialEq, Eq, Clone, Copy)]
enum LoadState {
    #[default]
    Loading,
    Loaded,
    Cached,
    Canceled
}

impl PageArena {
    fn alloc_free_pages(&self, count: usize) {
        if count == 0 {
            return;
        }
        // TODO would be much better if we could have uninit allocs.
        let mut free = self.free.lock().unwrap();
        let zeroed = Box::pin([0.0f32;VECTOR_PAGE_FLOAT_SIZE]);
        for _ in 0..=count {
            free.push(zeroed.clone());
        }
        free.push(zeroed);
    }

    fn free_page_from_free(&self) -> Option<Pin<Box<VectorPage>>> {
        let mut free = self.free.lock().unwrap();
        free.pop()
    }

    fn free_page_from_cache(&self) -> Option<Pin<Box<VectorPage>>> {
        let mut cache = self.cache.write().unwrap();
        cache.pop_lru().map(|p|p.1.page)
    }

    fn free_page(&self) -> Option<Pin<Box<VectorPage>>> {
        self.free_page_from_free()
            .or_else(||self.free_page_from_cache())
    }

    fn page_is_loaded(&self, spec: PageSpec) -> bool {
        let loaded = self.loaded.read().unwrap();
        loaded.contains_key(&spec)
    }

    fn page_is_cached(&self, spec: PageSpec) -> bool {
        let cache = self.cache.read().unwrap();
        cache.contains(&spec)
    }

    fn start_loading_or_wait(&self, spec: PageSpec) -> LoadState {
        let mut loading = self.loading.lock().unwrap();
        if let Some(x) = loading.get(&spec).map(|x|x.clone()) {
            // someone is already loading. Let's wait.
            std::mem::drop(loading);
            let (cv, m) = &*x;
            let mut load_state = m.lock().unwrap();
            while *load_state == LoadState::Loading {
                load_state = cv.wait(load_state).unwrap();
            }
            // this will now either be loaded or canceled
            *load_state
        } else {
            // doesn't seem like we're actually loading this thing yet.
            // let's make absolutely sure that we have to do this.
            // We're currently holding the loading lock, so we know for sure nobody will race us to start this load.
            // Since checking if a page is loaded or cached also locks, we have to make very sure that in other bits of code we aren't acquiring the loading lock after either the loaded or cache lock, as this could incur a lock cycle.
            // Luckily, there's only 3 functions where the loading lock is acquired, and this one is the only one where other locks are acquired as well. So we can be sure that this won't deadlock, despite the hold-and-wait.
            if self.page_is_loaded(spec) {
                LoadState::Loaded
            } else if self.page_is_cached(spec) {
                LoadState::Cached
            } else {
                loading.insert(spec, Default::default());
                LoadState::Loading
            }
        }
    }

    fn finish_loading(self: Arc<Self>, spec: PageSpec, page: Pin<Box<VectorPage>>) -> Arc<PageHandle> {
        let index = spec.index;
        let handle = Arc::new(PageHandle { spec, arena: self.clone(), p: &*page});
        let mut loaded = self.loaded.write().unwrap();
        loaded.insert(spec, PinnedVectorPage {
            handle: Arc::downgrade(&handle),
            page: LoadedVectorPage {
                index,
                page
            }});
        std::mem::drop(loaded);

        let mut loading = self.loading.lock().unwrap();
        let x = loading.remove(&spec).expect("entry that was finished loading was not in load map");
        let (cv, m) = &*x;

        let mut load_state = m.lock().unwrap();
        *load_state = LoadState::Loaded;
        cv.notify_all();

        handle
    }

    fn cancel_loading(&self, spec: PageSpec, page: Pin<Box<VectorPage>>) {
        let mut free = self.free.lock().unwrap();
        free.push(page);
        std::mem::drop(free);

        let mut loading = self.loading.lock().unwrap();
        let x = loading.remove(&spec).expect("entry that canceled loading was not in load map");
        let (cv, m) = &*x;

        let mut load_state = m.lock().unwrap();
        *load_state = LoadState::Canceled;
        cv.notify_all();
    }

    fn loaded_to_cached(&self, spec: PageSpec) -> bool {
        // We're acquiring two locks. In order to make sure there won't be deadlocks, we have to ensure that these locks are always acquired in this order.
        // Luckily, there's only two functions that need to acquire both of these locks, and we can easily verify that both do indeed acquire in this order, thus preventing deadlocks.
        let mut loaded = self.loaded.write().unwrap();
        let mut cache = self.cache.write().unwrap();

        assert!(loaded.get(&spec).expect("page that was supposedly loaded was not found in the load map")
                .handle.strong_count() == 0,
                "removed page from load map that still was referred to");
        assert!(!cache.contains(&spec),
                "page already in cache");
        let page = loaded.remove(&spec).unwrap();
        cache.get_or_insert(spec, move || page.page);
        // TODO ensure that this only happens if the loaded page wasn't referred to anymore.
        // Note that there's a race, where we might start to move to cache but find out that we actually shouldn't cause someone just acquired a new lock. So make sure this can fail.

        true
    }

    fn cached_to_loaded(self: Arc<Self>, spec: PageSpec) -> Arc<PageHandle> {
        // We're acquiring two locks. In order to make sure there won't be deadlocks, we have to ensure that these locks are always acquired in this order.
        // Luckily, there's only two functions that need to acquire both of these locks, and we can easily verify that both do indeed acquire in this order, thus preventing deadlocks.
        let mut loaded = self.loaded.write().unwrap();
        let mut cache = self.cache.write().unwrap();

        let page = cache.pop(&spec).expect("page not in cache");
        let handle = Arc::new(PageHandle { spec, arena: self.clone(), p: &*page.page });
        assert!(loaded.insert(spec, PinnedVectorPage { page, handle: Arc::downgrade(&handle) }).is_none(),
                "page from cache was already in load map");

        handle
    }
}

struct PageHandle {
    arena: Arc<PageArena>,
    spec: PageSpec,
    p: *const VectorPage
}

unsafe impl Send for PageHandle {}
unsafe impl Sync for PageHandle {}

impl Drop for PageHandle {
    fn drop(&mut self) {
        self.arena.loaded_to_cached(self.spec);
    }
}

struct VectorStore {
    dir: PathBuf,
    free: Vec<Pin<Box<VectorPage>>>,
    loaded: HashMap<PageSpec, LoadedVectorPage>,
    cache: LruCache<PageSpec, LoadedVectorPage>,
    domains: HashMap<String, Domain>
}

impl VectorStore {
    pub fn add_vecs(&mut self, domain: &str, vecs: &[Embedding]) -> io::Result<()> {
        let domain = self.get_or_add_domain(domain)?;
        domain.add_vecs(vecs)?;
        // TODO this could mean that a page is now invalid. Gotta make sure it gets updated :o
        // We know this has no effect on any loaded vectors, cause we do not overwrite things.
        // So any existing pointers out there can just remain valid. We just need to get at the right page and update it.
        Ok(())
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
            domain: domain.index,
            index,
        };
        let mut found = self.loaded.contains_key(&pageSpec);
        if !found {
            // page is not currently loaded. it might still be in cache.
            if let Some(page) = self.cache.pop(&pageSpec) {
                found = true;
                // we'll have to move it into loaded.
                self.loaded.insert(pageSpec, page);
            } else {
                // turns out it is not in cache. time to go to disk and get it from there.
                // We'll need a free page first to handle this load.
                // It can come from either the free arena, or from the cache.
                let free = self.free.pop()
                    .or_else(|| self.cache.pop_lru().map(|x|x.1.page));
                if let Some(mut free) = free {
                    let result = domain.load_page(index, &mut free);
                    if result.is_err() {
                        // whoops, something went wrong. move page back into the free arena
                        self.free.push(free);
                        return Err(result.err().unwrap());
                    }
                    let found = result.unwrap();
                    if found {
                        // great, we managed to load this page. let's move it into loaded
                        self.loaded.insert(pageSpec, LoadedVectorPage {
                            index,
                            page: free
                        });
                    } else {
                        // page not found, so move back into free arena.
                        self.free.push(free);
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
