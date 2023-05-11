use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, SeekFrom, Seek, Write};
use std::os::unix::prelude::FileExt;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, self};
use std::sync::{Arc, Weak, Condvar, Mutex, RwLock};
use std::ops::Deref;

use lru::LruCache;

use crate::openai::{Embedding, EMBEDDING_LENGTH, EmbeddingBytes, EMBEDDING_BYTE_LENGTH};

// 3 memory pages of 4K hold 2 OpenAI vectors.
// We set things up so that blocks are some multiple of 2 pages.
const VECTOR_PAGE_MULTIPLIER: usize = 1;
const VECTOR_PAGE_BYTE_SIZE: usize = VECTOR_PAGE_MULTIPLIER*3*4096;
const VECTOR_PAGE_FLOAT_SIZE: usize = VECTOR_PAGE_BYTE_SIZE/4;
const VECTORS_PER_PAGE: usize = VECTOR_PAGE_FLOAT_SIZE / EMBEDDING_LENGTH;

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
    read_file: File,
    write_file: Mutex<File>,
    num_vecs: AtomicUsize
}

impl Domain {
    fn open(dir: &PathBuf, name: &str, index: usize) -> io::Result<Self> {
        let mut path = dir.clone();
        path.push(format!("{name}.vecs"));
        let mut write_file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        let pos = write_file.seek(SeekFrom::End(0))?;
        if pos as usize % EMBEDDING_BYTE_LENGTH != 0 {
            panic!("domain {name} has unexpected length");
        }
        let num_vecs = AtomicUsize::new(pos as usize / EMBEDDING_BYTE_LENGTH);
        let write_file = Mutex::new(write_file);
        let read_file = File::options()
            .read(true)
            .write(false)
            .create(false)
            .truncate(false)
            .open(path)?;

        Ok(Domain {
            name: Arc::new(name.to_string()),
            index,
            read_file,
            write_file,
            num_vecs,
        })
    }

    fn add_vecs<'a, I:Iterator<Item=&'a Embedding>>(&self, vecs: I) -> io::Result<()> {
        let mut write_file = self.write_file.lock().unwrap();
        let mut count = 0;
        for embedding in vecs {
            let bytes: &EmbeddingBytes = unsafe {std::mem::transmute(embedding)};
            write_file.write_all(bytes)?;
            count += 1;
        }
        write_file.flush()?;
        write_file.sync_data()?;
        let mut num_vecs = self.num_vecs.load(atomic::Ordering::Relaxed);
        num_vecs += count;
        self.num_vecs.store(num_vecs, atomic::Ordering::Relaxed);

        Ok(())
    }

    fn load_page(&self, index: usize, data: &mut VectorPage) -> io::Result<bool> {
        let offset = index * std::mem::size_of::<VectorPage>();
        let end = self.num_vecs() * std::mem::size_of::<Embedding>();
        if end <= offset {
            // this page does not exist.
            return Ok(false);
        }
        let remainder = end - offset;
        let data_len = if remainder >= std::mem::size_of::<Embedding>() { std::mem::size_of::<Embedding>() } else { remainder };
        let data: &mut VectorPageBytes = unsafe { std::mem::transmute(data) };
        let data_slice = &mut data[..data_len];
        self.read_file.read_exact_at(data_slice, offset as u64)?;

        Ok(true)
    }

    fn load_partial_page(&self, index: usize, offset: usize, data: &mut [u8]) -> io::Result<()> {
        assert!(offset + data.len() <= std::mem::size_of::<VectorPage>(),
                "requested partial load would read past a page boundary");
        let offset = index * std::mem::size_of::<VectorPage>() + offset;
        self.read_file.read_exact_at(data, offset as u64)
    }

    fn num_vecs(&self) -> usize {
        self.num_vecs.load(atomic::Ordering::Relaxed)
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

        if loaded.get(&spec).expect("page that was supposedly loaded was not found in the load map")
            .handle.strong_count() != 0 {
                // Whoops! Looks like someone re-acquired this page while we weren't looking!
                // Best to leave it alone.
                return false;
            }
        assert!(!cache.contains(&spec),
                "page already in cache");
        let page = loaded.remove(&spec).unwrap();
        cache.get_or_insert(spec, move || page.page);

        true
    }

    fn cached_to_loaded(self: Arc<Self>, spec: PageSpec) -> Option<Arc<PageHandle>> {
        // We're acquiring two locks. In order to make sure there won't be deadlocks, we have to ensure that these locks are always acquired in this order.
        // Luckily, there's only two functions that need to acquire both of these locks, and we can easily verify that both do indeed acquire in this order, thus preventing deadlocks.
        let mut loaded = self.loaded.write().unwrap();
        let mut cache = self.cache.write().unwrap();

        let page = cache.pop(&spec);
        if page.is_none() {
            return None;
        }
        let page = page.unwrap();
        let handle = Arc::new(PageHandle { spec, arena: self.clone(), p: &*page.page });
        assert!(loaded.insert(spec, PinnedVectorPage { page, handle: Arc::downgrade(&handle) }).is_none(),
                "page from cache was already in load map");

        Some(handle)
    }

    fn page_from_loaded(self: Arc<Self>, spec: PageSpec) -> Option<Arc<PageHandle>> {
        // so tricky bit here is that we wish to load a page whose refcount could just have gone down to 0, but for which the drop on the handle hasn't run yet.
        // we will have to solve this race condition.
        // basically, while holding the lock we have to replace the pagehandle (as the original one cannot be safely upgraded anymore). We also have to inhibit the move to cache that was triggered.
        // this is just a small race condition window but it is there. so best make sure.
        let loaded = self.loaded.read().unwrap();
        if let Some(page) = loaded.get(&spec) {
            if let Some(handle) = page.handle.upgrade() {
                Some(handle)
            } else {
                // Uh oh, this handle was dropped but somehow we still encountered this page in the loaded map.
                // That means someone is right around the corner to move this page into the cache. We gotta stop them.
                std::mem::drop(loaded);
                let mut loaded = self.loaded.write().unwrap();
                if let Some(page) = loaded.get_mut(&spec) {
                    // ok it is still here. To be absolutely sure, we have to recheck the lock
                    if let Some(handle) = page.handle.upgrade() {
                        // someone got here before us and reacquired the page.
                        Some(handle)
                    } else {
                        // we got here first! create new handle.
                        let handle = Arc::new(PageHandle { spec, arena: self.clone(), p: &*page.page.page });
                        page.handle = Arc::downgrade(&handle);
                        Some(handle)
                    }
                } else {
                    // In the brief window between dropping the read lock and acquiring the write lock, this page got moved into cache.
                    None
                }
            }
        } else {
            None
        }
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

impl PageHandle {
    pub fn get_vec(&self, index: usize) -> &Embedding {
        if index >= VECTORS_PER_PAGE {
            panic!("index bigger than max vectors per page ({}): {}", VECTORS_PER_PAGE, index);
        }

        // This pointer should be valid, because the only way for
        // people to acquire pagehandles is through an interface that
        // returns the pagehandle as an arc, and the page doesn't get
        // moved out of the loaded pages unless this arc's refcount is
        // 0.
        unsafe {&*(self.p as *const Embedding).offset(index as isize)}
    }

    pub fn get_loaded_vec(self: Arc<Self>, index: usize) -> LoadedVec {
        if index >= VECTORS_PER_PAGE {
            panic!("index bigger than max vectors per page ({}): {}", VECTORS_PER_PAGE, index);
        }

        let vec = unsafe {(self.p as *const Embedding).offset(index as isize) };
        LoadedVec {
            page: self.clone(),
            vec
        }
    }

    pub unsafe fn get_page_mut(&self) -> &mut [Embedding;VECTORS_PER_PAGE] {
        let page = self.p as *mut [Embedding;VECTORS_PER_PAGE];
        &mut *page
    }
}

pub struct LoadedVec {
    page: Arc<PageHandle>,
    vec: *const Embedding
}

impl Deref for LoadedVec {
    type Target = Embedding;

    fn deref(&self) -> &Self::Target {
        // This pointer should be valid, because the only way for the
        // underlying page to move out of the load map is if the
        // pagehandle arc has no more strong references. Since we
        // ourselves hold one such reference, this won't happen for
        // the lifetime of LoadedVecl.


        unsafe {&*self.vec}
    }
}

struct VectorStore {
    dir: PathBuf,
    arena: Arc<PageArena>,
    domains: HashMap<String, Domain>
}

impl VectorStore {
    pub fn add_vecs(&mut self, domain: &str, vecs: &[Embedding]) -> io::Result<()> {
        let domain = self.get_or_add_domain(domain)?;
        domain.add_vecs(vecs.iter())?;
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
}
