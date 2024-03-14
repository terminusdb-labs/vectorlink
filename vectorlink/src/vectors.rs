#![allow(unused)]

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::os::unix::fs::MetadataExt;
use std::os::unix::prelude::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{self, AtomicUsize};
use std::sync::{Arc, Condvar, Mutex, RwLock, Weak};

use lru::LruCache;
use rand::{thread_rng, Rng};
use rayon::iter::plumbing::{bridge_producer_consumer, Producer};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use serde::Serialize;
use urlencoding::encode;

use crate::comparator::Centroid32Comparator;
use crate::vecmath::{
    Centroid32, Embedding, EmbeddingBytes, CENTROID_32_LENGTH, EMBEDDING_BYTE_LENGTH,
    EMBEDDING_LENGTH, QUANTIZED_EMBEDDING_LENGTH,
};
use parallel_hnsw::pq::HnswQuantizer;

// 3 memory pages of 4K hold 2 OpenAI vectors.
// We set things up so that blocks are some multiple of 2 pages.
const VECTOR_PAGE_MULTIPLIER: usize = 1;
const VECTOR_PAGE_BYTE_SIZE: usize = VECTOR_PAGE_MULTIPLIER * 3 * 4096;
const VECTOR_PAGE_FLOAT_SIZE: usize = VECTOR_PAGE_BYTE_SIZE / 4;
const VECTORS_PER_PAGE: usize = VECTOR_PAGE_FLOAT_SIZE / EMBEDDING_LENGTH;

type VectorPage = [f32; VECTOR_PAGE_FLOAT_SIZE];
type VectorPageBytes = [u8; VECTOR_PAGE_BYTE_SIZE];

struct LoadedVectorPage {
    index: usize,
    page: Box<VectorPage>,
}

struct PinnedVectorPage {
    page: LoadedVectorPage,
    handle: Weak<PageHandle>,
}

pub struct QuantizedDomain {
    vecs: Arc<RwLock<Vec<Centroid32>>>,
    file: Arc<Mutex<File>>,
    quantizer: HnswQuantizer<
        EMBEDDING_LENGTH,
        CENTROID_32_LENGTH,
        QUANTIZED_EMBEDDING_LENGTH,
        Centroid32Comparator,
    >,
}

pub struct Domain {
    name: Arc<String>,
    index: usize,
    read_file: File,
    write_file: Mutex<File>,
    num_vecs: AtomicUsize,
    quantized: Option<QuantizedDomain>,
}

impl Domain {
    pub fn name(&self) -> &str {
        &self.name
    }

    fn open(dir: &Path, name: &str, index: usize) -> io::Result<Self> {
        let mut path = dir.to_path_buf();
        let encoded_name = encode(name);
        path.push(format!("{encoded_name}.vecs"));
        let mut write_file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dbg!(&path))?;
        let pos = write_file.seek(SeekFrom::End(0))?;
        if pos as usize % EMBEDDING_BYTE_LENGTH != 0 {
            panic!("domain {encoded_name} has unexpected length");
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
            quantized: None,
        })
    }

    fn add_vecs<'a, I: Iterator<Item = &'a Embedding>>(
        &self,
        vecs: I,
    ) -> io::Result<(usize, usize)> {
        let mut write_file = self.write_file.lock().unwrap();
        let mut count = 0;
        for embedding in vecs {
            let bytes: &EmbeddingBytes = unsafe { std::mem::transmute(embedding) };
            write_file.write_all(bytes)?;
            count += 1;
            if let Some(q) = self.quantized.as_ref() {
                // quantize
                //q.quantizer.quantize();
            }
        }
        write_file.flush()?;
        write_file.sync_data()?;
        let num_vecs = self.num_vecs.load(atomic::Ordering::Relaxed);
        let new_num_vecs = num_vecs + count;
        self.num_vecs.store(new_num_vecs, atomic::Ordering::Relaxed);

        Ok((num_vecs, count))
    }

    fn load_page(&self, index: usize, data: &mut VectorPage) -> io::Result<bool> {
        let offset = index * std::mem::size_of::<VectorPage>();
        let end = self.num_vecs() * std::mem::size_of::<Embedding>();
        if end <= offset {
            // this page does not exist.
            //eprintln!(
            //    " :( page {} does not exist (need {} but end is {})",
            //    index, offset, end
            //);
            return Ok(false);
        }
        let remainder = end - offset;
        let data_len = if remainder >= std::mem::size_of::<VectorPage>() {
            std::mem::size_of::<VectorPage>()
        } else {
            remainder
        };
        // eprintln!(
        //    "loading page {}, range at offset {} of len {}",
        //    index, offset, data_len
        //);
        let data: &mut VectorPageBytes = unsafe { std::mem::transmute(data) };
        let data_slice = &mut data[..data_len];
        self.read_file.read_exact_at(data_slice, offset as u64)?;

        Ok(true)
    }

    fn load_partial_page(&self, index: usize, offset: usize, data: &mut [u8]) -> io::Result<()> {
        assert!(
            offset + data.len() <= std::mem::size_of::<VectorPage>(),
            "requested partial load would read past a page boundary"
        );
        let offset = index * std::mem::size_of::<VectorPage>() + offset;
        // eprintln!(
        //    "loading partial range at offset {} of len {}",
        //    offset,
        //    data.len()
        //);
        self.read_file.read_exact_at(data, offset as u64)
    }

    pub fn concatenate_file<P: AsRef<Path>>(&self, path: P) -> io::Result<usize> {
        let size = self.num_vecs();
        let mut wfl = self.write_file.lock().unwrap();
        wfl.seek(SeekFrom::End(0))?;
        let mut read_file = File::options().read(true).open(&path)?;
        io::copy(&mut read_file, &mut *wfl)?;
        Ok(size)
    }

    pub fn num_vecs(&self) -> usize {
        self.num_vecs.load(atomic::Ordering::Relaxed)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone, Copy)]
struct PageSpec {
    domain: usize,
    index: usize,
}

type GuardedLoadState = Arc<(Condvar, Mutex<LoadState>)>;

struct PageArena {
    free: Mutex<Vec<Box<VectorPage>>>,
    loading: Mutex<HashMap<PageSpec, GuardedLoadState>>,
    loaded: RwLock<HashMap<PageSpec, PinnedVectorPage>>,
    cache: RwLock<LruCache<PageSpec, LoadedVectorPage>>,
}

#[derive(Default, Clone)]
enum LoadState {
    #[default]
    Loading,
    Loaded(Arc<PageHandle>),
    Canceled,
}

impl LoadState {
    fn is_loading(&self) -> bool {
        matches!(self, Self::Loading)
    }
}

impl Default for PageArena {
    fn default() -> Self {
        Self {
            free: Default::default(),
            loading: Default::default(),
            loaded: Default::default(),
            cache: RwLock::new(LruCache::unbounded()),
        }
    }
}

impl PageArena {
    fn new() -> Self {
        Self::default()
    }

    fn alloc_free_pages(&self, count: usize) {
        if count == 0 {
            return;
        }
        // TODO would be much better if we could have uninit allocs.
        let mut free = self.free.lock().unwrap();
        free.reserve(count);
        let zeroed = Box::new([0.0f32; VECTOR_PAGE_FLOAT_SIZE]);
        for _ in 0..count - 1 {
            free.push(zeroed.clone());
        }
        free.push(zeroed);
    }

    fn free_page_from_free(&self) -> Option<Box<VectorPage>> {
        let mut free = self.free.lock().unwrap();
        free.pop()
    }

    fn free_page_from_cache(&self) -> Option<Box<VectorPage>> {
        let mut cache = self.cache.write().unwrap();
        cache.pop_lru().map(|p| p.1.page)
    }

    fn free_page(&self) -> Option<Box<VectorPage>> {
        self.free_page_from_free()
            .or_else(|| self.free_page_from_cache())
    }

    fn page_is_loaded(&self, spec: PageSpec) -> bool {
        let loaded = self.loaded.read().unwrap();
        loaded.contains_key(&spec)
    }

    fn page_is_cached(&self, spec: PageSpec) -> bool {
        let cache = self.cache.read().unwrap();
        cache.contains(&spec)
    }

    fn start_loading_or_wait(self: &Arc<Self>, spec: PageSpec) -> LoadState {
        let mut loading = self.loading.lock().unwrap();
        if let Some(x) = loading.get(&spec).cloned() {
            // someone is already loading. Let's wait.
            std::mem::drop(loading);
            let (cv, m) = &*x;
            let mut load_state = m.lock().unwrap();
            while load_state.is_loading() {
                load_state = cv.wait(load_state).unwrap();
            }
            // this will now either be loaded or canceled
            load_state.clone()
        } else {
            // doesn't seem like we're actually loading this thing yet.
            // let's make absolutely sure that we have to do this.
            // We're currently holding the loading lock, so we know for sure nobody will race us to start this load.
            // Since checking if a page is loaded or cached also locks, we have to make very sure that in other bits of code we aren't acquiring the loading lock after either the loaded or cache lock, as this could incur a lock cycle.
            // Luckily, there's only 3 functions where the loading lock is acquired, and this one is the only one where other locks are acquired as well. So we can be sure that this won't deadlock, despite the hold-and-wait.
            if let Some(handle) = self.page_from_any(spec) {
                LoadState::Loaded(handle)
            } else {
                loading.insert(spec, Default::default());
                LoadState::Loading
            }
        }
    }

    fn finish_loading(self: &Arc<Self>, spec: PageSpec, page: Box<VectorPage>) -> Arc<PageHandle> {
        let index = spec.index;
        let handle = Arc::new(PageHandle {
            spec,
            arena: self.clone(),
            p: &*page,
        });
        let mut loaded = self.loaded.write().unwrap();
        assert!(loaded
            .insert(
                spec,
                PinnedVectorPage {
                    handle: Arc::downgrade(&handle),
                    page: LoadedVectorPage { index, page },
                },
            )
            .is_none());
        std::mem::drop(loaded);

        let mut loading = self.loading.lock().unwrap();
        let x = loading
            .remove(&spec)
            .expect("entry that was finished loading was not in load map");
        let (cv, m) = &*x;

        let mut load_state = m.lock().unwrap();
        *load_state = LoadState::Loaded(handle.clone());
        cv.notify_all();

        handle
    }

    fn cancel_loading(&self, spec: PageSpec, page: Box<VectorPage>) {
        let mut free = self.free.lock().unwrap();
        free.push(page);
        std::mem::drop(free);

        let mut loading = self.loading.lock().unwrap();
        let x = loading
            .remove(&spec)
            .expect("entry that canceled loading was not in load map");
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

        if loaded
            .get(&spec)
            .expect("page that was supposedly loaded was not found in the load map")
            .handle
            .strong_count()
            != 0
        {
            // Whoops! Looks like someone re-acquired this page while we weren't looking!
            // Best to leave it alone.
            return false;
        }
        assert!(!cache.contains(&spec), "page already in cache");
        let page = loaded.remove(&spec).unwrap();
        cache.get_or_insert(spec, move || page.page);

        true
    }

    fn cached_to_loaded(self: &Arc<Self>, spec: PageSpec) -> Option<Arc<PageHandle>> {
        // We're acquiring two locks. In order to make sure there won't be deadlocks, we have to ensure that these locks are always acquired in this order.
        // Luckily, there's only two functions that need to acquire both of these locks, and we can easily verify that both do indeed acquire in this order, thus preventing deadlocks.
        let mut loaded = self.loaded.write().unwrap();
        let mut cache = self.cache.write().unwrap();

        let page = cache.pop(&spec);
        page.as_ref()?;
        let page = page.unwrap();
        let handle = Arc::new(PageHandle {
            spec,
            arena: self.clone(),
            p: &*page.page,
        });
        assert!(
            loaded
                .insert(
                    spec,
                    PinnedVectorPage {
                        page,
                        handle: Arc::downgrade(&handle)
                    }
                )
                .is_none(),
            "page from cache was already in load map"
        );

        Some(handle)
    }

    fn page_from_loaded(self: &Arc<Self>, spec: PageSpec) -> Option<Arc<PageHandle>> {
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
                        let handle = Arc::new(PageHandle {
                            spec,
                            arena: self.clone(),
                            p: &*page.page.page,
                        });
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

    pub fn page_from_any(self: &Arc<Self>, spec: PageSpec) -> Option<Arc<PageHandle>> {
        self.page_from_loaded(spec)
            .or_else(|| self.cached_to_loaded(spec))
    }

    pub fn statistics(&self) -> VectorStoreStatistics {
        let free = self.free.lock().unwrap().len();
        let loading = self.loading.lock().unwrap().len();
        let loaded = self.loaded.read().unwrap().len();
        let cached = self.cache.read().unwrap().len();

        VectorStoreStatistics {
            free,
            loading,
            loaded,
            cached,
        }
    }
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorStoreStatistics {
    free: usize,
    loading: usize,
    loaded: usize,
    cached: usize,
}

struct PageHandle {
    arena: Arc<PageArena>,
    spec: PageSpec,
    p: *const VectorPage,
}

unsafe impl Send for PageHandle {}
unsafe impl Sync for PageHandle {}

impl Drop for PageHandle {
    fn drop(&mut self) {
        // self.arena.loaded_to_cached(self.spec);
    }
}

impl PageHandle {
    pub fn get_vec(&self, index: usize) -> &Embedding {
        if index >= VECTORS_PER_PAGE {
            panic!(
                "index bigger than max vectors per page ({}): {}",
                VECTORS_PER_PAGE, index
            );
        }

        // This pointer should be valid, because the only way for
        // people to acquire pagehandles is through an interface that
        // returns the pagehandle as an arc, and the page doesn't get
        // moved out of the loaded pages unless this arc's refcount is
        // 0.
        unsafe { &*(self.p as *const Embedding).add(index) }
    }

    pub fn get_loaded_vec(self: &Arc<Self>, index: usize) -> LoadedVec {
        if index >= VECTORS_PER_PAGE {
            panic!(
                "index bigger than max vectors per page ({}): {}",
                VECTORS_PER_PAGE, index
            );
        }

        let vec = unsafe { (self.p as *const Embedding).add(index) };
        LoadedVec {
            page: self.clone(),
            vec,
        }
    }
}

#[derive(Clone)]
pub struct LoadedVec {
    page: Arc<PageHandle>,
    vec: *const Embedding,
}

impl LoadedVec {
    pub fn id(&self) -> usize {
        let page_offset = self.page.spec.index * VECTORS_PER_PAGE;
        let offset_in_page =
            (self.vec as usize - self.page.p as usize) / std::mem::size_of::<Embedding>();

        page_offset + offset_in_page
    }
}

unsafe impl Send for LoadedVec {}
unsafe impl Sync for LoadedVec {}

impl fmt::Debug for LoadedVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LoadedVec({:?})", self.vec)
    }
}

impl Deref for LoadedVec {
    type Target = Embedding;

    fn deref(&self) -> &Self::Target {
        // This pointer should be valid, because the only way for the
        // underlying page to move out of the load map is if the
        // pagehandle arc has no more strong references. Since we
        // ourselves hold one such reference, this won't happen for
        // the lifetime of LoadedVecl.

        unsafe { &*self.vec }
    }
}

impl PartialEq for LoadedVec {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

pub struct VectorStore {
    dir: PathBuf,
    arena: Arc<PageArena>,
    domains: RwLock<HashMap<String, Arc<Domain>>>,
}

impl VectorStore {
    pub fn new<P: Into<PathBuf>>(path: P, num_bufs: usize) -> Self {
        let arena = PageArena::new();
        arena.alloc_free_pages(num_bufs);

        Self {
            dir: path.into(),
            arena: Arc::new(arena),
            domains: Default::default(),
        }
    }

    pub fn get_domain(&self, name: &str) -> io::Result<Arc<Domain>> {
        let domains = self.domains.read().unwrap();
        if let Some(domain) = domains.get(name) {
            Ok(domain.clone())
        } else {
            std::mem::drop(domains);
            let mut domains = self.domains.write().unwrap();
            if let Some(domain) = domains.get(name) {
                Ok(domain.clone())
            } else {
                let domain = Arc::new(Domain::open(&self.dir, name, domains.len())?);
                domains.insert(name.to_string(), domain.clone());

                Ok(domain)
            }
        }
    }

    pub fn add_vecs<'a, I: Iterator<Item = &'a Embedding>>(
        &self,
        domain: &Domain,
        vecs: I,
    ) -> io::Result<Vec<usize>> {
        let (offset, num_added) = domain.add_vecs(vecs)?;
        if offset % VECTORS_PER_PAGE != 0 {
            // vecs got added to a page that might actually already be in memory. We'll have to refresh it.
            let page_index = offset / VECTORS_PER_PAGE;
            let page_spec = PageSpec {
                domain: domain.index,
                index: page_index,
            };

            if let Some(existing_page) = self.arena.page_from_any(page_spec) {
                // yup, that page existed. We'll partially read the file to get the vecs.
                let offset_in_page = offset - (page_index * VECTORS_PER_PAGE);
                let remainder_in_page = VECTORS_PER_PAGE - offset_in_page;
                let vecs_to_load = if num_added >= remainder_in_page {
                    remainder_in_page
                } else {
                    num_added
                };
                let data: &mut VectorPageBytes = unsafe {
                    std::mem::transmute(
                        &mut *(existing_page.p as *mut [Embedding; VECTORS_PER_PAGE]),
                    )
                };
                let offset_byte = offset_in_page * std::mem::size_of::<Embedding>();
                let end_byte = offset_byte + vecs_to_load * std::mem::size_of::<Embedding>();
                let mutation_range = &mut data[offset_byte..end_byte];
                domain.load_partial_page(page_index, offset_byte, mutation_range)?;
            }
        }
        Ok((offset..offset + num_added).collect())
    }

    pub fn get_random_vectors(&self, domain: &Domain, count: usize) -> io::Result<Vec<Embedding>> {
        let mut rng = thread_rng();
        let total = domain.num_vecs();
        let count = usize::min(total - 1, count);
        let mut candidates: HashSet<usize> = HashSet::new();
        assert!(total >= count);
        loop {
            if candidates.len() == count {
                break;
            } else {
                let res = rng.gen_range(0..total);
                candidates.insert(res);
            }
        }
        let mut vecs = Vec::with_capacity(count);

        for k in candidates {
            let v = *self.get_vec(domain, k)?.unwrap();
            vecs.push(v)
        }
        Ok(vecs)
    }

    pub fn get_vec(&self, domain: &Domain, index: usize) -> io::Result<Option<LoadedVec>> {
        if domain.num_vecs() <= index {
            return Ok(None);
        }

        let page_index = index / VECTORS_PER_PAGE;
        let index_in_page = index % VECTORS_PER_PAGE;
        let page_spec = PageSpec {
            domain: domain.index,
            index: page_index,
        };
        if let Some(page) = self.arena.page_from_any(page_spec) {
            Ok(Some(page.get_loaded_vec(index_in_page)))
        } else {
            // the page is on disk but not yet in memory. Let's load it.
            match self.arena.start_loading_or_wait(page_spec) {
                LoadState::Loading => {
                    //eprintln!(" loading page");
                    // we are the loader. get a free page and load things
                    if let Some(mut page) = self.arena.free_page() {
                        match domain.load_page(page_index, &mut page) {
                            Ok(true) => {
                                let handle = self.arena.finish_loading(page_spec, page);
                                Ok(Some(handle.get_loaded_vec(index_in_page)))
                            }
                            Ok(false) => {
                                Err(io::Error::new(io::ErrorKind::NotFound, "page not found"))
                            }
                            Err(e) => {
                                // something went wrong. cancel the load
                                self.arena.cancel_loading(page_spec, page);
                                Err(e)
                            }
                        }
                    } else {
                        Err(io::Error::new(
                            io::ErrorKind::Other,
                            "no more free space in vector arena",
                        ))
                    }
                }
                LoadState::Loaded(page) => Ok(Some(page.get_loaded_vec(index_in_page))),
                LoadState::Canceled => {
                    Err(io::Error::new(io::ErrorKind::Other, "load was canceled"))
                }
            }
        }
    }
    pub fn add_and_load_vecs<'a, I: Iterator<Item = &'a Embedding>>(
        &self,
        domain: &Domain,
        vecs: I,
    ) -> io::Result<Vec<LoadedVec>> {
        let ids = self.add_vecs(domain, vecs)?;

        let mut result = Vec::with_capacity(ids.len());
        for id in ids.into_iter() {
            let e = self.get_vec(domain, id)?.unwrap();
            result.push(e);
        }

        Ok(result)
    }

    pub fn add_and_load_vec(&self, domain: &Domain, vec: &Embedding) -> io::Result<LoadedVec> {
        let ids = self.add_vecs(domain, [vec].into_iter())?;

        Ok(self.get_vec(domain, ids[0])?.unwrap())
    }

    pub fn add_and_load_vec_array<const N: usize>(
        &self,
        domain: &Domain,
        embeddings: &[Embedding; N],
    ) -> io::Result<[LoadedVec; N]> {
        let ids = self.add_vecs(domain, embeddings.iter())?;

        let mut result: [MaybeUninit<LoadedVec>; N] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for (r, id) in result.iter_mut().zip(ids.into_iter()) {
            let e: LoadedVec = self.get_vec(domain, id)?.unwrap();
            r.write(e);
        }

        // It would be nicer if we could do a transmute here, as
        // transmute ensures that the conversion converts between
        // types of the same size, but it seems like this doesn't work
        // yet with const generic arrays. We do a pointer cast
        // instead.
        let magic = result.as_ptr() as *const [LoadedVec; N];
        std::mem::forget(result);
        let result = unsafe { magic.read() };

        Ok(result)
    }

    pub fn statistics(&self) -> VectorStoreStatistics {
        self.arena.statistics()
    }

    pub fn dir(&self) -> String {
        self.dir.to_str().unwrap().to_string()
    }
}

pub struct EmbeddingFileProducer<'a> {
    file: &'a std::fs::File,
    pos_front: usize,
    pos_back: usize,
}

impl<'a> EmbeddingFileProducer<'a> {
    pub fn new(file: &'a std::fs::File) -> Self {
        // figure out how large the file is, set pos_back appropriately
        let metadata = file.metadata().unwrap();
        assert!(metadata.size() % EMBEDDING_BYTE_LENGTH as u64 == 0);
        let len = metadata.size() / EMBEDDING_BYTE_LENGTH as u64;
        Self {
            file,
            pos_front: 0,
            pos_back: len as usize,
        }
    }
}

impl<'a> Iterator for EmbeddingFileProducer<'a> {
    type Item = Embedding;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos_front < self.pos_back {
            let mut result = [0_u8; EMBEDDING_BYTE_LENGTH];
            let byte_pos_front = self.pos_front * EMBEDDING_BYTE_LENGTH;
            self.file
                .read_exact_at(&mut result, byte_pos_front as u64)
                .unwrap_or_else(|e| {
                    panic!(
                        "reading embedding file at pos_front {} failed: {e:?}",
                        self.pos_front
                    );
                });
            self.pos_front += 1;
            Some(unsafe { std::mem::transmute(result) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.pos_back - self.pos_front;
        (size, Some(size))
    }
}

impl<'a> DoubleEndedIterator for EmbeddingFileProducer<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.pos_back > self.pos_front {
            self.pos_back -= 1;
            let mut result = [0_u8; EMBEDDING_BYTE_LENGTH];
            let byte_pos_back = self.pos_back * EMBEDDING_BYTE_LENGTH;
            self.file
                .read_exact_at(&mut result, byte_pos_back as u64)
                .unwrap_or_else(|e| {
                    panic!(
                        "reading embedding file at pos_back {} failed: {e:?}",
                        self.pos_back
                    );
                });
            Some(unsafe { std::mem::transmute(result) })
        } else {
            None
        }
    }
}

impl<'a> ExactSizeIterator for EmbeddingFileProducer<'a> {
    fn len(&self) -> usize {
        let (lower, upper) = self.size_hint();
        // Note: This assertion is overly defensive, but it checks the invariant
        // guaranteed by the trait. If this trait were rust-internal,
        // we could use debug_assert!; assert_eq! will check all Rust user
        // implementations too.
        assert_eq!(upper, Some(lower));
        lower
    }
}

impl<'a> Producer for EmbeddingFileProducer<'a> {
    type Item = Embedding;

    type IntoIter = Self;

    fn into_iter(self) -> Self::IntoIter {
        self
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let len = <Self as ExactSizeIterator>::len(&self);
        assert!(index < len);
        (
            Self {
                file: self.file,
                pos_front: self.pos_front,
                pos_back: self.pos_front + index,
            },
            Self {
                file: self.file,
                pos_front: self.pos_front + index + 1,
                pos_back: self.pos_back,
            },
        )
    }
}

impl<'a> ParallelIterator for EmbeddingFileProducer<'a> {
    type Item = Embedding;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        let len = <Self as ExactSizeIterator>::len(&self);
        bridge_producer_consumer(len, self, consumer)
    }
}

impl<'a> IndexedParallelIterator for EmbeddingFileProducer<'a> {
    fn len(&self) -> usize {
        <Self as ExactSizeIterator>::len(self)
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        let len = <Self as ExactSizeIterator>::len(&self);
        bridge_producer_consumer(len, self, consumer)
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        callback.callback(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::vecmath::random_embedding;

    use super::*;

    use rand::prelude::*;
    use rand::SeedableRng;

    #[test]
    fn create_and_load_vecs() {
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path();
        //let path = "/tmp/foo";
        let store = VectorStore::new(path, 100);
        let seed: u64 = 42;

        let mut rng = StdRng::seed_from_u64(seed);
        let e1 = random_embedding(&mut rng);
        let e2 = random_embedding(&mut rng);
        let e3 = random_embedding(&mut rng);

        let domain = store.get_domain("foo").unwrap();
        let ids = store.add_vecs(&domain, [e1, e2, e3].iter()).unwrap();
        assert_eq!(vec![0, 1, 2], ids);

        let e1_from_mem = store.get_vec(&domain, 0).unwrap().unwrap();
        let e2_from_mem = store.get_vec(&domain, 1).unwrap().unwrap();
        let e3_from_mem = store.get_vec(&domain, 2).unwrap().unwrap();

        assert_eq!(e1, *e1_from_mem);
        assert_eq!(e2, *e2_from_mem);
        assert_eq!(e3, *e3_from_mem);
        assert_eq!(0, e1_from_mem.id());
        assert_eq!(1, e2_from_mem.id());
        assert_eq!(2, e3_from_mem.id());

        let store2 = VectorStore::new(path, 100);
        let e1_from_disk = store2.get_vec(&domain, 0).unwrap().unwrap();
        let e2_from_disk = store2.get_vec(&domain, 1).unwrap().unwrap();
        let e3_from_disk = store2.get_vec(&domain, 2).unwrap().unwrap();

        assert_eq!(e1, *e1_from_disk);
        assert_eq!(e2, *e2_from_disk);
        assert_eq!(e3, *e3_from_disk);
    }

    #[test]
    fn load_incomplete_page_twice() {
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path();
        let store = VectorStore::new(path, 100);
        let seed: u64 = 42;

        let mut rng = StdRng::seed_from_u64(seed);
        let e1 = random_embedding(&mut rng);
        let e2 = random_embedding(&mut rng);

        let domain = store.get_domain("foo").unwrap();
        store.add_vecs(&domain, [e1].iter()).unwrap();
        let e1_from_mem = store.get_vec(&domain, 0).unwrap().unwrap();
        store.add_vecs(&domain, [e2].iter()).unwrap();
        let e2_from_mem = store.get_vec(&domain, 1).unwrap().unwrap();

        assert_eq!(e1, *e1_from_mem);
        assert_eq!(e2, *e2_from_mem);
    }

    #[test]
    fn load_from_cache() {
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path();
        let store = VectorStore::new(path, 100);
        let seed: u64 = 42;

        assert_eq!(
            VectorStoreStatistics {
                free: 100,
                loading: 0,
                loaded: 0,
                cached: 0
            },
            store.statistics()
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let e1 = random_embedding(&mut rng);

        let domain = store.get_domain("foo").unwrap();
        store.add_vecs(&domain, [e1].iter()).unwrap();
        let e1_from_mem = store.get_vec(&domain, 0).unwrap().unwrap();
        assert_eq!(
            VectorStoreStatistics {
                free: 99,
                loading: 0,
                loaded: 1,
                cached: 0
            },
            store.statistics()
        );
        std::mem::drop(e1_from_mem);
        assert_eq!(
            VectorStoreStatistics {
                free: 99,
                loading: 0,
                loaded: 0,
                cached: 1
            },
            store.statistics()
        );
        let _e1_from_mem = store.get_vec(&domain, 0).unwrap().unwrap();

        assert_eq!(
            VectorStoreStatistics {
                free: 99,
                loading: 0,
                loaded: 1,
                cached: 0
            },
            store.statistics()
        );
    }

    #[test]
    fn reload_from_disk() {
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path();
        let store = VectorStore::new(path, 1);
        let seed: u64 = 42;

        assert_eq!(
            VectorStoreStatistics {
                free: 1,
                loading: 0,
                loaded: 0,
                cached: 0
            },
            store.statistics()
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let e1 = random_embedding(&mut rng);
        let e2 = random_embedding(&mut rng);
        let e3 = random_embedding(&mut rng);

        let domain = store.get_domain("foo").unwrap();
        store.add_vecs(&domain, [e1, e2, e3].iter()).unwrap();
        let e1_from_mem = store.get_vec(&domain, 0).unwrap().unwrap();
        assert_eq!(
            VectorStoreStatistics {
                free: 0,
                loading: 0,
                loaded: 1,
                cached: 0
            },
            store.statistics()
        );
        std::mem::drop(e1_from_mem);
        assert_eq!(
            VectorStoreStatistics {
                free: 0,
                loading: 0,
                loaded: 0,
                cached: 1
            },
            store.statistics()
        );
        let e3_from_mem = store.get_vec(&domain, 2).unwrap().unwrap();

        assert_eq!(
            VectorStoreStatistics {
                free: 0,
                loading: 0,
                loaded: 1,
                cached: 0
            },
            store.statistics()
        );
        assert_eq!(e3, *e3_from_mem);
    }

    #[test]
    fn add_and_load_single() {
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path();
        let store = VectorStore::new(path, 100);
        let seed: u64 = 42;
        let mut rng = StdRng::seed_from_u64(seed);
        let domain = store.get_domain("foo").unwrap();

        let e1 = random_embedding(&mut rng);
        let e1_from_mem = store.add_and_load_vec(&domain, &e1).unwrap();

        assert_eq!(e1, *e1_from_mem);
    }

    #[test]
    fn add_and_load_vec() {
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path();
        let store = VectorStore::new(path, 100);
        let seed: u64 = 42;
        let mut rng = StdRng::seed_from_u64(seed);
        let domain = store.get_domain("foo").unwrap();

        let e1 = random_embedding(&mut rng);
        let e2 = random_embedding(&mut rng);
        let e3 = random_embedding(&mut rng);
        let e4 = random_embedding(&mut rng);
        let e5 = random_embedding(&mut rng);
        let result = store
            .add_and_load_vecs(&domain, [e1, e2, e3, e4, e5].iter())
            .unwrap();

        assert_eq!(e1, *result[0]);
        assert_eq!(e2, *result[1]);
        assert_eq!(e3, *result[2]);
        assert_eq!(e4, *result[3]);
        assert_eq!(e5, *result[4]);
    }

    #[test]
    fn add_and_load_array() {
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path();
        let store = VectorStore::new(path, 100);
        let seed: u64 = 42;
        let mut rng = StdRng::seed_from_u64(seed);
        let domain = store.get_domain("foo").unwrap();

        let e1 = random_embedding(&mut rng);
        let e2 = random_embedding(&mut rng);
        let e3 = random_embedding(&mut rng);
        let e4 = random_embedding(&mut rng);
        let e5 = random_embedding(&mut rng);
        let [e1_from_memory, e2_from_memory, e3_from_memory, e4_from_memory, e5_from_memory] =
            store
                .add_and_load_vec_array(&domain, &[e1, e2, e3, e4, e5])
                .unwrap();

        assert_eq!(e1, *e1_from_memory);
        assert_eq!(e2, *e2_from_memory);
        assert_eq!(e3, *e3_from_memory);
        assert_eq!(e4, *e4_from_memory);
        assert_eq!(e5, *e5_from_memory);
    }
}
