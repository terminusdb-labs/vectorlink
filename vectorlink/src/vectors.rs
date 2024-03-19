#![allow(unused)]

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::mem::MaybeUninit;
use std::ops::{Deref, Range};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
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
use crate::store::{LoadedVectorRange, SequentialVectorLoader, VectorLoader};
use crate::vecmath::{
    Centroid32, Embedding, EmbeddingBytes, CENTROID_32_LENGTH, EMBEDDING_BYTE_LENGTH,
    EMBEDDING_LENGTH, QUANTIZED_32_EMBEDDING_LENGTH,
};
use parallel_hnsw::pq::HnswQuantizer;

pub struct Domain {
    name: Arc<String>,
    index: usize,
    path: PathBuf,
    file: File,
    num_vecs: AtomicUsize,
    write_lock: Mutex<()>,
}

impl Domain {
    pub fn name(&self) -> &str {
        &self.name
    }

    fn open(dir: &Path, name: &str, index: usize) -> io::Result<Self> {
        let mut path = dir.to_path_buf();
        let encoded_name = encode(name);
        path.push(format!("{encoded_name}.vecs"));
        let end = std::fs::metadata(&path)?.size();
        let num_vecs = AtomicUsize::new(end as usize / EMBEDDING_BYTE_LENGTH);
        // todo: shouldn't we be creating this file if it is not there?
        let file = OpenOptions::new()
            .custom_flags(libc::O_DIRECT)
            .read(true)
            .open(&path)?;

        Ok(Domain {
            name: Arc::new(name.to_string()),
            file,
            index,
            path,
            num_vecs,
            write_lock: Mutex::new(()),
        })
    }

    fn add_vecs<'a, I: Iterator<Item = &'a Embedding>>(
        &self,
        vecs: I,
    ) -> io::Result<(usize, usize)> {
        let lock = self.write_lock.lock().unwrap();
        let mut write_file = OpenOptions::new()
            .create(false)
            .write(true)
            .open(&self.path)?;
        write_file.seek(SeekFrom::End(0))?;
        let mut count = 0;
        for embedding in vecs {
            let bytes: &EmbeddingBytes = unsafe { std::mem::transmute(embedding) };
            write_file.write_all(bytes)?;
            count += 1;
        }
        write_file.flush()?;
        write_file.sync_data()?;
        let num_vecs = self.num_vecs.load(atomic::Ordering::Relaxed);
        let new_num_vecs = num_vecs + count;
        self.num_vecs.store(new_num_vecs, atomic::Ordering::Relaxed);

        Ok((num_vecs, count))
    }

    pub fn concatenate_file<P: AsRef<Path>>(&self, path: P) -> io::Result<usize> {
        let lock = self.write_lock.lock().unwrap();
        let size = self.num_vecs();
        let mut write_file = OpenOptions::new()
            .create(false)
            .write(true)
            .open(&self.path)?;
        write_file.seek(SeekFrom::End(0))?;
        let mut read_file = File::options().read(true).open(&path)?;
        io::copy(&mut read_file, &mut write_file)?;
        Ok(size)
    }

    pub fn num_vecs(&self) -> usize {
        self.num_vecs.load(atomic::Ordering::Relaxed)
    }

    pub fn loader<'a>(&'a self) -> VectorLoader<'a, Embedding> {
        VectorLoader::new(&self.file)
    }

    pub fn vec(&self, id: usize) -> io::Result<Embedding> {
        self.loader().vec(id)
    }

    pub fn vec_range(&self, range: Range<usize>) -> io::Result<LoadedVectorRange<Embedding>> {
        self.loader().load_range(range)
    }

    pub fn all_vecs(&self) -> io::Result<LoadedVectorRange<Embedding>> {
        self.loader().load_range(0..self.num_vecs())
    }

    pub fn vector_chunks(
        &self,
        chunk_size: usize,
    ) -> io::Result<SequentialVectorLoader<Embedding>> {
        SequentialVectorLoader::open(&self.path, chunk_size)
    }
}

#[derive(Clone)]
pub struct LoadedVec {
    range: Arc<LoadedVectorRange<Embedding>>,
    vec: usize,
}

impl LoadedVec {
    pub fn id(&self) -> usize {
        self.vec
    }
}

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

        unsafe { self.range.vec(self.vec) }
    }
}

impl PartialEq for LoadedVec {
    fn eq(&self, other: &Self) -> bool {
        Arc::as_ptr(&self.range) == Arc::as_ptr(&other.range) && self.vec == other.vec
    }
}

pub struct VectorStore {
    dir: PathBuf,
    domains: RwLock<HashMap<String, Arc<Domain>>>,
}

impl VectorStore {
    pub fn new<P: Into<PathBuf>>(path: P, num_bufs: usize) -> Self {
        Self {
            dir: path.into(),
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
