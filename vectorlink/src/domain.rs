use std::{
    any::Any,
    collections::HashMap,
    io,
    ops::{Deref, DerefMut, Range},
    path::Path,
    sync::{Arc, RwLock},
};

use parallel_hnsw::{
    pq::{HnswQuantizer, Quantizer},
    Comparator,
};
use urlencoding::encode;

use crate::{
    store::{ImmutableVectorFile, LoadedVectorRange, SequentialVectorLoader, VectorFile},
    vecmath::Embedding,
};

pub trait GenericDomain: 'static + Any + Send + Sync {
    fn name(&self) -> &str;
    fn num_vecs(&self) -> usize;
}

pub fn downcast_generic_domain<T: 'static + Send + Sync>(
    domain: Arc<dyn GenericDomain>,
) -> Arc<Domain<T>> {
    Arc::downcast::<Domain<T>>(domain)
        .expect("Could not downcast domain to expected embedding size")
}

pub trait Deriver: Any {
    type From: Copy;

    fn concatenate_derived(&self, loader: SequentialVectorLoader<Self::From>) -> io::Result<()>;
    fn chunk_size(&self) -> usize {
        1_000
    }
}

pub struct PqDerivedDomain<
    const SIZE: usize,
    const CENTROID_SIZE: usize,
    const QUANTIZED_SIZE: usize,
    C,
> {
    name: String,
    file: RwLock<VectorFile<[u16; QUANTIZED_SIZE]>>,
    quantizer: HnswQuantizer<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, C>,
}

impl<
        const SIZE: usize,
        const CENTROID_SIZE: usize,
        const QUANTIZED_SIZE: usize,
        C: 'static + Comparator<T = [f32; CENTROID_SIZE]>,
    > Deriver for PqDerivedDomain<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, C>
{
    type From = [f32; SIZE];

    fn concatenate_derived(&self, loader: SequentialVectorLoader<Self::From>) -> io::Result<()> {
        for chunk in loader {
            let chunk = chunk?;
            let mut result = Vec::with_capacity(chunk.len());
            for vec in chunk.iter() {
                let quantized = self.quantizer.quantize(vec);
                result.push(quantized);
            }
            let mut file = self.file.write().unwrap();
            file.append_vector_range(&result)?;
        }

        Ok(())
    }
}

pub struct Domain<T> {
    name: String,
    file: RwLock<VectorFile<T>>,
    derived_domains: RwLock<HashMap<String, Arc<dyn Deriver<From = T> + Send + Sync>>>,
}

impl<T: 'static + Copy + Send + Sync> GenericDomain for Domain<T> {
    fn name(&self) -> &str {
        &self.name
    }

    fn num_vecs(&self) -> usize {
        self.file().num_vecs()
    }
}

#[allow(unused)]
impl<T: Copy + 'static> Domain<T> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn num_vecs(&self) -> usize {
        self.file().num_vecs()
    }

    pub fn open<P: AsRef<Path>>(dir: P, name: &str) -> io::Result<Self> {
        let mut path = dir.as_ref().to_path_buf();
        let encoded_name = encode(name);
        path.push(format!("{encoded_name}.vecs"));
        let file = RwLock::new(VectorFile::open_create(&path, true)?);

        Ok(Domain {
            name: name.to_string(),
            derived_domains: RwLock::new(HashMap::new()),
            file,
        })
    }

    pub fn file<'a>(&'a self) -> impl Deref<Target = VectorFile<T>> + 'a {
        self.file.read().unwrap()
    }

    fn file_mut<'a>(&'a self) -> impl DerefMut<Target = VectorFile<T>> + 'a {
        self.file.write().unwrap()
    }

    pub fn immutable_file(&self) -> ImmutableVectorFile<T> {
        self.file().as_immutable()
    }

    pub fn concatenate_file<P: AsRef<Path>>(&self, path: P) -> io::Result<(usize, usize)> {
        let read_vector_file = VectorFile::open(path, true)?;
        let old_size = self.num_vecs();
        let derived_domains = self.derived_domains.read().unwrap();
        for derived in derived_domains.values() {
            let chunk_size = derived.chunk_size();
            derived.concatenate_derived(read_vector_file.vector_chunks(chunk_size)?)?;
        }
        Ok((
            old_size,
            self.file_mut().append_vector_file(&read_vector_file)?,
        ))
    }

    pub fn vec(&self, id: usize) -> io::Result<T> {
        Ok(self.file().vec(id)?)
    }

    pub fn vec_range(&self, range: Range<usize>) -> io::Result<LoadedVectorRange<T>> {
        self.file().vector_range(range)
    }

    pub fn all_vecs(&self) -> io::Result<LoadedVectorRange<T>> {
        self.file().all_vectors()
    }

    pub fn vector_chunks(&self, chunk_size: usize) -> io::Result<SequentialVectorLoader<T>> {
        self.file().vector_chunks(chunk_size)
    }

    pub fn create_derived<D: Deriver<From = T> + 'static + Send + Sync>(
        &self,
        name: String,
        deriver: D,
    ) {
        let mut derived_domains = self.derived_domains.write().unwrap();
        assert!(
            !derived_domains.contains_key(&name),
            "tried to create derived domain that already exists"
        );

        derived_domains.insert(name, Arc::new(deriver));
    }

    pub fn derived_domain<'a, T2: Deriver + Send + Sync>(
        &'a self,
        name: &str,
    ) -> Option<impl Deref<Target = T2> + 'a> {
        let derived_domains = self.derived_domains.read().unwrap();
        let derived = derived_domains.get(name)?;

        Some(Arc::downcast::<T2>(derived.clone()).expect("derived domain was not of expected type"))
    }
}
