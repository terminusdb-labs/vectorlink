use std::{
    any::Any,
    io,
    ops::{Deref, DerefMut, Range},
    path::Path,
    sync::{Arc, RwLock},
};

use urlencoding::encode;

use crate::store::{ImmutableVectorFile, LoadedVectorRange, SequentialVectorLoader, VectorFile};

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

pub struct Domain<T> {
    name: String,
    file: RwLock<VectorFile<T>>,
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
impl<T: Copy> Domain<T> {
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

    fn add_vecs<'a, I: Iterator<Item = &'a T>>(&self, vecs: I) -> io::Result<(usize, usize)>
    where
        T: 'a,
    {
        let mut vector_file = self.file_mut();
        let old_len = vector_file.num_vecs();
        let count = vector_file.append_vectors(vecs)?;

        Ok((old_len, count))
    }

    pub fn concatenate_file<P: AsRef<Path>>(&self, path: P) -> io::Result<(usize, usize)> {
        let read_vector_file = VectorFile::open(path, true)?;
        let old_size = self.num_vecs();
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
}
