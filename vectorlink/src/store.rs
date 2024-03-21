use std::{
    fs::{File, OpenOptions},
    io::{self, Read},
    marker::PhantomData,
    ops::{Index, Range},
    os::{
        fd::AsRawFd,
        unix::fs::{FileExt, MetadataExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
};

#[derive(Clone)]
pub struct LoadedVectorRange<T> {
    range: Range<usize>,
    vecs: Vec<T>,
}
impl<T> Default for LoadedVectorRange<T> {
    fn default() -> Self {
        Self {
            range: 0..0,
            vecs: Vec::new(),
        }
    }
}

impl<T> Index<usize> for LoadedVectorRange<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.vec(index)
    }
}

impl<T> LoadedVectorRange<T> {
    pub fn new(vecs: Vec<T>, range: Range<usize>) -> Self {
        assert!(range.len() == vecs.len());

        Self { vecs, range }
    }
    pub fn load(file: &File, range: Range<usize>) -> io::Result<Self> {
        let size = std::mem::size_of::<T>() * range.len();
        let mut vecs: Vec<T> = Vec::with_capacity(range.len());
        {
            let buf = vecs.spare_capacity_mut();
            let bytes_buf =
                unsafe { std::slice::from_raw_parts_mut(buf.as_ptr() as *mut u8, size) };
            file.read_exact_at(bytes_buf, range.start as u64)?;
        }

        unsafe {
            vecs.set_len(range.len());
        }

        Ok(Self { range, vecs })
    }

    #[allow(unused)]
    pub fn valid_for(&self, range: Range<usize>) -> bool {
        self.range.contains(&range.start) && self.range.contains(&range.end)
    }

    pub fn vec(&self, index: usize) -> &T {
        assert!(self.range.contains(&index));

        &self.vecs[index - self.range.start]
    }

    pub fn vecs(&self) -> &[T] {
        &self.vecs
    }
    pub fn len(&self) -> usize {
        self.vecs.len()
    }

    pub fn into_vec(self) -> Vec<T> {
        self.vecs
    }
}

pub struct VectorLoader<'a, T> {
    file: &'a File,
    _x: PhantomData<T>,
}

impl<'a, T: Copy> VectorLoader<'a, T> {
    pub fn new(file: &'a File) -> Self {
        Self {
            file,
            _x: PhantomData,
        }
    }

    pub fn load_range(&self, range: Range<usize>) -> io::Result<LoadedVectorRange<T>> {
        LoadedVectorRange::load(&self.file, range)
    }

    pub fn vec(&self, index: usize) -> io::Result<T> {
        let size = std::mem::size_of::<T>();
        let mut data: Vec<u8> = Vec::with_capacity(size);
        {
            let buf = data.spare_capacity_mut();
            let bytes_buf =
                unsafe { std::slice::from_raw_parts_mut(buf.as_ptr() as *mut u8, size) };
            self.file.read_exact_at(bytes_buf, (index * size) as u64)?;
        }
        unsafe {
            data.set_len(size);

            Ok(*(data.as_ptr() as *const T))
        }
    }
}

#[allow(unused)]
pub struct SequentialVectorLoader<T> {
    file: File,
    chunk_size: usize,
    upto: usize,
    _x: PhantomData<T>,
}

pub fn n_bytes_to_n_vecs<T>(n_bytes: usize) -> usize {
    assert!(n_bytes % std::mem::size_of::<T>() == 0);
    n_bytes / std::mem::size_of::<T>()
}

#[allow(unused)]
impl<T> SequentialVectorLoader<T> {
    pub fn new(file: File, chunk_size: usize) -> io::Result<Self> {
        let n_bytes = file.metadata()?.size();
        let n_vecs = n_bytes_to_n_vecs::<T>(n_bytes as usize);
        Ok(Self::new_upto(file, chunk_size, n_vecs + 1))
    }

    pub fn new_upto(file: File, chunk_size: usize, upto: usize) -> Self {
        Self {
            file,
            chunk_size,
            upto,
            _x: PhantomData,
        }
    }

    pub fn open<P: AsRef<Path>>(path: P, chunk_size: usize) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).open(path)?;
        let fd = file.as_raw_fd();
        let ret = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_SEQUENTIAL) };
        if ret == 0 {
            Ok(Self::new(file, chunk_size)?)
        } else {
            Err(io::Error::new(io::ErrorKind::Other, "fadvice failed"))
        }
    }

    pub fn load_chunk(&mut self) -> io::Result<Option<Vec<T>>> {
        let mut data: Vec<T> = Vec::with_capacity(self.chunk_size);
        let mut bytes_read = 0;
        {
            let buf = data.spare_capacity_mut();
            let bytes_buf = unsafe {
                std::slice::from_raw_parts_mut(
                    buf.as_ptr() as *mut u8,
                    buf.len() * std::mem::size_of::<T>(),
                )
            };
            loop {
                let count = self.file.read(&mut bytes_buf[bytes_read..])?;
                bytes_read += count;
                if count == 0 || bytes_read == buf.len() {
                    // done reading!
                    break;
                }
            }
        }
        if bytes_read == 0 {
            Ok(None)
        } else {
            // make sure that we read a multiple of T
            assert!(bytes_read % std::mem::size_of::<T>() == 0);
            unsafe {
                data.set_len(bytes_read / std::mem::size_of::<T>());
            }

            Ok(Some(data))
        }
    }
}

impl<T> Iterator for SequentialVectorLoader<T> {
    type Item = io::Result<Vec<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        // The iterator is a simple transformation of load_chunk, switching the option and the result
        match self.load_chunk() {
            Ok(None) => None,
            Ok(Some(v)) => Some(Ok(v)),
            Err(e) => Some(Err(e)),
        }
    }
}

pub struct VectorFile<T> {
    path: PathBuf,
    file: File,
    num_vecs: usize,
    _x: PhantomData<T>,
}

impl<T: Copy> VectorFile<T> {
    pub fn new(path: PathBuf, file: File, num_vecs: usize) -> Self {
        Self {
            path,
            file,
            num_vecs,
            _x: PhantomData,
        }
    }

    #[allow(unused)]
    pub fn create_new<P: AsRef<Path>>(path: P, os_cached: bool) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        if !os_cached {
            options.custom_flags(libc::O_DIRECT);
        }
        let file = options.open(&path)?;
        Ok(Self::new(path, file, 0))
    }
    pub fn create<P: AsRef<Path>>(path: P, os_cached: bool) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(true);
        if !os_cached {
            options.custom_flags(libc::O_DIRECT);
        }
        let file = options.open(&path)?;
        Ok(Self::new(path, file, 0))
    }
    pub fn open<P: AsRef<Path>>(path: P, os_cached: bool) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(false);
        if !os_cached {
            options.custom_flags(libc::O_DIRECT);
        }

        let file = options.open(&path)?;
        let byte_size = file.metadata()?.size() as usize;
        let single_vec_size = std::mem::size_of::<T>();

        assert!(byte_size % single_vec_size == 0);

        let num_vecs = byte_size / single_vec_size;

        Ok(Self::new(path, file, num_vecs))
    }

    pub fn open_create<P: AsRef<Path>>(path: P, os_cached: bool) -> io::Result<Self> {
        if path.as_ref().exists() {
            Self::open(path, os_cached)
        } else {
            Self::create(path, os_cached)
        }
    }

    pub fn append_vector_range(&mut self, vectors: &[T]) -> io::Result<usize> {
        let vector_bytes = unsafe {
            std::slice::from_raw_parts(
                vectors.as_ptr() as *const T as *const u8,
                vectors.len() * std::mem::size_of::<T>(),
            )
        };
        self.file.write_all_at(
            vector_bytes,
            (self.num_vecs * std::mem::size_of::<T>()) as u64,
        )?;
        self.num_vecs = self.num_vecs + vectors.len();
        self.file.sync_data()?; // TODO probably don't do it here cause we might want to append multiple ranges

        Ok(vectors.len())
    }

    pub fn append_vector_file(&mut self, file: &VectorFile<T>) -> io::Result<usize> {
        let mut read_offset = 0;
        let mut write_offset = (self.num_vecs * std::mem::size_of::<T>()) as u64;

        let num_vecs_to_write = file.num_vecs;
        let mut num_bytes_to_write = num_vecs_to_write * std::mem::size_of::<T>();

        let mut buf = vec![0_u8; 4096];
        while num_bytes_to_write != 0 {
            let n = file.file.read_at(&mut buf, read_offset)?;
            self.file.write_all_at(&buf[..n], write_offset)?;
            num_bytes_to_write -= n;
            read_offset += n as u64;
            write_offset += n as u64;
        }
        self.file.sync_data()?;

        Ok(num_vecs_to_write)
    }

    pub fn vector_loader(&self) -> VectorLoader<T> {
        VectorLoader::new(&self.file)
    }

    pub fn vector_range(&self, range: Range<usize>) -> io::Result<LoadedVectorRange<T>> {
        self.vector_loader().load_range(range)
    }

    pub fn vec(&self, index: usize) -> io::Result<T> {
        self.vector_loader().vec(index)
    }

    pub fn all_vectors(&self) -> io::Result<LoadedVectorRange<T>> {
        self.vector_loader().load_range(0..self.num_vecs)
    }

    pub fn num_vecs(&self) -> usize {
        self.num_vecs
    }

    pub fn vector_chunks(&self, chunk_size: usize) -> io::Result<SequentialVectorLoader<T>> {
        SequentialVectorLoader::open(&self.path, chunk_size)
    }

    pub fn as_immutable(&self) -> ImmutableVectorFile<T> {
        ImmutableVectorFile(Self {
            path: self.path.clone(),
            file: self
                .file
                .try_clone()
                .expect("could not clone file handle while creating immutable vector filehandle"),
            num_vecs: self.num_vecs,
            _x: PhantomData,
        })
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

pub struct ImmutableVectorFile<T>(VectorFile<T>);
impl<T> Clone for ImmutableVectorFile<T> {
    fn clone(&self) -> Self {
        Self(VectorFile {
            path: self.0.path.clone(),
            file: self
                .0
                .file
                .try_clone()
                .expect("could not clone file handle while creating immutable vector filehandle"),
            num_vecs: self.0.num_vecs,
            _x: PhantomData,
        })
    }
}

#[allow(unused)]
impl<T: Copy> ImmutableVectorFile<T> {
    pub fn vector_loader(&self) -> VectorLoader<T> {
        self.0.vector_loader()
    }

    pub fn vector_range(&self, range: Range<usize>) -> io::Result<LoadedVectorRange<T>> {
        self.0.vector_range(range)
    }

    pub fn vec(&self, index: usize) -> io::Result<T> {
        self.0.vec(index)
    }

    pub fn all_vectors(&self) -> io::Result<LoadedVectorRange<T>> {
        self.0.all_vectors()
    }

    pub fn num_vecs(&self) -> usize {
        self.0.num_vecs()
    }

    pub fn vector_chunks(&self, chunk_size: usize) -> io::Result<SequentialVectorLoader<T>> {
        self.0.vector_chunks(chunk_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
