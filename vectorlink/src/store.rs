use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    marker::PhantomData,
    ops::{Deref, Range},
    os::{
        fd::AsRawFd,
        unix::fs::{FileExt, MetadataExt, OpenOptionsExt},
    },
    path::Path,
};

pub struct LoadedVectorRange<T> {
    range: Range<usize>,
    vecs: Vec<T>,
}

impl<T: Copy> LoadedVectorRange<T> {
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
                    bytes_read * std::mem::size_of::<T>(),
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

pub fn append_vector_range<W: Write + Seek, T: Copy>(mut w: W, vectors: &[T]) -> io::Result<()> {
    let vector_bytes = unsafe {
        std::slice::from_raw_parts(
            vectors.as_ptr() as *const u8,
            vectors.len() * std::mem::size_of::<T>(),
        )
    };

    w.seek(SeekFrom::End(0))?;
    w.write_all(vector_bytes)?;
    w.flush()?;

    Ok(())
}

pub fn append_vectors<W: Write + Seek, T: Copy, DT: Deref<Target = T>, I: Iterator<Item = DT>>(
    mut w: W,
    vectors: I,
) -> io::Result<()> {
    w.seek(SeekFrom::End(0))?;
    for vec in vectors {
        let vector_bytes = unsafe {
            std::slice::from_raw_parts(&*vec as *const T as *const u8, std::mem::size_of::<T>())
        };

        w.write_all(vector_bytes)?;
    }
    w.flush()?;

    Ok(())
}

pub struct VectorFile<T> {
    file: File,
    _x: PhantomData<T>,
}

impl<T: Copy> VectorFile<T> {
    pub fn new(file: File) -> Self {
        Self {
            file,
            _x: PhantomData,
        }
    }
    pub fn create<P: AsRef<Path>>(path: P, os_cached: bool) -> io::Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        if !os_cached {
            options.custom_flags(libc::O_DIRECT);
        }
        let file = options.open(&path)?;
        Ok(Self::new(file))
    }
    pub fn open<P: AsRef<Path>>(path: P, os_cached: bool) -> io::Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(false);
        if !os_cached {
            options.custom_flags(libc::O_DIRECT);
        }
        let file = options.open(&path)?;
        Ok(Self::new(file))
    }
    pub fn open_create<P: AsRef<Path>>(path: P, os_cached: bool) -> io::Result<Self> {
        if path.as_ref().exists() {
            Self::open(path, os_cached)
        } else {
            Self::create(path, os_cached)
        }
    }

    pub fn append_vector_range(&mut self, vectors: &[T]) -> io::Result<()> {
        todo!();
    }
    pub fn append_vectors<I: Iterator<Item = T>>(&mut self, vectors: I) -> io::Result<()> {
        todo!();
    }

    pub fn vector_range(&self, range: Range<usize>) -> io::Result<LoadedVectorRange<T>> {
        LoadedVectorRange::load(&self.file, range)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
