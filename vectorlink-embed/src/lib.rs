pub mod openai;

pub trait Embedder {
    type Embedding;
    fn embeddings_for<S: AsRef<str>>(&self, strings: &[S]) -> Vec<Self::Embedding>;
}
// tasking? embedding a block of vecs is a job that could take several startups, so we need the usual progress tracking.
// meaning there's an embedding strings file, and a progress file, and a result file

pub trait TaskBackend {
    type TaskBuilder;
    type Task;
    fn create_task(&self) -> Self::TaskBuilder;
    fn save_task(&self, task: Self::Task);
}
