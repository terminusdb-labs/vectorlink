# Vectorlink v2
A redefinition of the contract.

## Basic concepts
## Vector
A vector is a list of f32.

### Vector Domain
A vector domain is a named collection of vectors. Each vector is part
of only one vector domain. Each vector within a domain has a unique
ID.

### Embedder
Domains are backed by a embedder. Embedders take a string, and turn
them into a vector. Different embedders produce vectors of a different
length.

Embedders require a configuration. Part of this is known ahead, part
of it (like authentication keys) might arrive later as part of actual
calls.

Embedders spawn tasks for doing the actual embedding.

Embedders might be quantized. In this case they hold on to some state
saying how to do the quantization.

Embedders know how to compare the sorts of vectors they produce.

### Index
Domains have indexes. Indexes are HNSW that can be configured in
different ways.

Building an index is a task.

### Tasks
Various operations take a long time to complete, and might be
interrupted by network outages, crashes, etc. Therefore the tracking
of these tasks has to be a top-level concern of VectorLink.

VectorLink doesn't need a full-fledged task system, but should at
least provide basic building blocks for the participation in one.

We should therefore assume the existence of tasks as a
larger-than-vectorlink concept, potentially involving external
tracking servers, etc. To this framework, we provide the following
functionalities:

- start a vectorlink task based on a set of input parameters and a
  given (assumed globally unique) name.  If the task needs unavailable
  resources, we start waiting for them.
- retrieve status of any task.
- interrupt a task.

And receive the following functionalities:
- check task interruption
  something we can call at various convenient points in a longer
  running task to optionally quit doing our task.
- signal task status
  This is either a basic liveness call or a progress update.
- signal task success
  This persists a response object for task status readers
- signal task error
  This persists an error object for task status readers

Now one could imagine a task system that persistently stores tasks and
checks their liveness. VectorLink, connected to this task system,
would keep it up to date about all long-running tasks currently
happening in this instance. If any sort of interruption happens,
another equivalent vectorlink install could be instructed to pick up
the task instead.

But we can simply start with a fully in-process system that just does
the bare minimum.

## Operations
### Domains
#### Create Domain
create a new named domain.

This will take one parameter.
- embedder: The embedder backend to use.

#### Delete Domain
Delete a domain along with all its vectors and indexes.

### Vectors
#### Create Embeddings
Turns strings into embeddings, according to the embedder backend of the given domain

This takes two parameters:
- domain: the domain to put these vectors in
- strings: array of strings to embed

And returns two results:
- startId: The new id of the first string. The remaining strings will have id startId+i (where i is the index of the string in the input array)
- taskId: A task id for this index task.

The taskId can be used with the task api to query its state.

#### Create Direct Embeddings
Turns strings into embeddings without storing them.
This takes two parameters:
- strings
- embeddding

And returns an array of embeddings in an agreed upon format.

(less useful, but good for testing  and maybe useful as a generic frontend to different embedding backends?)

### Index
#### Create index
Create a new index.
This will take 2 parameters:
- index name
- hnsw parameters

Output is immediate (no task) and indicates success or failure

If this is pq, it should also submit a list of vector ids to seed the
quanta with

#### Update index
Updates the index with a given vector id block.

This takes a list of tuples:
- startVec
- len

And will update the index with these blocks of vectors at once

Output is a taskid for the indexing task, and is returned before
indexing is actually done.

#### Delete index
Deletes the index.

## Improve index
Improve quality of the index.
This is returns a taskid.

#### Search
Given a query string or vector id, this will return the closest
matches in the index.

#### Nearest neighbors
Given a vector id, this will return the nearest neigbbors of this vector.

Optionally, give all nearest neighbors at once?

### Task
#### get task status
