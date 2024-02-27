# Tasks
In vectorlink, a task is a long-running process whose progress and
eventual result we wish to track. This crate is concerned with the
overhead of this tracking.

## Why tasks?
### Recoverability
lots of these operations aren't cheap, in terms of resources,
finances, and time. We can't afford having a task crash near the end,
only for its almost-done-result to be lost forever.

It is better to be able to restart an operation. For this, a task
needs to persist enough state that at a later time we can reconstruct
the task from this state.

This also goes for the final result. Between the moment a task is done
and the moment someone comes around to check this result, the
environment might have crashed.

### Progress tracking and logging
It is good to know a task is actually doing something. a framework
lets us keep track of this a little better.

### Interrupting
Sometimes we learn that there's no point in keeping a particular
task going. This happens for example when the original requester has
gone away. It is good to be able to stop a long-running task when that
happens.

## Persistency
For recoverability there's a persistent element to tasks.
This is a solved problem. The annoyance is that it is solved in
various ways, and the best way depends on the specific environment the
user is in. For example, if storing vectors is a concern of a bigger
database environment, probably it already has an excellent way of
persisting task state.

Persistency therefore has to be pluggable and the task system
shouldn't assume a particular backend.

In particular, we don't want task implementors to know or care about
how their task state is going to be persisted, just that it will
be. And we don't want the task persistence backend to know or care
about what a task state actually expresses, what format it is in, what
conventions it uses, etc. It is just storing some opaque data.

### Serialize / Deserialize task state
Some use of serde will suffice. The next step can just consider what
it got an opaque binary.

### Load/Store in persistent taskid-task_state map
Basically, just use some sort of database for this.

### basic API
We can tie the two components together in a basic API that given a
task_id produces the task_state in its correctly deserialized form,
and ismilar for setting the current state (or completion).

## Resuming tasks
Tasks should be manually resumed, ensuring they only run if someone is
interested in them.
Resuming could optionally be integrated with polling, where a frontend
process polls the task state, triggering a resume if the task is
resumable.

## Super simple single file case
storage backend can just be a single file that new state is serialized to.
by doing the atomic file move trick we can make sure that no crashes result in an unusable state file.

