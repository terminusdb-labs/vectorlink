# Vemdex: TerminusDB Semantic Indexer

The TerminusDB Semantic Indexer is a vector database with an index
based on Hierarchical Navigable Small World graphs written in rust. It
is designed to work closely with TerminusDB but can be used with any
project via a simple HTTP api. In order to work well with TerminusDB
it is designed with the following features:

* Domains: The database can manage several domains. In a domain you
  have a vector store which is append only. This allows you to share
  vectors across indexes.
* Commits: Each index exists at a commit. The index can point to any
  vector in a domain. This allows us to add and remove vectors by
  changing only the index.
* Incremental Indexing: The indexer can take a previous commit, and
  then perform the operations specified to obtain a new commit.
* Connects with a text-to-vector embedding API in order to convert
  content into vectors.

To invoke the server, you can run it as follows:

## Compiling

You can comile the system with cargo:

```shell
cargo compile --release
```

## Invoking

In order to invoke the server, you need to supply an OpenAI key. This
will provide you with embeddings for your text.

You can do this by either setting the env variable `OPENAI_KEY` or by
using the `--key` command line option.

```shell
terminusdb-semantic-indexer serve --directory /path/to/storage/dir
```

## Indexing

If you wan to index documents, you can any of these methods:

* Run a TerminusDB installation and refer to real commits and databases
* Put up an endpoint that will issue the appropriate operations for a
commit id and a domain
* use the `load` command

In any case, the database expects a content which will have the form
(in JSONlines format):

```json
{"id":"terminusdb:///star-wars/People/20", "op":"Inserted", "string":"The person's name is Yoda. They are described with the following synopsis: Yoda is a fictional character in the Star Wars franchise created by George Lucas, first appearing in the 1980 film The Empire Strikes Back. In the original films, he trains Luke Skywalker to fight against the Galactic Empire. In the prequel films, he serves as the Grand Master of the Jedi Order and as a high-ranking general of Clone Troopers in the Clone Wars. Following his death in Return of the Jedi at the age of 900, Yoda was the oldest living character in the Star Wars franchise in canon, until the introduction of Maz Kanata in Star Wars: The Force Awakens. Their gender is male. They have the following hair colours: white. They have a mass of 17. Their skin colours are green."}
{"id":"terminusdb:///star-wars/People/21", "op":"Deleted"}
{"id":"terminusdb:///star-wars/People/22", "op":"Replaced", "string":"The person's name is Boba Fett. They are described with the following synopsis: Boba Fett is a fictional character in the Star Wars franchise. In The Empire Strikes Back and Return of the Jedi, he is a bounty hunter hired by Darth Vader and also employed by Jabba the Hutt. He was also added briefly to the original film Star Wars when the film was digitally remastered. Star Wars: Episode II – Attack of the Clones establishes his origin as an unaltered clone of the bounty hunter Jango Fett raised as his son. He also appears in several episodes of Star Wars: The Clone Wars cartoon series which further describes his growth as a villain in the Star Wars universe. His aura of danger and mystery has created a cult following for the character. Their gender is male. They have the following hair colours: black. They have a mass of 78.2. Their skin colours are fair."}
```

To kick off indexing you can submit the following request to the Vemdex server

```shell
curl 'localhost:8080/index?commit=0vj85ifuvfcn4vwqf7w4mo2kfa3ekkn&domain=admin/star_wars'
```

This invokes the indexer for commit `0vj85ifuvfcn4vwqf7w4mo2kfa3ekkn`
and domain `admin/star_wars`.

## Searching

Searching is easy, you can specify a natural language query to the server as follows:

```shell
curl 'localhost:8080/search?commit=0vj85ifuvfcn4vwqf7w4mo2kfa3ekkn&domain=admin/star_wars'  -d "Wise old man"
```

You can also find nearby documents with:

```shell
curl 'localhost:8080/similar?commit=0vj85ifuvfcn4vwqf7w4mo2kfa3ekkn&domain=admin/star_wars?id=MyExternalID'
```

The `MyExternalID` refers to the name you gave the record during
indexing (specified by the `id` field).

## Todo

Lots of work to make this the open source versioned vector database
that the world deserves. Anyone who wants to work on the project to
advance these aims is welcome:

* Add other AI configurations for obtaining the embeddings - we'd like
  to be very complete and have ways of configuring other vendors and
  open source text-to-embedding systems.
* Greater scope of metric support
* Improve compression: We'd like to have a sytem of vector compression
  such as PQ for dealing with very large datasets.
* Better treatment of deletion and replace
* Better incrementality of the index structure
* Smaller graph reprsentations of the indicies - using succinct data
  structures to reduce memory overhead.

And if you have new ideas we'd love to hear them!


