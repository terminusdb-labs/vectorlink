FROM rust:bullseye as builder
WORKDIR /usr/src/app
COPY . .
RUN cargo build --release

FROM debian:bullseye-slim
WORKDIR /app
COPY --from=builder /usr/src/app/target/release/terminusdb-semantic-indexer /app/
CMD ["./terminusdb-semantic-indexer", "serve", "--directory", "/app/terminusdb/storage/semantic_db"]
