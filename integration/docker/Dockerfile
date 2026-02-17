FROM rust:1.86-slim-bookworm AS branchfs-builder

RUN apt-get update && apt-get install -y pkg-config libfuse3-dev && rm -rf /var/lib/apt/lists/*
RUN cargo install branchfs --locked

FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends fuse3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=branchfs-builder /usr/local/cargo/bin/branchfs /usr/local/bin/

RUN pip install --no-cache-dir BranchContext

RUN mkdir /src /workspace

COPY integration/docker/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
