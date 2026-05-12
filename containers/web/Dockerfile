# Build from the repository root:
#   docker build -f containers/web/Dockerfile -t catchy-web .
FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:/usr/local/bin:${PATH}" \
    CATCHY_DATA_DIR=/data \
    CATCHY_DIND=1 \
    DJANGO_DEBUG=0 \
    DJANGO_ALLOWED_HOSTS=*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    docker.io \
    docker-buildx \
    tzdata \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh -o /tmp/install-uv.sh \
    && env UV_INSTALL_DIR=/usr/local/bin sh /tmp/install-uv.sh \
    && rm /tmp/install-uv.sh

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY packages/core/pyproject.toml packages/core/pyproject.toml
COPY packages/claude-code/pyproject.toml packages/claude-code/pyproject.toml
COPY packages/codex/pyproject.toml packages/codex/pyproject.toml
COPY packages/web/pyproject.toml packages/web/pyproject.toml

RUN uv python install 3.14 \
    && uv sync --frozen --no-dev --package catchy-web --no-install-workspace

COPY packages/core packages/core
COPY packages/claude-code packages/claude-code
COPY packages/codex packages/codex
COPY packages/web packages/web
COPY containers/web/docker-entrypoint.sh /usr/local/bin/catchy-web-entrypoint

RUN uv sync --frozen --no-dev --package catchy-web \
    && mkdir -p /data /var/lib/docker /var/run \
    && chmod +x /usr/local/bin/catchy-web-entrypoint

VOLUME ["/data", "/var/lib/docker"]
EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/catchy-web-entrypoint"]
CMD ["gunicorn", "catchy.web.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3", "--worker-class", "gthread", "--threads", "8", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "--capture-output", "--log-level", "info"]
