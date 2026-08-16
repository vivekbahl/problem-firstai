FROM python:3.13-slim as base

# 1. Update OS packages securely
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Use your lightweight pip installation for uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# 3. Copy only dependency files first to maximize Docker layer caching
COPY pyproject.toml uv.lock ./

# 4. Install dependencies without treating your source code folder as a package
RUN uv sync --frozen --no-install-project

# 5. Copy your application code
COPY code/ ./code/

EXPOSE 8000

# 6. Secure, non-root user execution
USER nobody

ENTRYPOINT ["/app/.venv/bin/python", "code/run.py"]
CMD ["--mode", "part1"]
