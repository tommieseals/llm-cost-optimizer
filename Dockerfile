# LLM Cost Optimizer
# Multi-stage build for minimal image size

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir build

# Copy source
COPY pyproject.toml README.md ./
COPY src/ src/

# Build wheel
RUN python -m build --wheel

# Runtime stage
FROM python:3.11-slim

LABEL maintainer="Tommie Seals"
LABEL description="LLM Cost Optimizer - Analyze and optimize LLM API costs"
LABEL version="1.0.0"

WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash optimizer

# Copy wheel from builder
COPY --from=builder /app/dist/*.whl /tmp/

# Install the package
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm /tmp/*.whl

# Switch to non-root user
USER optimizer

# Create output directory
RUN mkdir -p /app/output

# Set entrypoint
ENTRYPOINT ["llm-optimize"]

# Default command (show help)
CMD ["--help"]

# Example usage:
# docker build -t llm-cost-optimizer .
# docker run -v $(pwd)/logs:/data llm-cost-optimizer analyze /data/usage.json
# docker run -v $(pwd)/logs:/data -v $(pwd)/output:/app/output llm-cost-optimizer report /data/usage.json --output /app/output
