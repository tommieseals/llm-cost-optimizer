# LLM Cost Optimizer Makefile

.PHONY: help install dev-install test lint format clean build docker-build docker-run

# Default target
help:
	@echo "LLM Cost Optimizer - Available targets:"
	@echo ""
	@echo "  install       Install the package"
	@echo "  dev-install   Install with dev dependencies"
	@echo "  test          Run tests"
	@echo "  lint          Run linting (ruff + mypy)"
	@echo "  format        Format code (black + isort)"
	@echo "  clean         Clean build artifacts"
	@echo "  build         Build distribution packages"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run Docker container (interactive)"
	@echo ""

# Install package
install:
	pip install -e .

# Install with dev dependencies
dev-install:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run linting
lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Build distribution
build: clean
	python -m build

# Build Docker image
docker-build:
	docker build -t llm-cost-optimizer:latest .

# Run Docker container
docker-run:
	docker run -it --rm \
		-v $$(pwd)/examples:/data \
		-v $$(pwd)/output:/app/output \
		llm-cost-optimizer:latest

# Analyze example data
example:
	llm-optimize analyze examples/sample_usage.json

# Generate example report
example-report:
	llm-optimize report examples/sample_usage.json --output ./output --formats all

# Interactive mode
interactive:
	llm-optimize interactive
