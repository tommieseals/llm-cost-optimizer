"""Tests for the analyzer module."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.analyzer import (
    MODEL_PRICING,
    UsageAnalyzer,
    UsageRecord,
    calculate_cost,
    classify_task,
    normalize_model_name,
)


class TestTaskClassification:
    """Tests for task classification."""

    def test_classify_simple_question(self):
        """Simple questions should be classified as 'simple'."""
        prompts = [
            "What is the capital of France?",
            "Who was the first president?",
            "When did WW2 end?",
        ]
        for prompt in prompts:
            assert classify_task(prompt) == "simple"

    def test_classify_code_task(self):
        """Code-related prompts should be classified as 'code'."""
        prompts = [
            "Write a Python function to calculate fibonacci",
            "Debug this JavaScript code",
            "Implement a binary search tree in TypeScript",
        ]
        for prompt in prompts:
            assert classify_task(prompt) == "code"

    def test_classify_analysis_task(self):
        """Analysis prompts should be classified as 'analysis'."""
        prompts = [
            "Analyze the following data and provide insights",
            "Compare these two approaches",
            "Evaluate the performance metrics",
        ]
        for prompt in prompts:
            assert classify_task(prompt) == "analysis"

    def test_classify_creative_task(self):
        """Creative prompts should be classified as 'creative'."""
        prompts = [
            "Write a short story about a robot",
            "Create a poem about nature",
            "Generate a creative narrative",
        ]
        for prompt in prompts:
            assert classify_task(prompt) == "creative"

    def test_classify_reasoning_task(self):
        """Complex reasoning should be classified as 'reasoning'."""
        prompts = [
            "Think step by step through this problem",
            "Use chain of thought reasoning",
            "This is a complex mathematical proof",
        ]
        for prompt in prompts:
            assert classify_task(prompt) == "reasoning"


class TestModelNormalization:
    """Tests for model name normalization."""

    def test_normalize_gpt4_variants(self):
        """GPT-4 variants should normalize correctly."""
        assert normalize_model_name("gpt-4-0125-preview") == "gpt-4-turbo"
        assert normalize_model_name("gpt-4-1106-preview") == "gpt-4-turbo"
        assert normalize_model_name("GPT-4-TURBO") == "gpt-4-turbo"

    def test_normalize_claude_variants(self):
        """Claude variants should normalize correctly."""
        assert normalize_model_name("claude-3-opus-20240229") == "claude-3-opus"
        assert normalize_model_name("claude-3-5-sonnet-20240620") == "claude-3.5-sonnet"

    def test_normalize_unknown_model(self):
        """Unknown models should pass through."""
        assert normalize_model_name("custom-model-v1") == "custom-model-v1"


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_gpt4_cost(self):
        """GPT-4 costs should calculate correctly."""
        # 1000 prompt tokens + 1000 completion tokens
        cost = calculate_cost("gpt-4", 1000, 1000)
        expected = (1000 / 1_000_000 * 30) + (1000 / 1_000_000 * 60)
        assert abs(cost - expected) < 0.0001

    def test_local_model_zero_cost(self):
        """Local models should have zero cost."""
        cost = calculate_cost("llama-3.1-8b", 10000, 10000)
        assert cost == 0.0

    def test_unknown_model_default_cost(self):
        """Unknown models should use default pricing."""
        cost = calculate_cost("unknown-model", 1000, 1000)
        assert cost > 0  # Should have some cost


class TestUsageAnalyzer:
    """Tests for the usage analyzer."""

    @pytest.fixture
    def sample_log_file(self):
        """Create a temporary sample log file."""
        data = {
            "usage": [
                {
                    "timestamp": "2024-01-15T09:00:00Z",
                    "model": "gpt-4-turbo",
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "prompt": "What is the capital of France?",
                },
                {
                    "timestamp": "2024-01-15T10:00:00Z",
                    "model": "gpt-4-turbo",
                    "prompt_tokens": 500,
                    "completion_tokens": 300,
                    "total_tokens": 800,
                    "prompt": "Write a Python function to sort a list",
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            return Path(f.name)

    def test_load_json(self, sample_log_file):
        """Should load JSON logs correctly."""
        analyzer = UsageAnalyzer()
        count = analyzer.load(sample_log_file)

        assert count == 2
        assert len(analyzer.records) == 2

    def test_analyze(self, sample_log_file):
        """Should analyze records correctly."""
        analyzer = UsageAnalyzer()
        analyzer.load(sample_log_file)
        analysis = analyzer.analyze()

        assert analysis.total_requests == 2
        assert analysis.total_tokens == 950
        assert analysis.total_cost > 0
        assert "gpt-4-turbo" in analysis.cost_by_model

    def test_task_distribution(self, sample_log_file):
        """Should calculate task distribution."""
        analyzer = UsageAnalyzer()
        analyzer.load(sample_log_file)
        analysis = analyzer.analyze()

        assert "simple" in analysis.task_distribution or "code" in analysis.task_distribution


class TestUsageRecord:
    """Tests for UsageRecord dataclass."""

    def test_tokens_per_dollar(self):
        """Should calculate efficiency metric correctly."""
        record = UsageRecord(
            timestamp=datetime.now(),
            model="gpt-4",
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            cost=0.05,
        )

        assert record.tokens_per_dollar == 30000  # 1500 / 0.05

    def test_tokens_per_dollar_zero_cost(self):
        """Should handle zero cost (local models)."""
        record = UsageRecord(
            timestamp=datetime.now(),
            model="llama-3.1-8b",
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            cost=0.0,
        )

        assert record.tokens_per_dollar == float("inf")


class TestModelPricing:
    """Tests for model pricing database."""

    def test_has_major_models(self):
        """Should have pricing for major models."""
        major_models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3.5-sonnet",
            "llama-3.1-8b",
            "llama-3.1-70b",
        ]

        for model in major_models:
            assert model in MODEL_PRICING, f"Missing pricing for {model}"

    def test_local_models_are_free(self):
        """Local models should have zero cost."""
        local_models = [
            "llama-3.1-8b",
            "llama-3.1-70b",
            "qwen2.5-3b",
            "qwen2.5-7b",
            "deepseek-coder-6.7b",
            "mistral-7b",
        ]

        for model in local_models:
            pricing = MODEL_PRICING.get(model, {})
            assert pricing.get("input", 1) == 0, f"{model} should be free"
            assert pricing.get("output", 1) == 0, f"{model} should be free"
