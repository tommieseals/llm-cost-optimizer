"""Tests for the optimizer module."""

from collections import defaultdict
from datetime import datetime

import pytest

from src.analyzer import UsageAnalysis, UsageRecord
from src.optimizer import (
    EXPENSIVE_MODELS,
    ROUTING_TARGETS,
    RoutingOptimizer,
    RoutingRule,
    generate_optimization_report,
)


class TestRoutingOptimizer:
    """Tests for the routing optimizer."""

    @pytest.fixture
    def sample_analysis(self):
        """Create sample analysis for testing."""
        records = [
            UsageRecord(
                timestamp=datetime(2024, 1, 15, 9, 0),
                model="gpt-4-turbo",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                task_type="simple",
                cost=0.005,
            ),
            UsageRecord(
                timestamp=datetime(2024, 1, 15, 10, 0),
                model="gpt-4-turbo",
                prompt_tokens=500,
                completion_tokens=300,
                total_tokens=800,
                task_type="code",
                cost=0.025,
            ),
            UsageRecord(
                timestamp=datetime(2024, 1, 15, 11, 0),
                model="claude-3-opus",
                prompt_tokens=1000,
                completion_tokens=800,
                total_tokens=1800,
                task_type="reasoning",
                cost=0.075,
            ),
        ]

        cost_by_model = defaultdict(float)
        tokens_by_model = defaultdict(int)
        requests_by_model = defaultdict(int)
        cost_by_task = defaultdict(float)
        tokens_by_task = defaultdict(int)
        task_distribution = defaultdict(int)

        for r in records:
            cost_by_model[r.model] += r.cost
            tokens_by_model[r.model] += r.total_tokens
            requests_by_model[r.model] += 1
            cost_by_task[r.task_type] += r.cost
            tokens_by_task[r.task_type] += r.total_tokens
            task_distribution[r.task_type] += 1

        return UsageAnalysis(
            records=records,
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2024, 1, 15),
            total_requests=3,
            total_tokens=2750,
            total_cost=0.105,
            cost_by_model=dict(cost_by_model),
            tokens_by_model=dict(tokens_by_model),
            requests_by_model=dict(requests_by_model),
            cost_by_task=dict(cost_by_task),
            tokens_by_task=dict(tokens_by_task),
            daily_costs={"2024-01-15": 0.105},
            hourly_distribution={9: 1, 10: 1, 11: 1},
            avg_tokens_per_request=916.67,
            avg_cost_per_request=0.035,
            task_distribution=dict(task_distribution),
        )

    def test_optimize_generates_rules(self, sample_analysis):
        """Optimizer should generate routing rules."""
        optimizer = RoutingOptimizer()
        recommendation = optimizer.optimize(sample_analysis)

        assert len(recommendation.rules) > 0

    def test_optimize_calculates_savings(self, sample_analysis):
        """Optimizer should calculate potential savings."""
        optimizer = RoutingOptimizer()
        recommendation = optimizer.optimize(sample_analysis)

        assert recommendation.monthly_savings >= 0
        assert recommendation.optimized_cost <= recommendation.current_cost

    def test_prefer_local_models(self, sample_analysis):
        """When prefer_local=True, should recommend local models."""
        optimizer = RoutingOptimizer(prefer_local=True)
        recommendation = optimizer.optimize(sample_analysis)

        # At least some rules should target local models
        local_models = {"llama-3.1-8b", "qwen2.5-7b", "deepseek-coder-6.7b", "phi3-mini"}
        local_rules = [r for r in recommendation.rules if r.target_model in local_models]

        assert len(local_rules) > 0

    def test_decision_tree_generated(self, sample_analysis):
        """Optimizer should generate a decision tree."""
        optimizer = RoutingOptimizer()
        recommendation = optimizer.optimize(sample_analysis)

        assert recommendation.decision_tree is not None
        assert "name" in recommendation.decision_tree
        assert "children" in recommendation.decision_tree

    def test_implementation_config_generated(self, sample_analysis):
        """Optimizer should generate implementation config."""
        optimizer = RoutingOptimizer()
        recommendation = optimizer.optimize(sample_analysis)

        config = recommendation.implementation_config
        assert "version" in config
        assert "routing_rules" in config
        assert "default_model" in config


class TestRoutingTargets:
    """Tests for routing target definitions."""

    def test_all_task_types_have_targets(self):
        """All defined task types should have routing targets."""
        expected_tasks = [
            "simple",
            "code",
            "analysis",
            "creative",
            "translation",
            "summarization",
            "extraction",
            "reasoning",
            "general",
        ]

        for task in expected_tasks:
            assert task in ROUTING_TARGETS, f"Missing targets for {task}"

    def test_targets_have_required_fields(self):
        """Each target should have model, type, and cost."""
        for task, targets in ROUTING_TARGETS.items():
            assert len(targets) > 0, f"No targets for {task}"

            for target in targets:
                assert "model" in target
                assert "type" in target
                assert "cost" in target


class TestExpensiveModels:
    """Tests for expensive model definitions."""

    def test_includes_premium_models(self):
        """Should include known expensive models."""
        expected_expensive = ["gpt-4", "gpt-4-turbo", "claude-3-opus"]

        for model in expected_expensive:
            assert model in EXPENSIVE_MODELS


class TestReportGeneration:
    """Tests for report generation."""

    @pytest.fixture
    def sample_analysis(self):
        """Create sample analysis for testing."""
        records = [
            UsageRecord(
                timestamp=datetime(2024, 1, 15, 9, 0),
                model="gpt-4-turbo",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                task_type="simple",
                cost=0.005,
            ),
            UsageRecord(
                timestamp=datetime(2024, 1, 15, 10, 0),
                model="gpt-4-turbo",
                prompt_tokens=500,
                completion_tokens=300,
                total_tokens=800,
                task_type="code",
                cost=0.025,
            ),
            UsageRecord(
                timestamp=datetime(2024, 1, 15, 11, 0),
                model="claude-3-opus",
                prompt_tokens=1000,
                completion_tokens=800,
                total_tokens=1800,
                task_type="reasoning",
                cost=0.075,
            ),
        ]

        cost_by_model = defaultdict(float)
        tokens_by_model = defaultdict(int)
        requests_by_model = defaultdict(int)
        cost_by_task = defaultdict(float)
        tokens_by_task = defaultdict(int)
        task_distribution = defaultdict(int)

        for r in records:
            cost_by_model[r.model] += r.cost
            tokens_by_model[r.model] += r.total_tokens
            requests_by_model[r.model] += 1
            cost_by_task[r.task_type] += r.cost
            tokens_by_task[r.task_type] += r.total_tokens
            task_distribution[r.task_type] += 1

        return UsageAnalysis(
            records=records,
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2024, 1, 15),
            total_requests=3,
            total_tokens=2750,
            total_cost=0.105,
            cost_by_model=dict(cost_by_model),
            tokens_by_model=dict(tokens_by_model),
            requests_by_model=dict(requests_by_model),
            cost_by_task=dict(cost_by_task),
            tokens_by_task=dict(tokens_by_task),
            daily_costs={"2024-01-15": 0.105},
            hourly_distribution={9: 1, 10: 1, 11: 1},
            avg_tokens_per_request=916.67,
            avg_cost_per_request=0.035,
            task_distribution=dict(task_distribution),
        )

    @pytest.fixture
    def sample_recommendation(self, sample_analysis):
        """Create sample recommendation."""
        optimizer = RoutingOptimizer()
        return optimizer.optimize(sample_analysis)

    def test_report_contains_key_sections(self, sample_analysis, sample_recommendation):
        """Report should contain key sections."""
        report = generate_optimization_report(sample_analysis, sample_recommendation)

        assert "CURRENT COSTS" in report or "Current" in report
        assert "OPTIMIZED" in report or "Optimized" in report
        assert "SAVINGS" in report or "Savings" in report

    def test_report_shows_dollar_amounts(self, sample_analysis, sample_recommendation):
        """Report should show dollar amounts."""
        report = generate_optimization_report(sample_analysis, sample_recommendation)

        assert "$" in report


class TestRoutingRule:
    """Tests for RoutingRule dataclass."""

    def test_rule_creation(self):
        """Should create routing rule correctly."""
        rule = RoutingRule(
            name="test_rule",
            condition="Test condition",
            source_model="gpt-4",
            target_model="llama-3.1-8b",
            task_type="simple",
            complexity="low",
            priority=100,
            estimated_savings=10.0,
            confidence=0.95,
        )

        assert rule.name == "test_rule"
        assert rule.estimated_savings == 10.0
        assert rule.confidence == 0.95
