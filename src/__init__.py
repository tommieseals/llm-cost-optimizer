"""
LLM Cost Optimizer
==================
Analyze LLM API usage patterns and optimize routing for cost savings.
"""

from .analyzer import UsageAnalyzer, UsageAnalysis, UsageRecord
from .optimizer import RoutingOptimizer, RoutingRecommendation, RoutingRule
from .visualizer import (
    ascii_bar_chart, ascii_pie_chart, ascii_decision_tree,
    generate_html_report, generate_markdown_report
)
from .cli import main

__version__ = "1.0.0"
__author__ = "Tommie Seals"
__all__ = [
    "UsageAnalyzer",
    "UsageAnalysis", 
    "UsageRecord",
    "RoutingOptimizer",
    "RoutingRecommendation",
    "RoutingRule",
    "main",
]
