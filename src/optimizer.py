#!/usr/bin/env python3
"""
LLM Cost Optimizer - Routing Optimizer
========================================
Generates optimal routing recommendations based on usage analysis.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime

from .analyzer import UsageAnalysis, UsageRecord, MODEL_PRICING


@dataclass
class RoutingRule:
    """Single routing rule"""
    name: str
    condition: str  # Human-readable condition
    source_model: str  # Current model
    target_model: str  # Recommended model
    task_type: str
    complexity: str
    priority: int = 0
    estimated_savings: float = 0.0
    confidence: float = 0.0


@dataclass
class RoutingRecommendation:
    """Complete routing recommendation"""
    rules: List[RoutingRule]
    decision_tree: Dict[str, Any]
    current_cost: float
    optimized_cost: float
    monthly_savings: float
    annual_savings: float
    savings_percentage: float
    implementation_config: Dict[str, Any]


# =============================================================================
# MODEL RECOMMENDATIONS
# =============================================================================

# Target models for different scenarios
ROUTING_TARGETS = {
    # Simple tasks → cheapest capable model
    "simple": [
        {"model": "llama-3.1-8b", "type": "local", "cost": 0.0},
        {"model": "gpt-3.5-turbo", "type": "cloud", "cost": 0.002},
        {"model": "gemini-1.5-flash", "type": "cloud", "cost": 0.000375},
    ],
    
    # Code tasks → specialized code models
    "code": [
        {"model": "deepseek-coder-6.7b", "type": "local", "cost": 0.0},
        {"model": "qwen2.5-14b", "type": "local", "cost": 0.0},
        {"model": "gpt-4o-mini", "type": "cloud", "cost": 0.00075},
    ],
    
    # Analysis tasks → mid-tier models
    "analysis": [
        {"model": "qwen2.5-14b", "type": "local", "cost": 0.0},
        {"model": "llama-3.1-70b", "type": "local", "cost": 0.0},
        {"model": "gpt-4o-mini", "type": "cloud", "cost": 0.00075},
    ],
    
    # Creative tasks → capable but cheaper
    "creative": [
        {"model": "mistral-7b", "type": "local", "cost": 0.0},
        {"model": "llama-3.1-70b", "type": "local", "cost": 0.0},
        {"model": "claude-3-haiku", "type": "cloud", "cost": 0.0015},
    ],
    
    # Translation → efficient models
    "translation": [
        {"model": "qwen2.5-7b", "type": "local", "cost": 0.0},
        {"model": "gpt-3.5-turbo", "type": "cloud", "cost": 0.002},
    ],
    
    # Summarization → mid-tier
    "summarization": [
        {"model": "llama-3.1-8b", "type": "local", "cost": 0.0},
        {"model": "gpt-3.5-turbo", "type": "cloud", "cost": 0.002},
    ],
    
    # Extraction → fast models
    "extraction": [
        {"model": "phi3-mini", "type": "local", "cost": 0.0},
        {"model": "gpt-3.5-turbo", "type": "cloud", "cost": 0.002},
    ],
    
    # Complex reasoning → keep expensive but optimize
    "reasoning": [
        {"model": "llama-3.1-70b", "type": "local", "cost": 0.0},
        {"model": "gpt-4o", "type": "cloud", "cost": 0.02},
    ],
    
    # General/unknown → mid-tier
    "general": [
        {"model": "qwen2.5-7b", "type": "local", "cost": 0.0},
        {"model": "gpt-3.5-turbo", "type": "cloud", "cost": 0.002},
    ],
}

# Expensive models to optimize away from
EXPENSIVE_MODELS = {
    "gpt-4": 0.045,
    "gpt-4-turbo": 0.020,
    "claude-3-opus": 0.045,
    "claude-3.5-sonnet": 0.009,
    "claude-3-sonnet": 0.009,
    "gpt-4o": 0.010,
}

# Quality requirements by task
QUALITY_REQUIREMENTS = {
    "simple": "low",
    "extraction": "low",
    "translation": "medium",
    "summarization": "medium",
    "code": "medium",
    "creative": "medium",
    "analysis": "high",
    "reasoning": "high",
    "complex": "very_high",
}


# =============================================================================
# OPTIMIZER
# =============================================================================

class RoutingOptimizer:
    """Generates optimal routing recommendations"""
    
    def __init__(self, prefer_local: bool = True, max_latency_ms: int = 5000):
        self.prefer_local = prefer_local
        self.max_latency_ms = max_latency_ms
        self.rules: List[RoutingRule] = []
    
    def optimize(self, analysis: UsageAnalysis) -> RoutingRecommendation:
        """Generate optimization recommendations"""
        
        # Group records by task type and model
        task_model_groups = defaultdict(list)
        for record in analysis.records:
            key = (record.task_type, record.model)
            task_model_groups[key].append(record)
        
        # Generate rules for each group
        rules = []
        optimized_records = []
        
        for (task_type, model), records in task_model_groups.items():
            rule = self._generate_rule(task_type, model, records)
            if rule:
                rules.append(rule)
            
            # Calculate optimized cost
            target = self._get_target_model(task_type, model)
            for record in records:
                optimized_record = self._optimize_record(record, target)
                optimized_records.append(optimized_record)
        
        # Calculate totals
        current_cost = analysis.total_cost
        optimized_cost = sum(r["cost"] for r in optimized_records)
        monthly_savings = current_cost - optimized_cost
        
        # Build decision tree
        decision_tree = self._build_decision_tree(rules)
        
        # Generate implementation config
        config = self._generate_config(rules)
        
        return RoutingRecommendation(
            rules=sorted(rules, key=lambda r: -r.estimated_savings),
            decision_tree=decision_tree,
            current_cost=current_cost,
            optimized_cost=optimized_cost,
            monthly_savings=monthly_savings,
            annual_savings=monthly_savings * 12,
            savings_percentage=(monthly_savings / current_cost * 100) if current_cost > 0 else 0,
            implementation_config=config
        )
    
    def _generate_rule(self, task_type: str, current_model: str, 
                       records: List[UsageRecord]) -> Optional[RoutingRule]:
        """Generate a routing rule for a task/model combination"""
        
        # Check if current model is already optimal
        if current_model not in EXPENSIVE_MODELS:
            return None
        
        target = self._get_target_model(task_type, current_model)
        if not target or target["model"] == current_model:
            return None
        
        # Calculate potential savings
        current_cost = sum(r.cost for r in records)
        optimized_cost = sum(
            self._calculate_cost(target["model"], r.prompt_tokens, r.completion_tokens)
            for r in records
        )
        savings = current_cost - optimized_cost
        
        if savings <= 0:
            return None
        
        quality_req = QUALITY_REQUIREMENTS.get(task_type, "medium")
        
        return RoutingRule(
            name=f"route_{task_type}_{current_model.replace('-', '_')}",
            condition=f"Task is '{task_type}' AND current model is '{current_model}'",
            source_model=current_model,
            target_model=target["model"],
            task_type=task_type,
            complexity=quality_req,
            priority=self._calculate_priority(task_type, quality_req),
            estimated_savings=savings,
            confidence=self._calculate_confidence(task_type, target["model"])
        )
    
    def _get_target_model(self, task_type: str, current_model: str) -> Optional[Dict]:
        """Get recommended target model for task type"""
        targets = ROUTING_TARGETS.get(task_type, ROUTING_TARGETS["general"])
        
        # Prefer local if configured
        if self.prefer_local:
            for target in targets:
                if target["type"] == "local":
                    return target
        
        # Otherwise return cheapest
        return targets[0] if targets else None
    
    def _optimize_record(self, record: UsageRecord, target: Optional[Dict]) -> Dict:
        """Calculate optimized cost for a record"""
        if not target:
            return {"cost": record.cost, "model": record.model}
        
        new_cost = self._calculate_cost(
            target["model"], 
            record.prompt_tokens, 
            record.completion_tokens
        )
        
        return {
            "cost": new_cost,
            "model": target["model"],
            "original_cost": record.cost,
            "savings": record.cost - new_cost
        }
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for a model"""
        pricing = MODEL_PRICING.get(model, {"input": 0.001, "output": 0.002})
        
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def _calculate_priority(self, task_type: str, quality_req: str) -> int:
        """Calculate rule priority (higher = check first)"""
        priority_map = {
            "simple": 100,  # Check simple first
            "extraction": 90,
            "translation": 80,
            "summarization": 70,
            "code": 60,
            "creative": 50,
            "analysis": 40,
            "general": 30,
            "reasoning": 20,
            "complex": 10,  # Check complex last (may need expensive model)
        }
        return priority_map.get(task_type, 50)
    
    def _calculate_confidence(self, task_type: str, target_model: str) -> float:
        """Calculate confidence in routing recommendation"""
        # Base confidence by task type
        base_confidence = {
            "simple": 0.95,
            "extraction": 0.90,
            "translation": 0.85,
            "summarization": 0.85,
            "code": 0.80,
            "creative": 0.75,
            "analysis": 0.70,
            "general": 0.65,
            "reasoning": 0.60,
            "complex": 0.50,
        }
        
        confidence = base_confidence.get(task_type, 0.7)
        
        # Boost confidence for local models (no API failures)
        if MODEL_PRICING.get(target_model, {}).get("input", 1) == 0:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _build_decision_tree(self, rules: List[RoutingRule]) -> Dict[str, Any]:
        """Build a decision tree structure for visualization"""
        tree = {
            "name": "Request",
            "children": []
        }
        
        # Group rules by task type
        by_task = defaultdict(list)
        for rule in rules:
            by_task[rule.task_type].append(rule)
        
        for task_type, task_rules in by_task.items():
            task_node = {
                "name": f"Task: {task_type}",
                "children": []
            }
            
            for rule in task_rules:
                route_node = {
                    "name": f"Route to {rule.target_model}",
                    "savings": f"${rule.estimated_savings:.2f}",
                    "confidence": f"{rule.confidence*100:.0f}%"
                }
                task_node["children"].append(route_node)
            
            tree["children"].append(task_node)
        
        # Add default route
        tree["children"].append({
            "name": "Default",
            "children": [{
                "name": "Keep original model",
                "reason": "Complex task or no better option"
            }]
        })
        
        return tree
    
    def _generate_config(self, rules: List[RoutingRule]) -> Dict[str, Any]:
        """Generate implementation configuration"""
        config = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "routing_rules": [],
            "default_model": "gpt-3.5-turbo",
            "fallback_model": "gpt-4-turbo",
            "local_models": {
                "enabled": self.prefer_local,
                "endpoint": "http://localhost:11434",
                "models": ["llama-3.1-8b", "qwen2.5-7b", "deepseek-coder-6.7b"]
            }
        }
        
        for rule in rules:
            config["routing_rules"].append({
                "name": rule.name,
                "task_type": rule.task_type,
                "target_model": rule.target_model,
                "priority": rule.priority,
                "conditions": {
                    "task_type": rule.task_type,
                    "complexity": rule.complexity
                }
            })
        
        return config


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_optimization_report(analysis: UsageAnalysis, 
                                  recommendation: RoutingRecommendation) -> str:
    """Generate a formatted optimization report"""
    
    lines = []
    
    # Header
    lines.append("╔" + "═" * 68 + "╗")
    lines.append("║" + "LLM COST OPTIMIZATION REPORT".center(68) + "║")
    lines.append("╠" + "═" * 68 + "╣")
    
    # Summary
    lines.append("║" + f"  Analysis Period: {analysis.start_date.strftime('%Y-%m-%d')} to {analysis.end_date.strftime('%Y-%m-%d')}".ljust(67) + "║")
    lines.append("║" + f"  Total Requests: {analysis.total_requests:,}".ljust(67) + "║")
    lines.append("║" + f"  Total Tokens: {analysis.total_tokens:,}".ljust(67) + "║")
    
    # Current costs
    lines.append("╠" + "═" * 68 + "╣")
    lines.append("║" + "  CURRENT COSTS".ljust(67) + "║")
    lines.append("║" + "  ─────────────".ljust(67) + "║")
    
    for model, cost in sorted(analysis.cost_by_model.items(), key=lambda x: -x[1])[:5]:
        pct = (cost / analysis.total_cost) * 100
        line = f"  {model}: ${cost:,.2f}  ({pct:.0f}%)"
        lines.append("║" + line.ljust(67) + "║")
    
    lines.append("║" + "  ─────────────────────────────────".ljust(67) + "║")
    lines.append("║" + f"  TOTAL: ${analysis.total_cost:,.2f}".ljust(67) + "║")
    
    # Optimized routing
    lines.append("╠" + "═" * 68 + "╣")
    lines.append("║" + "  OPTIMIZED ROUTING".ljust(67) + "║")
    lines.append("║" + "  ─────────────────".ljust(67) + "║")
    
    for rule in recommendation.rules[:5]:
        line = f"  {rule.task_type} → {rule.target_model}: saves ${rule.estimated_savings:,.2f}"
        lines.append("║" + line.ljust(67) + "║")
    
    lines.append("║" + "  ─────────────────────────────────".ljust(67) + "║")
    lines.append("║" + f"  OPTIMIZED TOTAL: ${recommendation.optimized_cost:,.2f}".ljust(67) + "║")
    
    # Savings
    lines.append("╠" + "═" * 68 + "╣")
    lines.append("║" + f"  💰 MONTHLY SAVINGS: ${recommendation.monthly_savings:,.2f} ({recommendation.savings_percentage:.1f}%)".ljust(67) + "║")
    lines.append("║" + f"  💰 ANNUAL SAVINGS:  ${recommendation.annual_savings:,.2f}".ljust(67) + "║")
    lines.append("╚" + "═" * 68 + "╝")
    
    return "\n".join(lines)


def export_decision_tree_ascii(tree: Dict[str, Any], indent: int = 0) -> str:
    """Export decision tree as ASCII art"""
    lines = []
    prefix = "  " * indent
    
    name = tree.get("name", "Node")
    lines.append(f"{prefix}├── {name}")
    
    if "savings" in tree:
        lines.append(f"{prefix}│   Savings: {tree['savings']}")
    if "confidence" in tree:
        lines.append(f"{prefix}│   Confidence: {tree['confidence']}")
    if "reason" in tree:
        lines.append(f"{prefix}│   ({tree['reason']})")
    
    children = tree.get("children", [])
    for i, child in enumerate(children):
        lines.append(export_decision_tree_ascii(child, indent + 1))
    
    return "\n".join(lines)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from .analyzer import UsageAnalyzer
    
    if len(sys.argv) < 2:
        print("Usage: python optimizer.py <usage_log_file>")
        sys.exit(1)
    
    # Load and analyze
    analyzer = UsageAnalyzer()
    analyzer.load(Path(sys.argv[1]))
    analysis = analyzer.analyze()
    
    # Optimize
    optimizer = RoutingOptimizer(prefer_local=True)
    recommendation = optimizer.optimize(analysis)
    
    # Print report
    print(generate_optimization_report(analysis, recommendation))
    
    print("\n\n=== Decision Tree ===\n")
    print(export_decision_tree_ascii(recommendation.decision_tree))
    
    print("\n\n=== Top Routing Rules ===\n")
    for rule in recommendation.rules[:10]:
        print(f"  {rule.name}")
        print(f"    {rule.source_model} → {rule.target_model}")
        print(f"    Saves: ${rule.estimated_savings:.2f} (confidence: {rule.confidence*100:.0f}%)")
        print()
