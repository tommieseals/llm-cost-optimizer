#!/usr/bin/env python3
"""
LLM Cost Optimizer - Usage Analyzer
====================================
Parses API usage logs from various providers and extracts patterns.
"""

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class UsageRecord:
    """Single API usage record"""
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt: str = ""
    response: str = ""
    latency_ms: int = 0
    cost: float = 0.0
    task_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_dollar(self) -> float:
        """Efficiency metric"""
        return self.total_tokens / self.cost if self.cost > 0 else float('inf')


@dataclass
class UsageAnalysis:
    """Complete usage analysis results"""
    records: list[UsageRecord]
    start_date: datetime
    end_date: datetime
    total_requests: int
    total_tokens: int
    total_cost: float
    cost_by_model: dict[str, float]
    tokens_by_model: dict[str, int]
    requests_by_model: dict[str, int]
    cost_by_task: dict[str, float]
    tokens_by_task: dict[str, int]
    daily_costs: dict[str, float]
    hourly_distribution: dict[int, int]
    avg_tokens_per_request: float
    avg_cost_per_request: float
    task_distribution: dict[str, int]


# =============================================================================
# MODEL PRICING DATABASE (per 1M tokens)
# =============================================================================

MODEL_PRICING = {
    # OpenAI
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-16k": {"input": 3.00, "output": 4.00},

    # Anthropic
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-2": {"input": 8.00, "output": 24.00},

    # Google
    "gemini-pro": {"input": 0.50, "output": 1.50},
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},

    # Local/Open Source (effectively free)
    "llama-3.1-8b": {"input": 0.00, "output": 0.00},
    "llama-3.1-70b": {"input": 0.00, "output": 0.00},
    "llama-3.1-405b": {"input": 0.00, "output": 0.00},
    "qwen2.5-3b": {"input": 0.00, "output": 0.00},
    "qwen2.5-7b": {"input": 0.00, "output": 0.00},
    "qwen2.5-14b": {"input": 0.00, "output": 0.00},
    "deepseek-coder-6.7b": {"input": 0.00, "output": 0.00},
    "mistral-7b": {"input": 0.00, "output": 0.00},
    "phi3-mini": {"input": 0.00, "output": 0.00},

    # Cloud Open Source (via API)
    "llama-3.1-70b-instruct": {"input": 0.52, "output": 0.75},
    "llama-3.1-8b-instruct": {"input": 0.06, "output": 0.06},
    "mistral-large": {"input": 2.00, "output": 6.00},
    "mixtral-8x7b": {"input": 0.24, "output": 0.24},
}

# =============================================================================
# TASK CLASSIFICATION
# =============================================================================

TASK_PATTERNS = {
    "code": [
        r"\b(write|create|implement|fix|debug|refactor)\b.*\b(code|function|class|method|script)\b",
        r"\b(python|javascript|java|typescript|rust|go|c\+\+|ruby)\b",
        r"\b(def |class |function |const |let |var |import |from )\b",
        r"```\w*\n",  # Code blocks
        r"\b(API|REST|GraphQL|endpoint|database|SQL|query)\b",
    ],
    "analysis": [
        r"\b(analyze|analysis|compare|evaluate|assess|review)\b",
        r"\b(data|metrics|statistics|trends|patterns)\b",
        r"\b(report|summary|findings|insights|conclusions)\b",
    ],
    "creative": [
        r"\b(write|create|generate|compose)\b.*\b(story|poem|article|essay|blog)\b",
        r"\b(creative|imaginative|original|unique)\b",
        r"\b(fiction|narrative|character|plot)\b",
    ],
    "simple": [
        r"^(what|who|when|where|why|how)\s+(is|are|was|were|do|does|did)\b",
        r"\b(define|explain|describe|tell me about)\b",
        r"^\w+\?$",  # Single word questions
    ],
    "translation": [
        r"\b(translate|translation|convert)\b.*\b(to|into|from)\b",
        r"\b(language|english|spanish|french|german|chinese|japanese)\b",
    ],
    "summarization": [
        r"\b(summarize|summary|summarization|tldr|brief)\b",
        r"\b(key points|main ideas|highlights|overview)\b",
    ],
    "reasoning": [
        r"\b(think|reason|logic|deduce|infer|conclude)\b",
        r"\b(step by step|chain of thought|reasoning|problem solving)\b",
        r"\b(complex|difficult|challenging|advanced)\b",
        r"\b(math|calculation|equation|formula)\b",
    ],
    "extraction": [
        r"\b(extract|parse|find|identify|locate)\b",
        r"\b(entities|names|dates|numbers|keywords)\b",
        r"\b(JSON|structured|format|schema)\b",
    ],
}


def classify_task(prompt: str) -> str:
    """Classify a prompt into task categories"""
    prompt_lower = prompt.lower()

    scores = defaultdict(int)

    for task_type, patterns in TASK_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE | re.MULTILINE):
                scores[task_type] += 1

    if not scores:
        # Fallback heuristics
        if len(prompt) < 100:
            return "simple"
        elif len(prompt) > 1000:
            return "complex"
        return "general"

    return max(scores, key=scores.get)


def estimate_complexity(prompt: str, completion_tokens: int) -> str:
    """Estimate task complexity"""
    prompt_len = len(prompt)

    if prompt_len < 100 and completion_tokens < 200:
        return "simple"
    elif prompt_len < 500 and completion_tokens < 500:
        return "medium"
    elif prompt_len < 2000 and completion_tokens < 1500:
        return "complex"
    else:
        return "very_complex"


# =============================================================================
# LOG PARSERS
# =============================================================================

def parse_openai_export(filepath: Path) -> list[UsageRecord]:
    """Parse OpenAI API usage export"""
    records = []

    with open(filepath) as f:
        data = json.load(f)

    # Handle different export formats
    entries = data if isinstance(data, list) else data.get("data", data.get("usage", []))

    for entry in entries:
        try:
            record = UsageRecord(
                timestamp=datetime.fromisoformat(entry.get("timestamp", entry.get("created_at", ""))),
                model=normalize_model_name(entry.get("model", "unknown")),
                prompt_tokens=entry.get("prompt_tokens", entry.get("n_context_tokens_total", 0)),
                completion_tokens=entry.get("completion_tokens", entry.get("n_generated_tokens_total", 0)),
                total_tokens=entry.get("total_tokens", 0),
                prompt=entry.get("prompt", entry.get("request", {}).get("prompt", "")),
                response=entry.get("response", entry.get("choices", [{}])[0].get("text", "")),
                latency_ms=entry.get("latency_ms", 0),
                metadata=entry
            )

            if record.total_tokens == 0:
                record.total_tokens = record.prompt_tokens + record.completion_tokens

            record.task_type = classify_task(record.prompt)
            record.cost = calculate_cost(record.model, record.prompt_tokens, record.completion_tokens)

            records.append(record)
        except Exception as e:
            print(f"Warning: Failed to parse entry: {e}")
            continue

    return records


def parse_anthropic_export(filepath: Path) -> list[UsageRecord]:
    """Parse Anthropic API usage export"""
    records = []

    with open(filepath) as f:
        data = json.load(f)

    entries = data if isinstance(data, list) else data.get("messages", [])

    for entry in entries:
        try:
            usage = entry.get("usage", {})
            record = UsageRecord(
                timestamp=datetime.fromisoformat(entry.get("created_at", "")),
                model=normalize_model_name(entry.get("model", "unknown")),
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                prompt=extract_anthropic_prompt(entry),
                response=entry.get("content", [{}])[0].get("text", ""),
                metadata=entry
            )

            record.task_type = classify_task(record.prompt)
            record.cost = calculate_cost(record.model, record.prompt_tokens, record.completion_tokens)

            records.append(record)
        except Exception as e:
            print(f"Warning: Failed to parse entry: {e}")
            continue

    return records


def parse_csv_logs(filepath: Path) -> list[UsageRecord]:
    """Parse generic CSV usage logs"""
    records = []

    with open(filepath) as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                # Flexible column name matching
                timestamp = row.get("timestamp") or row.get("created_at") or row.get("date")
                model = row.get("model") or row.get("model_id") or "unknown"
                prompt_tokens = int(row.get("prompt_tokens") or row.get("input_tokens") or 0)
                completion_tokens = int(row.get("completion_tokens") or row.get("output_tokens") or 0)
                total_tokens = int(row.get("total_tokens") or 0) or (prompt_tokens + completion_tokens)
                prompt = row.get("prompt") or row.get("input") or row.get("request") or ""

                record = UsageRecord(
                    timestamp=datetime.fromisoformat(timestamp) if timestamp else datetime.now(),
                    model=normalize_model_name(model),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    prompt=prompt,
                    metadata=dict(row)
                )

                record.task_type = classify_task(record.prompt)
                record.cost = calculate_cost(record.model, record.prompt_tokens, record.completion_tokens)

                records.append(record)
            except Exception as e:
                print(f"Warning: Failed to parse row: {e}")
                continue

    return records


def parse_custom_json(filepath: Path) -> list[UsageRecord]:
    """Parse custom JSON format with flexible field mapping"""
    records = []

    with open(filepath) as f:
        data = json.load(f)

    # Handle various structures
    if isinstance(data, list):
        entries = data
    elif "requests" in data:
        entries = data["requests"]
    elif "logs" in data:
        entries = data["logs"]
    elif "usage" in data:
        entries = data["usage"]
    else:
        entries = [data]

    for entry in entries:
        try:
            record = UsageRecord(
                timestamp=parse_timestamp(entry),
                model=normalize_model_name(find_field(entry, ["model", "model_id", "model_name"])),
                prompt_tokens=int(find_field(entry, ["prompt_tokens", "input_tokens", "n_prompt"], 0)),
                completion_tokens=int(find_field(entry, ["completion_tokens", "output_tokens", "n_completion"], 0)),
                total_tokens=int(find_field(entry, ["total_tokens", "tokens", "n_tokens"], 0)),
                prompt=find_field(entry, ["prompt", "input", "request", "query"], ""),
                response=find_field(entry, ["response", "output", "completion", "answer"], ""),
                latency_ms=int(find_field(entry, ["latency_ms", "latency", "duration_ms"], 0)),
                metadata=entry
            )

            if record.total_tokens == 0:
                record.total_tokens = record.prompt_tokens + record.completion_tokens

            record.task_type = classify_task(record.prompt)
            record.cost = calculate_cost(record.model, record.prompt_tokens, record.completion_tokens)

            records.append(record)
        except Exception as e:
            print(f"Warning: Failed to parse entry: {e}")
            continue

    return records


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_model_name(model: str) -> str:
    """Normalize model names across providers"""
    model = model.lower().strip()

    # Map common variations
    mappings = {
        "gpt-4-0125-preview": "gpt-4-turbo",
        "gpt-4-1106-preview": "gpt-4-turbo",
        "gpt-4-vision-preview": "gpt-4-turbo",
        "gpt-3.5-turbo-0125": "gpt-3.5-turbo",
        "gpt-3.5-turbo-1106": "gpt-3.5-turbo",
        "claude-3-opus-20240229": "claude-3-opus",
        "claude-3-sonnet-20240229": "claude-3-sonnet",
        "claude-3-haiku-20240307": "claude-3-haiku",
        "claude-3-5-sonnet-20240620": "claude-3.5-sonnet",
    }

    return mappings.get(model, model)


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost for a request"""
    pricing = MODEL_PRICING.get(model, {"input": 1.0, "output": 2.0})

    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def find_field(data: dict, field_names: list[str], default: Any = "") -> Any:
    """Find field by trying multiple names"""
    for name in field_names:
        if name in data:
            return data[name]
        # Check nested
        if "." in name:
            parts = name.split(".")
            value = data
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    break
            else:
                return value
    return default


def parse_timestamp(entry: dict) -> datetime:
    """Parse timestamp from various formats"""
    timestamp_fields = ["timestamp", "created_at", "time", "date", "datetime"]

    for field_name in timestamp_fields:
        if field_name in entry:
            value = entry[field_name]
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(value)
            elif isinstance(value, str):
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00"))
                except Exception:
                    pass

    return datetime.now()


def extract_anthropic_prompt(entry: dict) -> str:
    """Extract prompt from Anthropic message format"""
    messages = entry.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return " ".join(texts)
    return ""


# =============================================================================
# MAIN ANALYZER
# =============================================================================

class UsageAnalyzer:
    """Main usage analyzer class"""

    def __init__(self):
        self.records: list[UsageRecord] = []

    def load(self, filepath: Path, format: str = "auto") -> int:
        """Load usage logs from file"""
        filepath = Path(filepath)

        if format == "auto":
            format = self._detect_format(filepath)

        if format == "openai":
            records = parse_openai_export(filepath)
        elif format == "anthropic":
            records = parse_anthropic_export(filepath)
        elif format == "csv":
            records = parse_csv_logs(filepath)
        else:
            records = parse_custom_json(filepath)

        self.records.extend(records)
        return len(records)

    def _detect_format(self, filepath: Path) -> str:
        """Auto-detect log format"""
        suffix = filepath.suffix.lower()

        if suffix == ".csv":
            return "csv"

        with open(filepath) as f:
            content = f.read(1000)

        if '"model": "claude' in content or '"claude-' in content:
            return "anthropic"
        elif '"model": "gpt-' in content or "openai" in content.lower():
            return "openai"

        return "json"

    def analyze(self, start_date: datetime = None, end_date: datetime = None) -> UsageAnalysis:
        """Perform comprehensive usage analysis"""
        records = self.records

        # Filter by date if specified
        if start_date:
            records = [r for r in records if r.timestamp >= start_date]
        if end_date:
            records = [r for r in records if r.timestamp <= end_date]

        if not records:
            raise ValueError("No records to analyze")

        # Aggregate metrics
        cost_by_model = defaultdict(float)
        tokens_by_model = defaultdict(int)
        requests_by_model = defaultdict(int)
        cost_by_task = defaultdict(float)
        tokens_by_task = defaultdict(int)
        daily_costs = defaultdict(float)
        hourly_distribution = defaultdict(int)
        task_distribution = defaultdict(int)

        for record in records:
            cost_by_model[record.model] += record.cost
            tokens_by_model[record.model] += record.total_tokens
            requests_by_model[record.model] += 1
            cost_by_task[record.task_type] += record.cost
            tokens_by_task[record.task_type] += record.total_tokens
            daily_costs[record.timestamp.strftime("%Y-%m-%d")] += record.cost
            hourly_distribution[record.timestamp.hour] += 1
            task_distribution[record.task_type] += 1

        total_tokens = sum(r.total_tokens for r in records)
        total_cost = sum(r.cost for r in records)

        return UsageAnalysis(
            records=records,
            start_date=min(r.timestamp for r in records),
            end_date=max(r.timestamp for r in records),
            total_requests=len(records),
            total_tokens=total_tokens,
            total_cost=total_cost,
            cost_by_model=dict(cost_by_model),
            tokens_by_model=dict(tokens_by_model),
            requests_by_model=dict(requests_by_model),
            cost_by_task=dict(cost_by_task),
            tokens_by_task=dict(tokens_by_task),
            daily_costs=dict(daily_costs),
            hourly_distribution=dict(hourly_distribution),
            avg_tokens_per_request=total_tokens / len(records),
            avg_cost_per_request=total_cost / len(records),
            task_distribution=dict(task_distribution)
        )

    def get_patterns(self) -> dict[str, Any]:
        """Identify usage patterns"""
        patterns = {
            "peak_hours": [],
            "expensive_tasks": [],
            "optimization_candidates": [],
            "repetitive_queries": [],
        }

        # Find peak hours
        hourly = defaultdict(int)
        for r in self.records:
            hourly[r.timestamp.hour] += 1

        if hourly:
            avg_requests = sum(hourly.values()) / 24
            patterns["peak_hours"] = [h for h, c in hourly.items() if c > avg_requests * 1.5]

        # Find expensive tasks
        task_costs = defaultdict(float)
        task_counts = defaultdict(int)
        for r in self.records:
            task_costs[r.task_type] += r.cost
            task_counts[r.task_type] += 1

        if task_costs:
            avg_cost = sum(task_costs.values()) / len(task_costs)
            patterns["expensive_tasks"] = [
                {"task": t, "total_cost": c, "count": task_counts[t]}
                for t, c in task_costs.items()
                if c > avg_cost * 1.5
            ]

        # Find optimization candidates (expensive model for simple tasks)
        expensive_models = {"gpt-4", "gpt-4-turbo", "claude-3-opus", "claude-3.5-sonnet"}
        simple_tasks = {"simple", "extraction", "translation"}

        candidates = [
            r for r in self.records
            if r.model in expensive_models and r.task_type in simple_tasks
        ]

        patterns["optimization_candidates"] = {
            "count": len(candidates),
            "potential_savings": sum(r.cost for r in candidates) * 0.9,  # 90% savings possible
            "examples": [{"prompt": r.prompt[:100], "model": r.model, "cost": r.cost}
                        for r in candidates[:5]]
        }

        return patterns


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <usage_log_file>")
        sys.exit(1)

    analyzer = UsageAnalyzer()
    count = analyzer.load(Path(sys.argv[1]))
    print(f"Loaded {count} records")

    analysis = analyzer.analyze()

    print("\n=== Usage Analysis ===")
    print(f"Period: {analysis.start_date} to {analysis.end_date}")
    print(f"Total Requests: {analysis.total_requests:,}")
    print(f"Total Tokens: {analysis.total_tokens:,}")
    print(f"Total Cost: ${analysis.total_cost:.2f}")

    print("\nCost by Model:")
    for model, cost in sorted(analysis.cost_by_model.items(), key=lambda x: -x[1]):
        pct = (cost / analysis.total_cost) * 100
        print(f"  {model}: ${cost:.2f} ({pct:.1f}%)")

    print("\nTask Distribution:")
    for task, count in sorted(analysis.task_distribution.items(), key=lambda x: -x[1]):
        print(f"  {task}: {count}")
