# Architecture

This document describes the architecture of the LLM Cost Optimizer.

## Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                      LLM Cost Optimizer                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   CLI       │───▶│   Analyzer   │───▶│   Optimizer  │          │
│  │  (cli.py)   │    │(analyzer.py) │    │(optimizer.py)│          │
│  └─────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         │                   │                   │                   │
│         │           ┌───────▼───────┐   ┌───────▼───────┐          │
│         │           │ UsageAnalysis │   │ Recommendation│          │
│         │           └───────────────┘   └───────────────┘          │
│         │                   │                   │                   │
│         │                   └─────────┬─────────┘                   │
│         │                             │                             │
│         │                   ┌─────────▼─────────┐                   │
│         └──────────────────▶│    Visualizer     │                   │
│                             │  (visualizer.py)  │                   │
│                             └───────────────────┘                   │
│                                       │                             │
│                    ┌──────────────────┼──────────────────┐         │
│                    │                  │                  │         │
│             ┌──────▼──────┐   ┌───────▼──────┐   ┌──────▼──────┐  │
│             │  ASCII Art  │   │  HTML Report │   │   Charts    │  │
│             └─────────────┘   └──────────────┘   └─────────────┘  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. CLI (`cli.py`)

The command-line interface provides the main entry point for users. It supports:

- **analyze**: Parse and analyze usage logs
- **optimize**: Generate routing recommendations  
- **report**: Create full optimization reports
- **costs**: Show cost breakdowns
- **export-config**: Export routing configuration
- **interactive**: Interactive REPL mode

### 2. Analyzer (`analyzer.py`)

Parses usage logs from various providers:

- **OpenAI** - Native export format
- **Anthropic** - Claude API logs
- **CSV** - Generic tabular format
- **JSON** - Custom formats with auto-detection

Key features:
- Task classification using NLP patterns
- Cost calculation with 20+ model pricing database
- Pattern detection (peak hours, expensive tasks, etc.)

### 3. Optimizer (`optimizer.py`)

Generates routing recommendations:

- **Task-aware routing**: Match tasks to optimal models
- **Cost projection**: Calculate potential savings
- **Decision tree**: Visual routing logic
- **Config export**: Implementable routing rules

### 4. Visualizer (`visualizer.py`)

Creates visual reports:

- **ASCII charts**: Terminal-friendly visualizations
- **HTML reports**: Interactive web dashboards
- **Markdown**: Documentation-ready reports
- **Matplotlib**: Publication-quality charts (optional)

## Data Flow

```
Input Logs          Parse          Analyze         Optimize        Output
───────────────────────────────────────────────────────────────────────────
                    
  OpenAI.json ─┐                  ┌───────────┐   ┌───────────┐
               │    ┌─────────┐   │ Usage     │   │ Routing   │   Report
  Anthropic ───┼───▶│ Parser  │──▶│ Analysis  │──▶│ Rules     │──▶ Charts
               │    └─────────┘   │           │   │           │   Config
  Custom.csv ──┘                  └───────────┘   └───────────┘
```

## Model Pricing Database

The system includes pricing for 20+ models:

| Category | Models | Pricing (per 1M tokens) |
|----------|--------|------------------------|
| Premium | GPT-4, Claude 3 Opus | $15-60 |
| Mid-tier | GPT-4 Turbo, Claude 3.5 Sonnet | $3-30 |
| Economy | GPT-3.5 Turbo, Gemini Flash | $0.15-2 |
| Local | Llama, Qwen, DeepSeek | $0 |

## Task Classification

Tasks are classified using regex patterns:

```python
TASK_PATTERNS = {
    "code": [r"\b(write|debug|implement)\b.*\b(code|function)\b"],
    "simple": [r"^(what|who|when|where|why|how)\s+(is|are)\b"],
    "analysis": [r"\b(analyze|compare|evaluate)\b"],
    "creative": [r"\b(write|create)\b.*\b(story|poem)\b"],
    ...
}
```

## Routing Decision Tree

```
                    ┌──────────────────┐
                    │  Incoming Request │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Classify Task   │
                    └────────┬─────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐         ┌─────▼────┐         ┌────▼────┐
   │ Simple? │         │  Code?   │         │ Complex │
   └────┬────┘         └────┬─────┘         └────┬────┘
        │                   │                    │
   ┌────▼────────┐    ┌─────▼──────────┐   ┌────▼────────┐
   │ Local 8B   │    │ DeepSeek Coder │   │ Keep GPT-4  │
   │ (Llama/Qwen)│    │ (6.7B)         │   │ /Claude     │
   └─────────────┘    └────────────────┘   └─────────────┘
```

## Configuration Format

```yaml
version: "1.0"
routing_rules:
  - name: route_simple
    task_type: simple
    target_model: llama-3.1-8b
    priority: 100
    conditions:
      prompt_tokens_lt: 500
      
  - name: route_code
    task_type: code
    target_model: deepseek-coder-6.7b
    priority: 60
    
default_model: gpt-3.5-turbo
fallback_model: gpt-4-turbo

local_models:
  enabled: true
  endpoint: http://localhost:11434
```

## Extension Points

### Adding New Providers

1. Create parser in `analyzer.py`:
```python
def parse_newprovider_export(filepath: Path) -> List[UsageRecord]:
    ...
```

2. Register in `_detect_format()`:
```python
if "newprovider" in content:
    return "newprovider"
```

### Adding New Models

Update `MODEL_PRICING` in `analyzer.py`:
```python
MODEL_PRICING["new-model"] = {"input": 1.0, "output": 2.0}
```

### Custom Task Patterns

Add patterns to `TASK_PATTERNS`:
```python
TASK_PATTERNS["new_task"] = [r"pattern1", r"pattern2"]
```

## Performance Considerations

- **Memory**: Records are loaded into memory; for very large logs, consider streaming
- **Speed**: Analysis is O(n) where n = number of records
- **Caching**: Consider caching parsed results for repeated analysis

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_analyzer.py -v

# Coverage report
pytest --cov=src tests/
```
