# 🚀 LLM Cost Optimizer

**Slash your LLM API costs by 40-70% with intelligent routing analysis.**

[![CI](https://github.com/tommieseals/llm-cost-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/tommieseals/llm-cost-optimizer/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📊 The Problem

Most teams overpay for LLM APIs because they:
- Use expensive models (GPT-4, Claude) for simple tasks that cheaper models handle fine
- Don't track actual usage patterns to identify optimization opportunities
- Lack visibility into cost distribution across use cases
- Miss opportunities to route to local/open-source models

**The cost difference is massive:**
| Model | Cost per 1M tokens |
|-------|-------------------|
| GPT-4 Turbo | $30.00 |
| Claude 3.5 Sonnet | $15.00 |
| GPT-3.5 Turbo | $2.00 |
| Llama 3.1 70B (local) | $0.00 |
| Qwen 2.5 3B (local) | $0.00 |

## 💡 The Solution

LLM Cost Optimizer analyzes your API usage logs and recommends an optimal routing strategy:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Cost Optimizer                           │
├─────────────────────────────────────────────────────────────────┤
│  📥 Input: API Usage Logs (JSON/CSV)                           │
│       ↓                                                         │
│  🔍 Analysis: Task classification, cost breakdown, patterns     │
│       ↓                                                         │
│  🎯 Optimization: Model routing recommendations                 │
│       ↓                                                         │
│  📈 Output: Savings report + decision tree + visualizations    │
└─────────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

- **📊 Usage Analysis** - Parse logs from OpenAI, Anthropic, or custom formats
- **💰 Cost Calculation** - Accurate pricing across 20+ models
- **🎯 Smart Routing** - Task-aware model selection (code→coder, simple→cheap)
- **📈 Savings Projection** - Before/after comparison with projected annual savings
- **🌳 Decision Tree** - Visual routing logic you can implement
- **📉 Visualization** - Cost breakdown charts, trend analysis

## 🏃 Quick Start

### Installation

```bash
# Clone
git clone https://github.com/tommieseals/llm-cost-optimizer.git
cd llm-cost-optimizer

# Install
pip install -e .
# or
make install
```

### Analyze Your Usage

```bash
# From OpenAI export
llm-optimize analyze usage_export.json

# From custom CSV
llm-optimize analyze logs.csv --format csv

# Generate full report
llm-optimize report usage_export.json --output report/
```

### Example Output

```
╔══════════════════════════════════════════════════════════════════╗
║                    LLM COST OPTIMIZATION REPORT                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Analysis Period: 2024-01-01 to 2024-01-31                       ║
║  Total Requests: 15,432                                           ║
║  Total Tokens: 48,293,281                                         ║
╠══════════════════════════════════════════════════════════════════╣
║  CURRENT COSTS                                                    ║
║  ─────────────                                                    ║
║  GPT-4 Turbo:     $892.45  (62%)                                 ║
║  Claude Sonnet:   $412.32  (29%)                                 ║
║  GPT-3.5 Turbo:   $127.88  (9%)                                  ║
║  ─────────────────────────────                                   ║
║  TOTAL:          $1,432.65                                        ║
╠══════════════════════════════════════════════════════════════════╣
║  OPTIMIZED ROUTING                                                ║
║  ─────────────────                                                ║
║  Simple queries → Llama 3.1 8B (local):     $0.00  (was $312.40) ║
║  Code tasks → DeepSeek Coder 6.7B (local):  $0.00  (was $245.80) ║
║  Complex reasoning → GPT-4 (keep):        $334.45  (unchanged)   ║
║  ─────────────────────────────────                               ║
║  OPTIMIZED TOTAL:  $521.33                                        ║
╠══════════════════════════════════════════════════════════════════╣
║  💰 MONTHLY SAVINGS: $911.32 (63.6%)                              ║
║  💰 ANNUAL SAVINGS:  $10,935.84                                   ║
╚══════════════════════════════════════════════════════════════════╝
```

## 🏗️ Architecture

```
                           ┌─────────────────┐
                           │   Usage Logs    │
                           │  (JSON/CSV)     │
                           └────────┬────────┘
                                    │
                           ┌────────▼────────┐
                           │    Analyzer     │
                           │  ────────────   │
                           │ • Parse logs    │
                           │ • Classify tasks│
                           │ • Calculate cost│
                           └────────┬────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
     ┌────────▼────────┐   ┌────────▼────────┐   ┌───────▼────────┐
     │  Task Classifier │   │ Cost Calculator │   │ Pattern Finder │
     │  ─────────────── │   │ ─────────────── │   │ ────────────── │
     │ • Simple queries │   │ • Token pricing │   │ • Usage trends │
     │ • Code tasks     │   │ • Model costs   │   │ • Peak hours   │
     │ • Complex reason │   │ • Batch savings │   │ • Repetition   │
     └────────┬─────────┘   └────────┬────────┘   └───────┬────────┘
              │                      │                    │
              └──────────────────────┼────────────────────┘
                                     │
                           ┌─────────▼─────────┐
                           │    Optimizer      │
                           │   ────────────    │
                           │ • Route mapping   │
                           │ • Cost projection │
                           │ • Decision tree   │
                           └─────────┬─────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
     ┌────────▼────────┐    ┌────────▼────────┐    ┌───────▼───────┐
     │  Savings Report │    │  Decision Tree  │    │    Charts     │
     │  ─────────────  │    │  ─────────────  │    │  ──────────   │
     │ • Before/After  │    │ • Visual logic  │    │ • Cost pie    │
     │ • Monthly save  │    │ • Implementable │    │ • Trends      │
     │ • Annual proj   │    │ • Export config │    │ • Comparison  │
     └─────────────────┘    └─────────────────┘    └───────────────┘
```

## 📖 Documentation

- [Architecture Deep Dive](docs/architecture.md)
- [Supported Log Formats](docs/formats.md)
- [Model Pricing Database](docs/pricing.md)

## 🔧 CLI Reference

```bash
# Analyze usage logs
llm-optimize analyze <file> [--format json|csv] [--start DATE] [--end DATE]

# Generate full report with visualizations
llm-optimize report <file> [--output DIR] [--format html|md|json]

# Show cost breakdown
llm-optimize costs <file> [--by model|task|day]

# Export routing configuration
llm-optimize export-config <file> [--format yaml|json]

# Interactive mode
llm-optimize interactive
```

## 🎯 Task Classification

The optimizer classifies your requests into categories:

| Category | Indicators | Recommended Model |
|----------|-----------|-------------------|
| **Simple** | <100 tokens, basic Q&A | Llama 3.1 8B (local) |
| **Code** | Programming keywords, syntax | DeepSeek Coder 6.7B |
| **Analysis** | Data, compare, evaluate | Qwen 2.5 14B |
| **Creative** | Write, story, generate | Mistral 7B |
| **Complex** | Multi-step, reasoning | GPT-4 / Claude (keep) |
| **Vision** | Image analysis | Llama 90B Vision |

## 📊 Sample Visualization

### Cost Distribution (Before Optimization)
```
GPT-4 Turbo    ████████████████████████████████░░░░░░░░  62%  $892
Claude Sonnet  ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░  29%  $412
GPT-3.5        █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   9%  $128
```

### Cost Distribution (After Optimization)
```
GPT-4 Turbo    ████████████████████████████████░░░░░░░░  64%  $334
Local Models   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%  $0
Cloud Cheap    ████████████████████░░░░░░░░░░░░░░░░░░░░  36%  $187
```

## 🛠️ Development

```bash
# Install dev dependencies
make dev-install

# Run tests
make test

# Run linting
make lint

# Build Docker image
make docker-build
```

## 🐳 Docker

```bash
# Build
docker build -t llm-cost-optimizer .

# Run analysis
docker run -v $(pwd)/logs:/data llm-cost-optimizer analyze /data/usage.json
```

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

Built with insights from running multi-model LLM infrastructure in production.

---

**Questions?** Open an issue or reach out!

*Stop overpaying for LLM APIs. Optimize today.* 🚀
