#!/usr/bin/env python3
"""
LLM Cost Optimizer - Visualizer
================================
Generate charts, decision trees, and visual reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .analyzer import UsageAnalysis
from .optimizer import RoutingRecommendation

# Try to import optional visualization libraries
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# ASCII VISUALIZATION (No Dependencies)
# =============================================================================


def ascii_bar_chart(data: dict[str, float], title: str = "", width: int = 50) -> str:
    """Generate ASCII horizontal bar chart"""
    if not data:
        return "No data to display"

    lines = []
    if title:
        lines.append(f"\n{title}")
        lines.append("=" * len(title))

    max_val = max(data.values())
    max_label_len = max(len(str(k)) for k in data.keys())

    for label, value in sorted(data.items(), key=lambda x: -x[1]):
        bar_len = int((value / max_val) * width) if max_val > 0 else 0
        bar = "█" * bar_len + "░" * (width - bar_len)
        pct = (value / sum(data.values())) * 100 if sum(data.values()) > 0 else 0

        if isinstance(value, float) and value < 100:
            val_str = f"${value:.2f}"
        else:
            val_str = f"{value:,.0f}"

        lines.append(f"{label:<{max_label_len}}  {bar}  {pct:5.1f}%  {val_str}")

    return "\n".join(lines)


def ascii_pie_chart(data: dict[str, float], title: str = "") -> str:
    """Generate ASCII pie chart representation"""
    if not data:
        return "No data to display"

    lines = []
    if title:
        lines.append(f"\n{title}")
        lines.append("=" * len(title))

    total = sum(data.values())

    # Simple percentage breakdown
    for label, value in sorted(data.items(), key=lambda x: -x[1]):
        pct = (value / total) * 100 if total > 0 else 0
        blocks = int(pct / 5)  # Each block = 5%
        bar = "▓" * blocks + "░" * (20 - blocks)

        if isinstance(value, float) and value < 100:
            val_str = f"${value:.2f}"
        else:
            val_str = f"{value:,.0f}"

        lines.append(f"  {label:<20} {bar} {pct:5.1f}% ({val_str})")

    return "\n".join(lines)


def ascii_comparison_chart(
    before: dict[str, float], after: dict[str, float], title: str = "Before/After Comparison"
) -> str:
    """Generate side-by-side comparison chart"""
    lines = []
    lines.append(f"\n{title}")
    lines.append("=" * len(title))

    all_keys = set(before.keys()) | set(after.keys())
    max_label = max(len(k) for k in all_keys) if all_keys else 10

    lines.append(f"{'Item':<{max_label}}  {'BEFORE':>12}  {'AFTER':>12}  {'SAVINGS':>12}  CHANGE")
    lines.append("-" * (max_label + 60))

    total_before = 0
    total_after = 0

    for key in sorted(all_keys):
        b = before.get(key, 0)
        a = after.get(key, 0)
        savings = b - a
        pct_change = ((b - a) / b * 100) if b > 0 else 0

        total_before += b
        total_after += a

        # Visual indicator
        if savings > 0:
            indicator = "↓ " + "▼" * min(int(pct_change / 10), 10)
        elif savings < 0:
            indicator = "↑ " + "▲" * min(int(abs(pct_change) / 10), 10)
        else:
            indicator = "─"

        lines.append(
            f"{key:<{max_label}}  ${b:>10.2f}  ${a:>10.2f}  ${savings:>10.2f}  {indicator}"
        )

    lines.append("-" * (max_label + 60))
    total_savings = total_before - total_after
    pct_savings = (total_savings / total_before * 100) if total_before > 0 else 0
    lines.append(
        f"{'TOTAL':<{max_label}}  ${total_before:>10.2f}  ${total_after:>10.2f}  ${total_savings:>10.2f}  ({pct_savings:.1f}%)"
    )

    return "\n".join(lines)


def ascii_decision_tree(tree: dict[str, Any], prefix: str = "", is_last: bool = True) -> str:
    """Generate ASCII decision tree visualization"""
    lines = []

    connector = "└── " if is_last else "├── "
    extension = "    " if is_last else "│   "

    name = tree.get("name", "Node")
    lines.append(f"{prefix}{connector}{name}")

    # Add details
    details = []
    if "savings" in tree:
        details.append(f"💰 {tree['savings']}")
    if "confidence" in tree:
        details.append(f"📊 {tree['confidence']}")
    if "reason" in tree:
        details.append(f"📝 {tree['reason']}")

    if details:
        for detail in details:
            lines.append(f"{prefix}{extension}    {detail}")

    children = tree.get("children", [])
    for i, child in enumerate(children):
        is_last_child = i == len(children) - 1
        child_tree = ascii_decision_tree(child, prefix + extension, is_last_child)
        lines.append(child_tree)

    return "\n".join(lines)


def ascii_trend_chart(daily_data: dict[str, float], title: str = "Daily Trend") -> str:
    """Generate ASCII trend chart with sparkline"""
    if not daily_data:
        return "No trend data available"

    lines = []
    lines.append(f"\n{title}")
    lines.append("=" * len(title))

    # Sort by date
    sorted_data = sorted(daily_data.items())
    values = [v for _, v in sorted_data]

    if not values:
        return "No data"

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val > min_val else 1

    # Generate sparkline
    sparkline_chars = "▁▂▃▄▅▆▇█"
    sparkline = ""
    for v in values:
        normalized = (v - min_val) / range_val
        idx = min(int(normalized * (len(sparkline_chars) - 1)), len(sparkline_chars) - 1)
        sparkline += sparkline_chars[idx]

    lines.append(f"\nSparkline: {sparkline}")
    lines.append(f"Range: ${min_val:.2f} - ${max_val:.2f}")
    lines.append(f"Average: ${sum(values) / len(values):.2f}")

    # Show first and last 5 days
    lines.append("\nRecent Days:")
    for date, value in sorted_data[-7:]:
        bar_len = int((value - min_val) / range_val * 30) if range_val > 0 else 0
        bar = "█" * bar_len
        lines.append(f"  {date}: ${value:>8.2f} {bar}")

    return "\n".join(lines)


# =============================================================================
# HTML VISUALIZATION
# =============================================================================


def generate_html_report(
    analysis: UsageAnalysis, recommendation: RoutingRecommendation, output_path: Path
) -> Path:
    """Generate interactive HTML report"""

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Cost Optimization Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{
            background: white;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{ color: #1a1a2e; margin-bottom: 8px; }}
        h2 {{ color: #4a4a6a; margin-bottom: 16px; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .stat {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 20px;
            color: white;
            text-align: center;
        }}
        .stat-value {{ font-size: 2rem; font-weight: bold; }}
        .stat-label {{ opacity: 0.9; }}
        .savings {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .chart-container {{ height: 300px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .tag {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        .tag-local {{ background: #d4edda; color: #155724; }}
        .tag-cloud {{ background: #cce5ff; color: #004085; }}
        .progress {{ height: 8px; background: #eee; border-radius: 4px; overflow: hidden; }}
        .progress-bar {{ height: 100%; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>🚀 LLM Cost Optimization Report</h1>
            <p style="color: #666; margin-top: 8px;">
                Analysis Period: {analysis.start_date.strftime("%Y-%m-%d")} to {
        analysis.end_date.strftime("%Y-%m-%d")
    }
            </p>
        </div>

        <div class="stats">
            <div class="stat">
                <div class="stat-value">{analysis.total_requests:,}</div>
                <div class="stat-label">Total Requests</div>
            </div>
            <div class="stat">
                <div class="stat-value">{analysis.total_tokens:,}</div>
                <div class="stat-label">Total Tokens</div>
            </div>
            <div class="stat">
                <div class="stat-value">${analysis.total_cost:,.2f}</div>
                <div class="stat-label">Current Cost</div>
            </div>
            <div class="stat savings">
                <div class="stat-value">${recommendation.monthly_savings:,.2f}</div>
                <div class="stat-label">Monthly Savings ({
        recommendation.savings_percentage:.1f}%)</div>
            </div>
        </div>

        <div class="card">
            <h2>📊 Cost by Model</h2>
            <div class="chart-container">
                <canvas id="modelChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>📈 Cost Comparison: Before vs After</h2>
            <div class="chart-container">
                <canvas id="comparisonChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>🎯 Routing Recommendations</h2>
            <table>
                <thead>
                    <tr>
                        <th>Task Type</th>
                        <th>Current Model</th>
                        <th>Recommended</th>
                        <th>Monthly Savings</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {
        "".join(
            f"""
                    <tr>
                        <td><strong>{rule.task_type}</strong></td>
                        <td>{rule.source_model}</td>
                        <td>
                            {rule.target_model}
                            <span class="tag {'tag-local' if 'llama' in rule.target_model or 'qwen' in rule.target_model or 'deepseek' in rule.target_model else 'tag-cloud'}">
                                {'Local' if 'llama' in rule.target_model or 'qwen' in rule.target_model or 'deepseek' in rule.target_model else 'Cloud'}
                            </span>
                        </td>
                        <td style="color: #28a745; font-weight: 600;">${rule.estimated_savings:,.2f}</td>
                        <td>
                            <div class="progress">
                                <div class="progress-bar" style="width: {rule.confidence * 100}%; background: {'#28a745' if rule.confidence > 0.8 else '#ffc107' if rule.confidence > 0.6 else '#dc3545'};"></div>
                            </div>
                            {rule.confidence * 100:.0f}%
                        </td>
                    </tr>
                    """
            for rule in recommendation.rules[:10]
        )
    }
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>🌳 Decision Tree</h2>
            <pre style="background: #f8f9fa; padding: 20px; border-radius: 8px; overflow-x: auto;">
{ascii_decision_tree(recommendation.decision_tree)}
            </pre>
        </div>

        <div class="card">
            <h2>💾 Implementation Config</h2>
            <pre style="background: #1a1a2e; color: #a8ff60; padding: 20px; border-radius: 8px; overflow-x: auto;">
{json.dumps(recommendation.implementation_config, indent=2)}
            </pre>
        </div>
    </div>

    <script>
        // Model cost pie chart
        new Chart(document.getElementById('modelChart'), {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(list(analysis.cost_by_model.keys()))},
                datasets: [{{
                    data: {json.dumps(list(analysis.cost_by_model.values()))},
                    backgroundColor: [
                        '#667eea', '#764ba2', '#f093fb', '#f5576c',
                        '#4facfe', '#43e97b', '#fa709a', '#fee140'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'right' }}
                }}
            }}
        }});

        // Before/After comparison
        new Chart(document.getElementById('comparisonChart'), {{
            type: 'bar',
            data: {{
                labels: ['Current Cost', 'Optimized Cost', 'Savings'],
                datasets: [{{
                    data: [{analysis.total_cost:.2f}, {recommendation.optimized_cost:.2f}, {
        recommendation.monthly_savings:.2f}],
                    backgroundColor: ['#dc3545', '#28a745', '#17a2b8']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{ beginAtZero: true, ticks: {{ callback: v => '$' + v }} }}
                }}
            }}
        }});
    </script>
</body>
</html>'''

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)

    return output_path


# =============================================================================
# MATPLOTLIB VISUALIZATION (Optional)
# =============================================================================


def generate_matplotlib_charts(
    analysis: UsageAnalysis, recommendation: RoutingRecommendation, output_dir: Path
) -> list[Path]:
    """Generate matplotlib charts (if available)"""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping chart generation")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    # 1. Cost by model pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = list(analysis.cost_by_model.keys())
    values = list(analysis.cost_by_model.values())
    colors = plt.cm.Set3(range(len(labels)))

    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%", colors=colors, explode=[0.02] * len(labels)
    )
    ax.set_title("Cost Distribution by Model", fontsize=14, fontweight="bold")

    path = output_dir / "cost_by_model.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    generated.append(path)

    # 2. Before/After comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Current", "Optimized", "Savings"]
    values = [analysis.total_cost, recommendation.optimized_cost, recommendation.monthly_savings]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=2)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"${val:,.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Cost ($)", fontsize=12)
    ax.set_title("Cost Optimization Impact", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = output_dir / "cost_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    generated.append(path)

    # 3. Daily cost trend
    if analysis.daily_costs:
        fig, ax = plt.subplots(figsize=(12, 5))

        dates = sorted(analysis.daily_costs.keys())
        costs = [analysis.daily_costs[d] for d in dates]

        ax.fill_between(range(len(dates)), costs, alpha=0.3, color="#667eea")
        ax.plot(range(len(dates)), costs, color="#667eea", linewidth=2, marker="o", markersize=4)

        # Show every nth label
        n = max(1, len(dates) // 10)
        ax.set_xticks(range(0, len(dates), n))
        ax.set_xticklabels([dates[i] for i in range(0, len(dates), n)], rotation=45)

        ax.set_ylabel("Daily Cost ($)", fontsize=12)
        ax.set_title("Daily Cost Trend", fontsize=14, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        path = output_dir / "daily_trend.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        generated.append(path)

    # 4. Task distribution
    if analysis.task_distribution:
        fig, ax = plt.subplots(figsize=(10, 6))

        tasks = list(analysis.task_distribution.keys())
        counts = list(analysis.task_distribution.values())

        bars = ax.barh(
            tasks, counts, color=plt.cm.viridis([i / len(tasks) for i in range(len(tasks))])
        )

        ax.set_xlabel("Number of Requests", fontsize=12)
        ax.set_title("Task Type Distribution", fontsize=14, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        path = output_dir / "task_distribution.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        generated.append(path)

    return generated


# =============================================================================
# MARKDOWN REPORT
# =============================================================================


def generate_markdown_report(analysis: UsageAnalysis, recommendation: RoutingRecommendation) -> str:
    """Generate markdown format report"""

    md = f"""# LLM Cost Optimization Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Analysis Period:** {analysis.start_date.strftime("%Y-%m-%d")} to {analysis.end_date.strftime("%Y-%m-%d")}

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| Total Requests | {analysis.total_requests:,} |
| Total Tokens | {analysis.total_tokens:,} |
| Current Cost | ${analysis.total_cost:,.2f} |
| Optimized Cost | ${recommendation.optimized_cost:,.2f} |
| **Monthly Savings** | **${recommendation.monthly_savings:,.2f}** |
| **Annual Savings** | **${recommendation.annual_savings:,.2f}** |
| Savings Percentage | {recommendation.savings_percentage:.1f}% |

---

## 💰 Current Cost Breakdown

| Model | Cost | Percentage | Requests |
|-------|------|------------|----------|
"""

    for model in sorted(analysis.cost_by_model.keys(), key=lambda m: -analysis.cost_by_model[m]):
        cost = analysis.cost_by_model[model]
        pct = (cost / analysis.total_cost) * 100
        reqs = analysis.requests_by_model.get(model, 0)
        md += f"| {model} | ${cost:,.2f} | {pct:.1f}% | {reqs:,} |\n"

    md += """
---

## 🎯 Routing Recommendations

"""

    for i, rule in enumerate(recommendation.rules[:10], 1):
        model_type = (
            "🏠 Local"
            if rule.target_model in ["llama-3.1-8b", "qwen2.5-7b", "deepseek-coder-6.7b"]
            else "☁️ Cloud"
        )
        md += f"""### {i}. {rule.task_type.title()} Tasks

- **Current:** {rule.source_model}
- **Recommended:** {rule.target_model} ({model_type})
- **Monthly Savings:** ${rule.estimated_savings:,.2f}
- **Confidence:** {rule.confidence * 100:.0f}%

"""

    md += f"""
---

## 🌳 Decision Tree

```
{ascii_decision_tree(recommendation.decision_tree)}
```

---

## 💾 Implementation Config

```json
{json.dumps(recommendation.implementation_config, indent=2)}
```

---

## 📈 Next Steps

1. **Set up local models** - Deploy Ollama with recommended models
2. **Implement routing logic** - Use the config above in your gateway
3. **Monitor performance** - Track quality metrics after switching
4. **Iterate** - Adjust routing rules based on real-world performance

---

*Generated by [LLM Cost Optimizer](https://github.com/tommieseals/llm-cost-optimizer)*
"""

    return md


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    from .analyzer import UsageAnalyzer
    from .optimizer import RoutingOptimizer

    if len(sys.argv) < 2:
        print("Usage: python visualizer.py <usage_log_file> [output_dir]")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./output")

    # Load and analyze
    analyzer = UsageAnalyzer()
    analyzer.load(filepath)
    analysis = analyzer.analyze()

    # Optimize
    optimizer = RoutingOptimizer()
    recommendation = optimizer.optimize(analysis)

    # Generate visualizations
    print("Generating ASCII charts...")
    print(ascii_bar_chart(analysis.cost_by_model, "Cost by Model"))
    print(ascii_pie_chart(analysis.task_distribution, "Task Distribution"))

    print("\nGenerating HTML report...")
    html_path = generate_html_report(analysis, recommendation, output_dir / "report.html")
    print(f"  ✓ {html_path}")

    print("\nGenerating Markdown report...")
    md_report = generate_markdown_report(analysis, recommendation)
    md_path = output_dir / "report.md"
    md_path.write_text(md_report)
    print(f"  ✓ {md_path}")

    if HAS_MATPLOTLIB:
        print("\nGenerating charts...")
        charts = generate_matplotlib_charts(analysis, recommendation, output_dir / "charts")
        for chart in charts:
            print(f"  ✓ {chart}")

    print(f"\n✨ All reports generated in {output_dir}")
