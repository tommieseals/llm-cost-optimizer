#!/usr/bin/env python3
"""
LLM Cost Optimizer - Command Line Interface
=============================================
Main entry point for the optimization tool.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

from .analyzer import UsageAnalyzer, UsageAnalysis
from .optimizer import RoutingOptimizer, RoutingRecommendation, generate_optimization_report
from .visualizer import (
    ascii_bar_chart, ascii_pie_chart, ascii_comparison_chart,
    ascii_decision_tree, ascii_trend_chart,
    generate_html_report, generate_markdown_report, generate_matplotlib_charts
)


def cmd_analyze(args):
    """Analyze usage logs"""
    filepath = Path(args.file)
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return 1
    
    analyzer = UsageAnalyzer()
    count = analyzer.load(filepath, format=args.format)
    
    print(f"✓ Loaded {count} records from {filepath.name}")
    
    # Parse date filters
    start_date = datetime.fromisoformat(args.start) if args.start else None
    end_date = datetime.fromisoformat(args.end) if args.end else None
    
    analysis = analyzer.analyze(start_date=start_date, end_date=end_date)
    
    # Output
    if args.json:
        output = {
            "start_date": analysis.start_date.isoformat(),
            "end_date": analysis.end_date.isoformat(),
            "total_requests": analysis.total_requests,
            "total_tokens": analysis.total_tokens,
            "total_cost": analysis.total_cost,
            "cost_by_model": analysis.cost_by_model,
            "tokens_by_model": analysis.tokens_by_model,
            "task_distribution": analysis.task_distribution,
            "daily_costs": analysis.daily_costs,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print(f"\n{'='*60}")
        print(f"  USAGE ANALYSIS REPORT")
        print(f"{'='*60}")
        print(f"  Period: {analysis.start_date.strftime('%Y-%m-%d')} to {analysis.end_date.strftime('%Y-%m-%d')}")
        print(f"  Requests: {analysis.total_requests:,}")
        print(f"  Tokens: {analysis.total_tokens:,}")
        print(f"  Total Cost: ${analysis.total_cost:,.2f}")
        print(f"  Avg Cost/Request: ${analysis.avg_cost_per_request:.4f}")
        print(f"{'='*60}\n")
        
        print(ascii_bar_chart(analysis.cost_by_model, "Cost by Model ($)"))
        print(ascii_pie_chart(analysis.task_distribution, "\nTask Distribution"))
        
        if analysis.daily_costs:
            print(ascii_trend_chart(analysis.daily_costs, "\nDaily Cost Trend"))
        
        # Show patterns
        patterns = analyzer.get_patterns()
        if patterns["optimization_candidates"]["count"] > 0:
            print(f"\n⚠️  OPTIMIZATION OPPORTUNITY DETECTED")
            print(f"   {patterns['optimization_candidates']['count']} requests using expensive models for simple tasks")
            print(f"   Potential savings: ${patterns['optimization_candidates']['potential_savings']:.2f}")
    
    return 0


def cmd_optimize(args):
    """Generate optimization recommendations"""
    filepath = Path(args.file)
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return 1
    
    # Load and analyze
    analyzer = UsageAnalyzer()
    count = analyzer.load(filepath, format=args.format)
    analysis = analyzer.analyze()
    
    # Optimize
    optimizer = RoutingOptimizer(
        prefer_local=args.prefer_local,
        max_latency_ms=args.max_latency
    )
    recommendation = optimizer.optimize(analysis)
    
    # Output
    if args.json:
        output = {
            "current_cost": recommendation.current_cost,
            "optimized_cost": recommendation.optimized_cost,
            "monthly_savings": recommendation.monthly_savings,
            "annual_savings": recommendation.annual_savings,
            "savings_percentage": recommendation.savings_percentage,
            "rules": [
                {
                    "name": r.name,
                    "task_type": r.task_type,
                    "source_model": r.source_model,
                    "target_model": r.target_model,
                    "estimated_savings": r.estimated_savings,
                    "confidence": r.confidence,
                }
                for r in recommendation.rules
            ],
            "implementation_config": recommendation.implementation_config,
        }
        print(json.dumps(output, indent=2))
    else:
        print(generate_optimization_report(analysis, recommendation))
        
        print("\n\n" + "="*60)
        print("  DECISION TREE")
        print("="*60 + "\n")
        print(ascii_decision_tree(recommendation.decision_tree))
    
    return 0


def cmd_report(args):
    """Generate full optimization report"""
    filepath = Path(args.file)
    output_dir = Path(args.output)
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Loading {filepath.name}...")
    
    # Load and analyze
    analyzer = UsageAnalyzer()
    count = analyzer.load(filepath, format=args.format)
    analysis = analyzer.analyze()
    
    print(f"   ✓ Loaded {count} records")
    
    # Optimize
    print(f"🎯 Generating optimization recommendations...")
    optimizer = RoutingOptimizer(prefer_local=not args.cloud_only)
    recommendation = optimizer.optimize(analysis)
    
    print(f"   ✓ Found ${recommendation.monthly_savings:,.2f} in potential monthly savings")
    
    # Generate reports
    print(f"\n📝 Generating reports...")
    
    # Text report
    text_report = generate_optimization_report(analysis, recommendation)
    text_path = output_dir / "report.txt"
    text_path.write_text(text_report)
    print(f"   ✓ {text_path}")
    
    # Markdown report
    if "md" in args.formats or "all" in args.formats:
        md_report = generate_markdown_report(analysis, recommendation)
        md_path = output_dir / "report.md"
        md_path.write_text(md_report)
        print(f"   ✓ {md_path}")
    
    # HTML report
    if "html" in args.formats or "all" in args.formats:
        html_path = generate_html_report(analysis, recommendation, output_dir / "report.html")
        print(f"   ✓ {html_path}")
    
    # JSON export
    if "json" in args.formats or "all" in args.formats:
        json_data = {
            "generated_at": datetime.now().isoformat(),
            "analysis": {
                "start_date": analysis.start_date.isoformat(),
                "end_date": analysis.end_date.isoformat(),
                "total_requests": analysis.total_requests,
                "total_tokens": analysis.total_tokens,
                "total_cost": analysis.total_cost,
                "cost_by_model": analysis.cost_by_model,
                "task_distribution": analysis.task_distribution,
            },
            "optimization": {
                "current_cost": recommendation.current_cost,
                "optimized_cost": recommendation.optimized_cost,
                "monthly_savings": recommendation.monthly_savings,
                "annual_savings": recommendation.annual_savings,
                "savings_percentage": recommendation.savings_percentage,
            },
            "config": recommendation.implementation_config,
        }
        json_path = output_dir / "report.json"
        json_path.write_text(json.dumps(json_data, indent=2, default=str))
        print(f"   ✓ {json_path}")
    
    # Charts
    if args.charts:
        print(f"\n📈 Generating charts...")
        charts = generate_matplotlib_charts(analysis, recommendation, output_dir / "charts")
        for chart in charts:
            print(f"   ✓ {chart}")
    
    print(f"\n✨ All reports generated in {output_dir}/")
    print(f"\n💰 Summary: ${recommendation.monthly_savings:,.2f}/month ({recommendation.savings_percentage:.1f}% savings)")
    
    return 0


def cmd_costs(args):
    """Show cost breakdown"""
    filepath = Path(args.file)
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return 1
    
    analyzer = UsageAnalyzer()
    analyzer.load(filepath, format=args.format)
    analysis = analyzer.analyze()
    
    if args.by == "model":
        data = analysis.cost_by_model
        title = "Cost by Model"
    elif args.by == "task":
        data = analysis.cost_by_task
        title = "Cost by Task Type"
    elif args.by == "day":
        data = analysis.daily_costs
        title = "Cost by Day"
    else:
        data = analysis.cost_by_model
        title = "Cost by Model"
    
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(ascii_bar_chart(data, title))
    
    return 0


def cmd_export_config(args):
    """Export routing configuration"""
    filepath = Path(args.file)
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return 1
    
    # Load and analyze
    analyzer = UsageAnalyzer()
    analyzer.load(filepath)
    analysis = analyzer.analyze()
    
    # Optimize
    optimizer = RoutingOptimizer(prefer_local=not args.cloud_only)
    recommendation = optimizer.optimize(analysis)
    
    config = recommendation.implementation_config
    
    if args.output_format == "yaml":
        try:
            import yaml
            output = yaml.dump(config, default_flow_style=False, sort_keys=False)
        except ImportError:
            print("Warning: PyYAML not installed, using JSON", file=sys.stderr)
            output = json.dumps(config, indent=2)
    else:
        output = json.dumps(config, indent=2)
    
    if args.output:
        Path(args.output).write_text(output)
        print(f"✓ Config exported to {args.output}")
    else:
        print(output)
    
    return 0


def cmd_interactive(args):
    """Interactive mode"""
    print("🚀 LLM Cost Optimizer - Interactive Mode")
    print("=" * 50)
    print("\nCommands:")
    print("  load <file>     - Load usage logs")
    print("  analyze         - Show analysis")
    print("  optimize        - Show optimization")
    print("  report <dir>    - Generate full report")
    print("  help            - Show this help")
    print("  quit            - Exit")
    print()
    
    analyzer = None
    analysis = None
    recommendation = None
    
    while True:
        try:
            cmd = input("llm-optimize> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not cmd:
            continue
        
        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None
        
        if action in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        
        elif action == "help":
            print("\nCommands:")
            print("  load <file>     - Load usage logs")
            print("  analyze         - Show analysis")
            print("  optimize        - Show optimization")
            print("  report <dir>    - Generate full report")
            print("  costs [model|task|day] - Show cost breakdown")
            print("  help            - Show this help")
            print("  quit            - Exit")
        
        elif action == "load":
            if not arg:
                print("Usage: load <file>")
                continue
            filepath = Path(arg)
            if not filepath.exists():
                print(f"Error: File not found: {filepath}")
                continue
            analyzer = UsageAnalyzer()
            count = analyzer.load(filepath)
            print(f"✓ Loaded {count} records")
        
        elif action == "analyze":
            if not analyzer:
                print("Error: No data loaded. Use 'load <file>' first.")
                continue
            analysis = analyzer.analyze()
            print(f"\nPeriod: {analysis.start_date} to {analysis.end_date}")
            print(f"Requests: {analysis.total_requests:,}")
            print(f"Tokens: {analysis.total_tokens:,}")
            print(f"Cost: ${analysis.total_cost:,.2f}")
            print(ascii_bar_chart(analysis.cost_by_model, "\nCost by Model"))
        
        elif action == "optimize":
            if not analysis:
                print("Error: Run 'analyze' first.")
                continue
            optimizer = RoutingOptimizer()
            recommendation = optimizer.optimize(analysis)
            print(generate_optimization_report(analysis, recommendation))
        
        elif action == "costs":
            if not analysis:
                print("Error: Run 'analyze' first.")
                continue
            by = arg or "model"
            if by == "model":
                print(ascii_bar_chart(analysis.cost_by_model, "Cost by Model"))
            elif by == "task":
                print(ascii_bar_chart(analysis.cost_by_task, "Cost by Task"))
            elif by == "day":
                print(ascii_trend_chart(analysis.daily_costs, "Daily Costs"))
        
        elif action == "report":
            if not analysis or not recommendation:
                print("Error: Run 'analyze' and 'optimize' first.")
                continue
            output_dir = Path(arg or "./output")
            output_dir.mkdir(parents=True, exist_ok=True)
            generate_html_report(analysis, recommendation, output_dir / "report.html")
            md_path = output_dir / "report.md"
            md_path.write_text(generate_markdown_report(analysis, recommendation))
            print(f"✓ Reports generated in {output_dir}/")
        
        else:
            print(f"Unknown command: {action}. Type 'help' for commands.")
    
    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="LLM Cost Optimizer - Analyze usage and optimize routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llm-optimize analyze usage.json
  llm-optimize optimize usage.json --prefer-local
  llm-optimize report usage.json --output ./reports
  llm-optimize costs usage.json --by model
  llm-optimize export-config usage.json --format yaml
  llm-optimize interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze usage logs")
    analyze_parser.add_argument("file", help="Usage log file (JSON/CSV)")
    analyze_parser.add_argument("--format", "-f", choices=["auto", "openai", "anthropic", "csv", "json"],
                                default="auto", help="Log format")
    analyze_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    analyze_parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    analyze_parser.add_argument("--json", action="store_true", help="Output as JSON")
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Generate optimization recommendations")
    optimize_parser.add_argument("file", help="Usage log file")
    optimize_parser.add_argument("--format", "-f", choices=["auto", "openai", "anthropic", "csv", "json"],
                                 default="auto", help="Log format")
    optimize_parser.add_argument("--prefer-local", action="store_true", default=True,
                                 help="Prefer local models (default: true)")
    optimize_parser.add_argument("--cloud-only", action="store_true",
                                 help="Only recommend cloud models")
    optimize_parser.add_argument("--max-latency", type=int, default=5000,
                                 help="Max acceptable latency (ms)")
    optimize_parser.add_argument("--json", action="store_true", help="Output as JSON")
    optimize_parser.set_defaults(func=cmd_optimize)
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate full optimization report")
    report_parser.add_argument("file", help="Usage log file")
    report_parser.add_argument("--output", "-o", default="./output", help="Output directory")
    report_parser.add_argument("--format", "-f", choices=["auto", "openai", "anthropic", "csv", "json"],
                               default="auto", help="Log format")
    report_parser.add_argument("--formats", nargs="+", default=["all"],
                               choices=["all", "html", "md", "json", "txt"],
                               help="Report formats to generate")
    report_parser.add_argument("--charts", action="store_true", help="Generate matplotlib charts")
    report_parser.add_argument("--cloud-only", action="store_true",
                               help="Only recommend cloud models")
    report_parser.set_defaults(func=cmd_report)
    
    # Costs command
    costs_parser = subparsers.add_parser("costs", help="Show cost breakdown")
    costs_parser.add_argument("file", help="Usage log file")
    costs_parser.add_argument("--by", choices=["model", "task", "day"], default="model",
                              help="Group costs by")
    costs_parser.add_argument("--format", "-f", choices=["auto", "openai", "anthropic", "csv", "json"],
                              default="auto", help="Log format")
    costs_parser.add_argument("--json", action="store_true", help="Output as JSON")
    costs_parser.set_defaults(func=cmd_costs)
    
    # Export config command
    export_parser = subparsers.add_parser("export-config", help="Export routing configuration")
    export_parser.add_argument("file", help="Usage log file")
    export_parser.add_argument("--output", "-o", help="Output file")
    export_parser.add_argument("--output-format", choices=["json", "yaml"], default="json",
                               help="Output format")
    export_parser.add_argument("--cloud-only", action="store_true",
                               help="Only recommend cloud models")
    export_parser.set_defaults(func=cmd_export_config)
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.set_defaults(func=cmd_interactive)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
