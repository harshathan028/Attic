#!/usr/bin/env python3
"""
AI Content Factory - Custom Multi-Agent Orchestrator

A production-ready content generation pipeline with:
- Live data ingestion (RSS, API, PDF, CSV)
- Long-term learning memory
- Multi-agent orchestration
- Quality evaluation and optimization
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.pipeline import ContentPipeline

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(
            console=console,
            show_path=True,
            rich_tracebacks=True,
        )],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Content Factory - Multi-Agent Content Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --topic "AI in Healthcare"
  
  # With live data from RSS feed
  python main.py --topic "Climate Policy" --live-data --data-source-url "https://news.google.com/rss"
  
  # With CSV data file
  python main.py --topic "Sales Analysis" --live-data --data-file data.csv
  
  # With API endpoint
  python main.py --topic "Weather Report" --live-data --api-endpoint "https://api.example.com/data"
  
  # Disable learning
  python main.py --topic "Quick Test" --no-learning
        """
    )
    
    # Required arguments (unless --show-stats is used)
    parser.add_argument(
        "--topic",
        type=str,
        help="Topic to generate content about",
    )
    
    # Live data options
    parser.add_argument(
        "--live-data",
        action="store_true",
        help="Enable live data ingestion mode",
    )
    parser.add_argument(
        "--data-source-url",
        type=str,
        help="URL for live data source (RSS feed, API, PDF URL)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        help="Local file path for data (CSV, PDF)",
    )
    parser.add_argument(
        "--api-endpoint",
        type=str,
        help="REST API endpoint for data",
    )
    
    # Learning options
    parser.add_argument(
        "--no-learning",
        action="store_true",
        help="Disable learning memory",
    )
    
    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (default: print to console)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    # Config options
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file",
    )
    
    # Stats options
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show learning statistics and exit",
    )
    
    return parser.parse_args()


def show_stats() -> None:
    """Display learning statistics."""
    from memory import RunHistoryStore, StrategyMemory, PerformanceTracker
    
    console.print("\n[bold blue]📊 Learning Statistics[/bold blue]\n")
    
    # Run history stats
    run_history = RunHistoryStore()
    run_stats = run_history.get_statistics()
    
    table = Table(title="Run History")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Runs", str(run_stats.get("total_runs", 0)))
    table.add_row("Average Score", f"{run_stats.get('avg_score', 0):.1f}")
    table.add_row("Best Score", f"{run_stats.get('max_score', 0):.1f}")
    table.add_row("Unique Topics", str(run_stats.get("unique_topics", 0)))
    table.add_row("Avg Execution Time", f"{run_stats.get('avg_execution_time', 0):.1f}s")
    
    console.print(table)
    
    # Strategy stats
    strategy_memory = StrategyMemory()
    strategies = strategy_memory.get_all_strategies(min_score=0)
    
    table = Table(title="\nLearned Strategies")
    table.add_column("Strategy ID", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Avg Score", style="green")
    table.add_column("Uses", style="magenta")
    
    for s in strategies[:10]:
        table.add_row(
            s.strategy_id[:20] + "..." if len(s.strategy_id) > 20 else s.strategy_id,
            s.topic_category,
            f"{s.average_score:.1f}",
            str(s.use_count),
        )
    
    console.print(table)
    
    # Performance statsistics
    performance = PerformanceTracker()
    perf_stats = performance.get_overall_stats()
    
    table = Table(title="\nPerformance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Measurements", str(perf_stats.get("total_measurements", 0)))
    table.add_row("Success Rate", f"{perf_stats.get('success_rate', 0):.1f}%")
    table.add_row("Overall Avg Score", f"{perf_stats.get('overall_avg_score', 0):.1f}")
    table.add_row("Total Failures", str(perf_stats.get("total_failures", 0)))
    
    console.print(table)


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    # Show header
    console.print(Panel.fit(
        "[bold blue]AI Content Factory[/bold blue]\n"
        "[dim]Custom Multi-Agent Orchestrator[/dim]",
        border_style="blue",
    ))
    
    # Show stats if requested
    if args.show_stats:
        show_stats()
        return 0
    
    # Topic is required for content generation
    if not args.topic:
        console.print("[bold red]Error:[/bold red] --topic is required for content generation")
        console.print("Use --show-stats to view learning statistics without generating content")
        return 1
    
    console.print(f"\n[bold]Topic:[/bold] {args.topic}")
    
    # Show mode info
    if args.live_data:
        console.print("[green]📡 Live Data Mode: ENABLED[/green]")
        if args.data_source_url:
            console.print(f"   Source URL: {args.data_source_url}")
        if args.data_file:
            console.print(f"   Data File: {args.data_file}")
        if args.api_endpoint:
            console.print(f"   API Endpoint: {args.api_endpoint}")
    
    if not args.no_learning:
        console.print("[cyan]🧠 Learning Memory: ENABLED[/cyan]")
    
    console.print()
    
    try:
        # Initialize pipeline
        console.print("[dim]Initializing pipeline...[/dim]")
        pipeline = ContentPipeline(
            config_path=args.config,
            live_data_enabled=args.live_data,
            learning_enabled=not args.no_learning,
        )
        
        # Execute pipeline
        console.print("[dim]Running agent pipeline...[/dim]\n")
        
        with console.status("[bold green]Agents working...", spinner="dots"):
            result = pipeline.run(
                topic=args.topic,
                data_source_url=args.data_source_url,
                data_file=args.data_file,
                api_endpoint=args.api_endpoint,
            )
        
        # Get summary
        summary = pipeline.get_execution_summary()
        
        console.print("\n" + "=" * 60)
        console.print("[bold green]✓ Content Generation Complete[/bold green]")
        console.print(f"\nAgents executed: {summary.get('agents_executed', 0)}/{summary.get('total_agents', 0)}")
        console.print(f"Total time: {summary.get('total_time', 0):.2f}s")
        console.print(f"Run ID: {summary.get('run_id', 'N/A')}")
        
        # Show learning stats
        if not args.no_learning:
            learning_stats = pipeline.get_learning_stats()
            if learning_stats.get("enabled"):
                run_history = learning_stats.get("run_history", {})
                console.print(f"\n[dim]Learning: {run_history.get('total_runs', 0)} runs recorded, "
                            f"avg score: {run_history.get('avg_score', 0):.1f}[/dim]")
        
        # Output result
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(result)
            console.print(f"\n[green]Output saved to: {output_path}[/green]")
        else:
            console.print(Panel(
                result,
                title="Generated Article",
                border_style="green",
                padding=(1, 2),
            ))
        
        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        logging.exception("Pipeline failed")
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
