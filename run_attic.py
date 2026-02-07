#!/usr/bin/env python3
"""
ATTIC - Adaptive Tool-driven Intelligent Content Orchestrator

OpenCode-style interactive CLI launcher for AI Content Factory.

Usage:
    python run_attic.py

This launches the interactive ATTIC shell where you can type
natural language prompts to generate content.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.WARNING
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(
            console=Console(stderr=True),
            show_path=False,
            rich_tracebacks=True,
        )],
    )


def main() -> int:
    """
    Launch ATTIC interactive shell.

    Returns:
        Exit code (0 for success).
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ATTIC - Adaptive Tool-driven Intelligent Content Orchestrator",
        epilog="Launch the interactive CLI by running without arguments.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Show version if requested
    if args.version:
        from attic_cli import __version__
        print(f"ATTIC v{__version__}")
        return 0
    
    try:
        # Import and launch the shell
        from attic_cli.interactive_shell import InteractiveShell
        
        shell = InteractiveShell(config_path=args.config)
        shell.run()
        
        return 0
        
    except KeyboardInterrupt:
        console = Console()
        console.print("\n[yellow]Interrupted[/yellow]")
        return 130
        
    except ImportError as e:
        console = Console()
        console.print(f"[red]Missing dependency:[/red] {e}")
        console.print("\nInstall with: [cyan]pip install prompt_toolkit rich[/cyan]")
        return 1
        
    except Exception as e:
        console = Console()
        console.print(f"[red]Error:[/red] {e}")
        logging.exception("ATTIC failed to start")
        return 1


if __name__ == "__main__":
    sys.exit(main())
