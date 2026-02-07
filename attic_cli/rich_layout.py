"""
Rich Layout - UI components for ATTIC CLI.

Provides styled panels, progress displays, and result rendering.
"""

import logging
from typing import Any, Dict, List, Optional

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.style import Style
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)


class RichLayout:
    """
    Rich-based UI layout manager for ATTIC CLI.
    
    Handles all visual rendering including progress,
    results, errors, and status displays.
    """

    # Agent display names and icons
    AGENT_DISPLAY = {
        "live_data_agent": ("📡", "Live Data Agent", "cyan"),
        "research_agent": ("🔍", "Research Agent", "blue"),
        "writer_agent": ("✍️", "Writer Agent", "green"),
        "factcheck_agent": ("✓", "Fact Check Agent", "yellow"),
        "optimizer_agent": ("⚡", "Optimizer Agent", "magenta"),
    }

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the layout manager.

        Args:
            console: Rich console instance.
        """
        self.console = console or Console()
        self._live: Optional[Live] = None
        self._progress: Optional[Progress] = None
        self._tasks: Dict[str, Any] = {}

    def show_parsing_result(self, config: Dict[str, Any]) -> None:
        """
        Display parsed configuration.

        Args:
            config: Parsed CLI config dictionary.
        """
        table = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
            expand=False,
        )
        table.add_column("Field", style="bright_black")
        table.add_column("Value", style="bright_white")

        table.add_row("📌 Topic", config.get("topic", ""))
        
        if config.get("live_data_enabled"):
            table.add_row("📡 Live Data", "Enabled")
            
            if config.get("data_source_url"):
                table.add_row("   URL", config["data_source_url"][:50] + "...")
            if config.get("data_file"):
                table.add_row("   File", config["data_file"])
            if config.get("api_endpoint"):
                table.add_row("   API", config["api_endpoint"][:50] + "...")
        
        if config.get("detected_intent"):
            table.add_row("🎯 Intent", config["detected_intent"].title())

        panel = Panel(
            table,
            title="[bold cyan]Parsed Configuration[/bold cyan]",
            border_style="dim cyan",
            padding=(0, 1),
        )
        
        self.console.print()
        self.console.print(panel)
        self.console.print()

    def create_progress_display(self, agents: List[str]) -> Progress:
        """
        Create a progress display for agent execution.

        Args:
            agents: List of agent names to track.

        Returns:
            Progress instance.
        """
        self._progress = Progress(
            SpinnerColumn("dots"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=20),
            TaskProgressColumn(),
            console=self.console,
            expand=False,
        )
        
        # Add tasks for each agent
        for agent_name in agents:
            icon, display_name, color = self.AGENT_DISPLAY.get(
                agent_name,
                ("●", agent_name.replace("_", " ").title(), "white")
            )
            
            task_id = self._progress.add_task(
                f"{icon} {display_name}",
                total=100,
                visible=True,
            )
            self._tasks[agent_name] = task_id
        
        return self._progress

    def start_live_display(self, agents: List[str]) -> None:
        """
        Start live progress display.

        Args:
            agents: List of agents to display.
        """
        progress = self.create_progress_display(agents)
        
        panel = Panel(
            progress,
            title="[bold cyan]🚀 Pipeline Execution[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
        
        self._live = Live(
            panel,
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()

    def update_agent_status(
        self,
        agent_name: str,
        status: str = "running",
        progress: int = 50,
    ) -> None:
        """
        Update an agent's status in the display.

        Args:
            agent_name: Name of the agent.
            status: Status string ('waiting', 'running', 'complete', 'error').
            progress: Progress percentage (0-100).
        """
        if not self._progress or agent_name not in self._tasks:
            return
        
        task_id = self._tasks[agent_name]
        icon, display_name, color = self.AGENT_DISPLAY.get(
            agent_name,
            ("●", agent_name.replace("_", " ").title(), "white")
        )
        
        status_icons = {
            "waiting": ("○", "dim"),
            "running": ("●", color),
            "complete": ("✓", "green"),
            "error": ("✗", "red"),
        }
        
        status_icon, status_color = status_icons.get(status, ("●", "white"))
        
        self._progress.update(
            task_id,
            description=f"[{status_color}]{status_icon}[/{status_color}] {display_name}",
            completed=progress,
        )

    def stop_live_display(self) -> None:
        """Stop the live progress display."""
        if self._live:
            self._live.stop()
            self._live = None
        self._progress = None
        self._tasks.clear()

    def show_result(self, content: str, title: str = "Generated Content") -> None:
        """
        Display the generated content result.

        Args:
            content: Generated content string.
            title: Panel title.
        """
        # Try to render as markdown
        try:
            rendered = Markdown(content)
        except Exception:
            rendered = Text(content)
        
        panel = Panel(
            rendered,
            title=f"[bold green]📄 {title}[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
        
        self.console.print()
        self.console.print(panel)
        self.console.print()

    def show_error(self, message: str, title: str = "Error") -> None:
        """
        Display an error message.

        Args:
            message: Error message.
            title: Panel title.
        """
        panel = Panel(
            Text(message, style="red"),
            title=f"[bold red]❌ {title}[/bold red]",
            border_style="red",
            padding=(1, 2),
        )
        
        self.console.print()
        self.console.print(panel)
        self.console.print()

    def show_warning(self, message: str) -> None:
        """
        Display a warning message.

        Args:
            message: Warning message.
        """
        self.console.print(f"[yellow]⚠ {message}[/yellow]")

    def show_info(self, message: str) -> None:
        """
        Display an info message.

        Args:
            message: Info message.
        """
        self.console.print(f"[cyan]ℹ {message}[/cyan]")

    def show_success(self, message: str) -> None:
        """
        Display a success message.

        Args:
            message: Success message.
        """
        self.console.print(f"[green]✓ {message}[/green]")

    def show_stats(self, stats: Dict[str, Any]) -> None:
        """
        Display session/pipeline statistics.

        Args:
            stats: Statistics dictionary.
        """
        table = Table(
            title="📊 Statistics",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bright_white", justify="right")
        
        for key, value in stats.items():
            display_key = key.replace("_", " ").title()
            
            if isinstance(value, float):
                display_value = f"{value:.2f}"
            else:
                display_value = str(value)
            
            table.add_row(display_key, display_value)
        
        self.console.print()
        self.console.print(table)
        self.console.print()

    def show_history(self, history: List[Any]) -> None:
        """
        Display command history.

        Args:
            history: List of history records.
        """
        if not history:
            self.show_info("No history yet.")
            return
        
        table = Table(
            title="📜 Recent History",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Prompt", style="white", max_width=50)
        table.add_column("Status", style="white", width=8)
        table.add_column("Time", style="dim", width=8)
        
        for i, record in enumerate(reversed(history), 1):
            status = "[green]✓[/green]" if record.success else "[red]✗[/red]"
            time_str = f"{record.execution_time:.1f}s"
            prompt_short = record.prompt[:47] + "..." if len(record.prompt) > 50 else record.prompt
            
            table.add_row(str(i), prompt_short, status, time_str)
        
        self.console.print()
        self.console.print(table)
        self.console.print()

    def show_config(self, config: Dict[str, Any]) -> None:
        """
        Display current configuration.

        Args:
            config: Configuration dictionary.
        """
        table = Table(
            title="⚙️ Current Configuration",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="bright_white")
        
        for key, value in config.items():
            display_key = key.replace("_", " ").title()
            table.add_row(display_key, str(value))
        
        self.console.print()
        self.console.print(table)
        self.console.print()

    def show_help(self) -> None:
        """Display help information."""
        help_text = """
[bold cyan]ATTIC Commands[/bold cyan]

[bold]Built-in Commands:[/bold]
  [cyan]help[/cyan]      Show this help message
  [cyan]history[/cyan]   Show recent command history
  [cyan]repeat[/cyan]    Repeat the last command
  [cyan]stats[/cyan]     Show session statistics
  [cyan]config[/cyan]    Show current configuration
  [cyan]clear[/cyan]     Clear the screen
  [cyan]exit[/cyan]      Exit ATTIC

[bold]Natural Language Prompts:[/bold]
  Just type naturally! ATTIC understands:
  
  [yellow]•[/yellow] summarize latest ai chip news
  [yellow]•[/yellow] analyze this csv sales.csv
  [yellow]•[/yellow] research climate policy from rss
  [yellow]•[/yellow] write report from pdf market.pdf
  [yellow]•[/yellow] explain quantum computing

[bold]Data Sources:[/bold]
  [yellow]•[/yellow] RSS feeds: Add "from rss" or include a feed URL
  [yellow]•[/yellow] CSV files: Reference a .csv file
  [yellow]•[/yellow] PDF docs:  Reference a .pdf file
  [yellow]•[/yellow] APIs:      Include an API endpoint URL

[bold]Keyboard Shortcuts:[/bold]
  [cyan]↑/↓[/cyan]       Navigate command history
  [cyan]Ctrl+C[/cyan]    Cancel current operation
  [cyan]Ctrl+D[/cyan]    Exit ATTIC
"""
        panel = Panel(
            help_text,
            title="[bold cyan]📚 ATTIC Help[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
        
        self.console.print()
        self.console.print(panel)
        self.console.print()

    def clear(self) -> None:
        """Clear the console."""
        self.console.clear()
