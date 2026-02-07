"""
ATTIC Banner - ASCII art logo and startup display.

Renders the ATTIC branding with cyber-style colors.
"""

import logging
from typing import Optional

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.style import Style

logger = logging.getLogger(__name__)


class ATTICBanner:
    """ATTIC ASCII art banner renderer."""

    # Cyber-style ASCII logo
    LOGO = r"""
    ▄▀█ ▀█▀ ▀█▀ █ █▀▀
    █▀█ ░█░ ░█░ █ █▄▄
    """

    LOGO_LARGE = r"""
     █████╗ ████████╗████████╗██╗ ██████╗
    ██╔══██╗╚══██╔══╝╚══██╔══╝██║██╔════╝
    ███████║   ██║      ██║   ██║██║     
    ██╔══██║   ██║      ██║   ██║██║     
    ██║  ██║   ██║      ██║   ██║╚██████╗
    ╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚═╝ ╚═════╝
    """

    SUBTITLE = "Adaptive Tool-driven Intelligent Content Orchestrator"
    HINT = "Ask anything..."
    TIP = "Type 'help' for commands • Ctrl+D to exit"

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the banner renderer.

        Args:
            console: Rich console instance. Creates new if not provided.
        """
        self.console = console or Console()

    def render_logo(self) -> Text:
        """
        Render the ATTIC logo with gradient colors.

        Returns:
            Rich Text object with styled logo.
        """
        logo_text = Text()
        
        # Gradient from cyan to magenta
        colors = ["bright_cyan", "cyan", "blue", "magenta", "bright_magenta"]
        
        for i, line in enumerate(self.LOGO_LARGE.split("\n")):
            if line.strip():
                color = colors[min(i, len(colors) - 1)]
                logo_text.append(line + "\n", style=Style(color=color, bold=True))
        
        return logo_text

    def render_subtitle(self) -> Text:
        """
        Render the subtitle with styling.

        Returns:
            Rich Text object with styled subtitle.
        """
        return Text(
            self.SUBTITLE,
            style=Style(color="white", dim=True),
            justify="center",
        )

    def render_hint(self) -> Text:
        """
        Render the hint text.

        Returns:
            Rich Text object with styled hint.
        """
        hint = Text()
        hint.append("\n")
        hint.append(self.HINT, style=Style(color="bright_yellow", italic=True))
        return hint

    def render_tip(self) -> Text:
        """
        Render the tip text.

        Returns:
            Rich Text object with styled tip.
        """
        return Text(
            self.TIP,
            style=Style(color="bright_black"),
            justify="center",
        )

    def render_full_banner(self) -> Panel:
        """
        Render the complete banner with all elements.

        Returns:
            Rich Panel containing the full banner.
        """
        content = Text()
        
        # Add logo
        content.append_text(self.render_logo())
        
        # Add subtitle
        content.append("\n")
        content.append(
            self.SUBTITLE,
            style=Style(color="bright_white", dim=True),
        )
        
        # Add hint
        content.append("\n")
        content.append(
            self.HINT,
            style=Style(color="bright_yellow", italic=True),
        )
        
        # Add tip
        content.append("\n\n")
        content.append(
            self.TIP,
            style=Style(color="bright_black"),
        )

        return Panel(
            Align.center(content),
            border_style="cyan",
            padding=(1, 2),
        )

    def display(self) -> None:
        """Display the full banner to console."""
        self.console.clear()
        self.console.print()
        self.console.print(Align.center(self.render_full_banner()))
        self.console.print()

    def display_mini(self) -> None:
        """Display a minimal banner for returning to prompt."""
        mini = Text()
        mini.append("ATTIC", style=Style(color="cyan", bold=True))
        mini.append(" > ", style=Style(color="bright_black"))
        
        self.console.print()
        self.console.print(Align.center(mini))

    def get_prompt_prefix(self) -> str:
        """
        Get the styled prompt prefix for prompt_toolkit.

        Returns:
            ANSI-styled prompt string.
        """
        return "\x1b[36;1mattic\x1b[0m \x1b[90m>\x1b[0m "
