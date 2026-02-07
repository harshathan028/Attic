"""
ATTIC CLI - Adaptive Tool-driven Intelligent Content Orchestrator

A modern OpenCode-style interactive CLI for the AI Content Factory.
"""

from .banner import ATTICBanner
from .session_state import SessionState
from .command_parser import CommandParser, CLIConfig
from .rich_layout import RichLayout
from .interactive_shell import InteractiveShell

__all__ = [
    "ATTICBanner",
    "SessionState",
    "CommandParser",
    "CLIConfig",
    "RichLayout",
    "InteractiveShell",
]

__version__ = "1.0.0"
