"""
Interactive Shell - Main ATTIC CLI shell with prompt_toolkit.

Provides the interactive command-line interface with history,
auto-completion, and keyboard shortcuts.
"""

import logging
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console

from .banner import ATTICBanner
from .command_parser import CommandParser, CLIConfig
from .rich_layout import RichLayout
from .session_state import SessionState

logger = logging.getLogger(__name__)


class InteractiveShell:
    """
    ATTIC Interactive Shell - OpenCode-style CLI interface.
    
    Provides a full-screen interactive experience with:
    - Command history and navigation
    - Natural language prompt parsing
    - Live pipeline execution display
    - Rich result rendering
    """

    # Built-in commands
    BUILTIN_COMMANDS = {
        "help": "Show help information",
        "history": "Show command history",
        "repeat": "Repeat the last command",
        "stats": "Show session statistics",
        "config": "Show current configuration",
        "clear": "Clear the screen",
        "exit": "Exit ATTIC",
        "quit": "Exit ATTIC",
    }

    def __init__(
        self,
        pipeline_runner: Optional[Callable] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the interactive shell.

        Args:
            pipeline_runner: Callback function to run the pipeline.
            config_path: Path to pipeline config file.
        """
        self.console = Console()
        self.banner = ATTICBanner(self.console)
        self.layout = RichLayout(self.console)
        self.parser = CommandParser()
        self.state = SessionState()
        
        self.pipeline_runner = pipeline_runner
        self.config_path = config_path
        self._pipeline = None
        
        # Setup prompt session
        history_file = Path(__file__).parent.parent / "data" / ".attic_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.session = PromptSession(
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            enable_history_search=True,
            style=self._get_prompt_style(),
            key_bindings=self._get_key_bindings(),
        )
        
        self.running = True
        logger.info("Initialized ATTIC Interactive Shell")

    def _get_prompt_style(self) -> PTStyle:
        """Get prompt_toolkit style."""
        return PTStyle.from_dict({
            "prompt": "cyan bold",
            "": "white",
        })

    def _get_key_bindings(self) -> KeyBindings:
        """Configure key bindings."""
        kb = KeyBindings()
        
        @kb.add("c-d")
        def exit_handler(event):
            """Handle Ctrl+D for exit."""
            self.running = False
            event.app.exit()
        
        @kb.add("c-c")
        def cancel_handler(event):
            """Handle Ctrl+C for cancel."""
            event.current_buffer.reset()
        
        return kb

    def _init_pipeline(self) -> None:
        """Initialize the pipeline lazily."""
        if self._pipeline is None:
            try:
                # Import here to avoid circular imports
                from orchestrator.pipeline import ContentPipeline
                
                self._pipeline = ContentPipeline(
                    config_path=self.config_path,
                    live_data_enabled=False,  # Set per-run
                    learning_enabled=self.state.learning_enabled,
                )
                logger.info("Initialized ContentPipeline for CLI")
            except Exception as e:
                logger.error(f"Failed to initialize pipeline: {e}")
                raise

    def run(self) -> None:
        """Run the interactive shell main loop."""
        # Display banner
        self.banner.display()
        
        try:
            while self.running:
                try:
                    # Get user input
                    prompt_text = self.banner.get_prompt_prefix()
                    user_input = self.session.prompt(prompt_text)
                    
                    if not user_input or not user_input.strip():
                        continue
                    
                    # Process the input
                    self._handle_input(user_input.strip())
                    
                except KeyboardInterrupt:
                    self.console.print("\n[dim]Use 'exit' or Ctrl+D to quit[/dim]")
                    continue
                except EOFError:
                    self.running = False
                    break
        
        finally:
            self._cleanup()
            self.console.print("\n[cyan]Goodbye! 👋[/cyan]\n")

    def _handle_input(self, user_input: str) -> None:
        """
        Handle user input.

        Args:
            user_input: User's command or prompt.
        """
        input_lower = user_input.lower()
        
        # Check for built-in commands
        if input_lower in self.BUILTIN_COMMANDS:
            self._handle_builtin(input_lower)
            return
        
        # Parse as natural language prompt
        self._run_pipeline(user_input)

    def _handle_builtin(self, command: str) -> None:
        """
        Handle built-in commands.

        Args:
            command: Built-in command name.
        """
        if command == "help":
            self.layout.show_help()
            
        elif command == "history":
            self.layout.show_history(self.state.get_run_history())
            
        elif command == "repeat":
            if self.state.can_repeat():
                self.console.print(f"[dim]Repeating: {self.state.last_prompt}[/dim]")
                self._run_pipeline(self.state.last_prompt)
            else:
                self.layout.show_warning("No previous command to repeat")
                
        elif command == "stats":
            # Get both session and pipeline stats
            session_stats = self.state.get_stats()
            
            try:
                from memory import RunHistoryStore
                run_history = RunHistoryStore()
                pipeline_stats = run_history.get_statistics()
                
                combined = {
                    "Session Runs": session_stats["session_runs"],
                    "Session Success Rate": f"{session_stats['success_rate']:.1f}%",
                    "Total Pipeline Runs": pipeline_stats.get("total_runs", 0),
                    "Avg Pipeline Score": f"{pipeline_stats.get('avg_score', 0):.1f}",
                    "Best Score": f"{pipeline_stats.get('max_score', 0):.1f}",
                }
                self.layout.show_stats(combined)
            except Exception:
                self.layout.show_stats(session_stats)
                
        elif command == "config":
            config = {
                "Live Data": "Auto-detect",
                "Learning": "Enabled" if self.state.learning_enabled else "Disabled",
                "Verbose": self.state.verbose,
                "Run Counter": self.state.run_counter,
            }
            self.layout.show_config(config)
            
        elif command == "clear":
            self.banner.display()
            
        elif command in ("exit", "quit"):
            self.running = False

    def _run_pipeline(self, prompt: str) -> None:
        """
        Run the pipeline with a natural language prompt.

        Args:
            prompt: User's natural language prompt.
        """
        start_time = time.time()
        
        try:
            # Parse the prompt
            config = self.parser.parse(prompt)
            
            # Validate
            valid, error = self.parser.validate_config(config)
            if not valid:
                self.layout.show_error(error, "Invalid Input")
                return
            
            # Show parsed config
            self.layout.show_parsing_result({
                "topic": config.topic,
                "live_data_enabled": config.live_data_enabled,
                "data_source_url": config.data_source_url,
                "data_file": config.data_file,
                "api_endpoint": config.api_endpoint,
                "detected_intent": config.detected_intent,
            })
            
            # Initialize pipeline if needed
            self._init_pipeline()
            
            # Get agent list
            agents = self._get_agent_list(config)
            
            # Show progress
            self.layout.start_live_display(agents)
            
            # Update initial status
            for agent in agents:
                self.layout.update_agent_status(agent, "waiting", 0)
            
            # Run pipeline with progress updates
            result = self._execute_with_progress(config, agents)
            
            # Stop progress display
            self.layout.stop_live_display()
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Record the run
            self.state.record_run(
                prompt=prompt,
                config=config.to_dict(),
                result=result,
                success=True,
                execution_time=execution_time,
            )
            
            # Show result
            self.layout.show_result(result)
            self.layout.show_success(f"Completed in {execution_time:.1f}s")
            
        except Exception as e:
            # Stop progress if running
            self.layout.stop_live_display()
            
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            # Record failed run
            self.state.record_run(
                prompt=prompt,
                config=config.to_dict() if 'config' in dir() else {},
                result="",
                success=False,
                execution_time=execution_time,
                error=error_msg,
            )
            
            self.layout.show_error(error_msg, "Pipeline Error")
            logger.exception("Pipeline execution failed")

    def _get_agent_list(self, config: CLIConfig) -> List[str]:
        """
        Get the list of agents that will run.

        Args:
            config: CLI configuration.

        Returns:
            List of agent names.
        """
        agents = []
        
        if config.live_data_enabled:
            agents.append("live_data_agent")
        
        agents.extend([
            "research_agent",
            "writer_agent",
            "factcheck_agent",
            "optimizer_agent",
        ])
        
        return agents

    def _execute_with_progress(
        self,
        config: CLIConfig,
        agents: List[str],
    ) -> str:
        """
        Execute pipeline with progress updates using background thread.

        The pipeline runs in a separate thread while the main thread
        polls for progress updates and refreshes the display.

        Args:
            config: CLI configuration.
            agents: List of agents to track.

        Returns:
            Generated content.
        """
        # Queue for thread-safe communication
        progress_queue: queue.Queue = queue.Queue()
        result_holder: Dict[str, Any] = {"result": None, "error": None}
        
        # Configure pipeline for this run
        self._pipeline.live_data_enabled = config.live_data_enabled
        
        def progress_callback(agent_name: str, status: str) -> None:
            """Thread-safe progress callback - puts updates in queue."""
            progress_queue.put((agent_name, status))
        
        def run_pipeline_thread():
            """Run pipeline in background thread."""
            try:
                result = self._pipeline.run_from_cli_config(
                    topic=config.topic,
                    data_source_url=config.data_source_url,
                    data_file=config.data_file,
                    api_endpoint=config.api_endpoint,
                    live_data_enabled=config.live_data_enabled,
                    progress_callback=progress_callback,
                )
                result_holder["result"] = result
            except Exception as e:
                result_holder["error"] = e
            finally:
                # Signal completion
                progress_queue.put(("__DONE__", "done"))
        
        # Start pipeline in background thread
        pipeline_thread = threading.Thread(target=run_pipeline_thread, daemon=True)
        pipeline_thread.start()
        
        # Mark first agent as running
        current_agent_idx = 0
        if agents:
            self.layout.update_agent_status(agents[0], "running", 50)
        
        # Poll for updates while pipeline runs
        done = False
        while not done:
            try:
                # Check for progress updates (non-blocking with timeout)
                agent_name, status = progress_queue.get(timeout=0.1)
                
                if agent_name == "__DONE__":
                    done = True
                elif status == "start":
                    # Mark current agent as running
                    self.layout.update_agent_status(agent_name, "running", 50)
                elif status == "complete":
                    # Mark agent as complete
                    self.layout.update_agent_status(agent_name, "complete", 100)
                    current_agent_idx += 1
                    # Mark next agent as running
                    if current_agent_idx < len(agents):
                        self.layout.update_agent_status(
                            agents[current_agent_idx], "running", 50
                        )
                elif status == "error":
                    self.layout.update_agent_status(agent_name, "error", 100)
                    
            except queue.Empty:
                # No updates yet, just refresh the display
                pass
            
            # Force display refresh to show spinner animation
            if self.layout._live:
                self.layout._live.refresh()
        
        # Wait for thread to fully complete
        pipeline_thread.join(timeout=5.0)
        
        # Check for errors
        if result_holder["error"]:
            raise result_holder["error"]
        
        # Mark all agents as complete (in case any were missed)
        for agent in agents:
            self.layout.update_agent_status(agent, "complete", 100)
        
        return result_holder["result"] or ""

    def _cleanup(self) -> None:
        """Cleanup resources on exit."""
        self.state.save_state()
        logger.info("ATTIC shell cleanup complete")


def create_shell(
    pipeline_runner: Optional[Callable] = None,
    config_path: Optional[str] = None,
) -> InteractiveShell:
    """
    Factory function to create an interactive shell.

    Args:
        pipeline_runner: Optional pipeline runner callback.
        config_path: Optional config file path.

    Returns:
        Configured InteractiveShell instance.
    """
    return InteractiveShell(
        pipeline_runner=pipeline_runner,
        config_path=config_path,
    )
