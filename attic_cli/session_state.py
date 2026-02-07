"""
Session State - Persistent state management for ATTIC CLI.

Stores session data including prompts, configs, and results.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """Record of a single pipeline run."""
    prompt: str
    config: Dict[str, Any]
    result: str
    success: bool
    timestamp: str
    execution_time: float = 0.0
    error: Optional[str] = None


@dataclass
class SessionState:
    """
    Session state container for ATTIC CLI.
    
    Maintains state across the interactive session and
    provides persistence between sessions.
    """
    run_counter: int = 0
    last_prompt: Optional[str] = None
    last_config: Optional[Dict[str, Any]] = None
    last_result: Optional[str] = None
    last_success: bool = True
    history: List[RunRecord] = field(default_factory=list)
    session_start: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Config overrides for the session
    live_data_enabled: bool = False
    learning_enabled: bool = True
    verbose: bool = False

    def __post_init__(self):
        """Initialize state directory."""
        self.state_dir = Path(__file__).parent.parent / "data" / ".attic_session"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "session_state.json"
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        try:
            if self.state_file.exists():
                data = json.loads(self.state_file.read_text())
                self.run_counter = data.get("run_counter", 0)
                # Don't load last_prompt/result across sessions
                logger.debug(f"Loaded session state: {self.run_counter} previous runs")
        except Exception as e:
            logger.warning(f"Could not load session state: {e}")

    def save_state(self) -> None:
        """Persist state to disk."""
        try:
            data = {
                "run_counter": self.run_counter,
                "session_start": self.session_start,
            }
            self.state_file.write_text(json.dumps(data, indent=2))
            logger.debug("Saved session state")
        except Exception as e:
            logger.warning(f"Could not save session state: {e}")

    def record_run(
        self,
        prompt: str,
        config: Dict[str, Any],
        result: str,
        success: bool = True,
        execution_time: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        """
        Record a pipeline run.

        Args:
            prompt: User's original prompt.
            config: Pipeline configuration used.
            result: Generated content or error message.
            success: Whether the run succeeded.
            execution_time: Time taken in seconds.
            error: Error message if failed.
        """
        self.run_counter += 1
        self.last_prompt = prompt
        self.last_config = config
        self.last_result = result
        self.last_success = success

        record = RunRecord(
            prompt=prompt,
            config=config,
            result=result,
            success=success,
            timestamp=datetime.now().isoformat(),
            execution_time=execution_time,
            error=error,
        )
        self.history.append(record)
        self.save_state()
        
        logger.info(f"Recorded run #{self.run_counter}: success={success}")

    def get_last_run(self) -> Optional[RunRecord]:
        """
        Get the last run record.

        Returns:
            Last RunRecord or None if no runs.
        """
        return self.history[-1] if self.history else None

    def get_run_history(self, limit: int = 10) -> List[RunRecord]:
        """
        Get recent run history.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of recent RunRecords.
        """
        return self.history[-limit:]

    def clear_history(self) -> None:
        """Clear the session history."""
        self.history.clear()
        self.last_prompt = None
        self.last_config = None
        self.last_result = None
        logger.info("Cleared session history")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.

        Returns:
            Dictionary of session stats.
        """
        successful = sum(1 for r in self.history if r.success)
        failed = len(self.history) - successful
        total_time = sum(r.execution_time for r in self.history)
        
        return {
            "session_start": self.session_start,
            "total_runs": self.run_counter,
            "session_runs": len(self.history),
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / len(self.history) * 100) if self.history else 0,
            "total_execution_time": total_time,
            "avg_execution_time": total_time / len(self.history) if self.history else 0,
        }

    def can_repeat(self) -> bool:
        """
        Check if there's a previous run to repeat.

        Returns:
            True if a previous run exists.
        """
        return self.last_prompt is not None and self.last_config is not None

    def get_repeat_config(self) -> Optional[Dict[str, Any]]:
        """
        Get configuration for repeating last run.

        Returns:
            Last config dict or None.
        """
        return self.last_config.copy() if self.last_config else None
