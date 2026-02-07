"""
Agent Manager - Coordinates agent registration, routing, and execution.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agents.base_agent import BaseAgent, AgentContext

logger = logging.getLogger(__name__)


@dataclass
class AgentExecutionLog:
    """Log entry for agent execution."""
    agent_name: str
    start_time: float
    end_time: float
    success: bool
    output_length: int
    error: Optional[str] = None


class AgentManager:
    """
    Manages agent registration, task routing, and execution coordination.
    
    Provides:
    - Agent registration and storage
    - Task routing to appropriate agents
    - Memory passing between agents
    - Retry logic for failures
    - Execution logging
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_order: List[str] = []
        self.execution_logs: List[AgentExecutionLog] = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info("Initialized AgentManager")

    def register_agent(self, agent: BaseAgent, order: Optional[int] = None) -> None:
        """Register an agent with optional execution order."""
        self.agents[agent.name] = agent
        if order is not None:
            while len(self.execution_order) <= order:
                self.execution_order.append(None)
            self.execution_order[order] = agent.name
        else:
            self.execution_order.append(agent.name)
        logger.info(f"Registered agent: {agent.name}")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agents.get(name)

    def set_execution_order(self, order: List[str]) -> None:
        """Set the execution order of agents."""
        for name in order:
            if name not in self.agents:
                raise ValueError(f"Unknown agent: {name}")
        self.execution_order = order
        logger.info(f"Set execution order: {order}")

    def execute_agent(self, agent_name: str, context: AgentContext) -> str:
        """Execute a single agent with retry logic."""
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent not found: {agent_name}")

        last_error = None
        for attempt in range(self.max_retries):
            start_time = time.time()
            try:
                logger.info(f"Executing {agent_name} (attempt {attempt + 1})")
                output = agent.execute(context)
                
                self.execution_logs.append(AgentExecutionLog(
                    agent_name=agent_name,
                    start_time=start_time,
                    end_time=time.time(),
                    success=True,
                    output_length=len(output),
                ))
                return output

            except Exception as e:
                last_error = e
                logger.warning(f"{agent_name} failed (attempt {attempt + 1}): {e}")
                self.execution_logs.append(AgentExecutionLog(
                    agent_name=agent_name,
                    start_time=start_time,
                    end_time=time.time(),
                    success=False,
                    output_length=0,
                    error=str(e),
                ))
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))

        raise RuntimeError(f"Agent {agent_name} failed after {self.max_retries} attempts: {last_error}")

    def execute_pipeline(
        self,
        context: AgentContext,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ) -> AgentContext:
        """
        Execute all agents in order.
        
        Args:
            context: Agent execution context.
            progress_callback: Optional callback(agent_name, status).
                status is 'start', 'complete', or 'error'.
        """
        valid_order = [n for n in self.execution_order if n]
        logger.info(f"Executing pipeline: {valid_order}")

        for agent_name in valid_order:
            # Notify start
            if progress_callback:
                try:
                    progress_callback(agent_name, "start")
                except Exception:
                    pass  # Don't let callback errors break the pipeline
            
            try:
                self.execute_agent(agent_name, context)
                
                # Notify complete
                if progress_callback:
                    try:
                        progress_callback(agent_name, "complete")
                    except Exception:
                        pass
            except Exception as e:
                # Notify error
                if progress_callback:
                    try:
                        progress_callback(agent_name, "error")
                    except Exception:
                        pass
                raise

        return context

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions."""
        total_time = sum(log.end_time - log.start_time for log in self.execution_logs)
        successes = sum(1 for log in self.execution_logs if log.success)
        
        return {
            "total_executions": len(self.execution_logs),
            "successful": successes,
            "failed": len(self.execution_logs) - successes,
            "total_time_seconds": round(total_time, 2),
            "agents": [log.agent_name for log in self.execution_logs if log.success],
        }
