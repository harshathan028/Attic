"""
Base Agent - Foundation class for all specialized agents.

This module defines the abstract base class that all agents inherit from,
providing common functionality for prompt loading, execution, and tool access.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.llm_client import LLMClient
from tools.search_tool import SearchTool
from tools.vector_store import VectorStore
from tools.evaluator import ContentEvaluator

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """
    Context object passed between agents in the pipeline.
    
    Contains the current state of the content generation process,
    including original input, intermediate outputs, and metadata.
    """
    topic: str
    research_notes: str = ""
    draft_content: str = ""
    fact_check_results: str = ""
    optimized_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_outputs: Dict[str, str] = field(default_factory=dict)

    def get_previous_output(self, agent_name: str) -> Optional[str]:
        """Get the output from a previous agent."""
        return self.agent_outputs.get(agent_name)

    def set_output(self, agent_name: str, output: str) -> None:
        """Store output from an agent."""
        self.agent_outputs[agent_name] = output


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    role: str
    prompt_file: str
    tools: List[str] = field(default_factory=list)
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 2000


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents.
    
    Provides common functionality including:
    - Prompt template loading
    - Tool access management
    - Context input/output handling
    - Execution lifecycle hooks
    """

    def __init__(
        self,
        name: str,
        role: str,
        llm_client: LLMClient,
        prompt_file: Optional[str] = None,
        prompts_dir: str = "prompts",
        tools: Optional[Dict[str, Any]] = None,
        config: Optional[AgentConfig] = None,
    ):
        """
        Initialize the base agent.

        Args:
            name: Unique agent identifier.
            role: Description of the agent's role.
            llm_client: LLM client for text generation.
            prompt_file: Name of the prompt template file.
            prompts_dir: Directory containing prompt templates.
            tools: Dictionary of available tools.
            config: Optional AgentConfig object.
        """
        self.name = name
        self.role = role
        self.llm_client = llm_client
        self.prompt_file = prompt_file
        self.prompts_dir = prompts_dir
        self.tools = tools or {}
        self.config = config

        # Load prompt template
        self._prompt_template: Optional[str] = None
        if prompt_file:
            self._load_prompt_template()

        logger.info(f"Initialized agent: {name} ({role})")

    def _load_prompt_template(self) -> None:
        """Load the prompt template from file."""
        if not self.prompt_file:
            return

        # Find project root (where prompts directory is)
        current_dir = Path(__file__).parent.parent
        prompt_path = current_dir / self.prompts_dir / self.prompt_file

        try:
            if prompt_path.exists():
                self._prompt_template = prompt_path.read_text(encoding="utf-8")
                logger.debug(f"Loaded prompt template: {prompt_path}")
            else:
                logger.warning(f"Prompt file not found: {prompt_path}")
        except Exception as e:
            logger.error(f"Failed to load prompt template: {e}")

    def get_prompt(self, **kwargs) -> str:
        """
        Get the formatted prompt with variables substituted.

        Args:
            **kwargs: Variables to substitute in the template.

        Returns:
            Formatted prompt string.
        """
        if not self._prompt_template:
            return kwargs.get("default_prompt", "")

        prompt = self._prompt_template
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(value))

        return prompt

    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve.

        Returns:
            The tool instance or None if not found.
        """
        tool = self.tools.get(tool_name)
        if tool is None:
            logger.warning(f"Tool not found: {tool_name}")
        return tool

    @property
    def search_tool(self) -> Optional[SearchTool]:
        """Get the search tool if available."""
        return self.tools.get("search")

    @property
    def vector_store(self) -> Optional[VectorStore]:
        """Get the vector store if available."""
        return self.tools.get("vector_store")

    @property
    def evaluator(self) -> Optional[ContentEvaluator]:
        """Get the content evaluator if available."""
        return self.tools.get("evaluator")

    def pre_run(self, context: AgentContext) -> None:
        """
        Hook called before the main run method.
        
        Override in subclasses for pre-processing.
        """
        logger.debug(f"[{self.name}] Pre-run hook")

    def post_run(self, context: AgentContext, output: str) -> str:
        """
        Hook called after the main run method.
        
        Override in subclasses for post-processing.
        
        Args:
            context: The agent context.
            output: The raw output from run.
            
        Returns:
            Processed output.
        """
        logger.debug(f"[{self.name}] Post-run hook")
        return output

    @abstractmethod
    def run(self, context: AgentContext) -> str:
        """
        Execute the agent's main task.

        Args:
            context: The current pipeline context.

        Returns:
            The agent's output as a string.
        """
        pass

    def execute(self, context: AgentContext) -> str:
        """
        Full execution lifecycle including hooks.

        Args:
            context: The current pipeline context.

        Returns:
            The processed agent output.
        """
        logger.info(f"[{self.name}] Starting execution")

        try:
            self.pre_run(context)
            output = self.run(context)
            output = self.post_run(context, output)

            # Store output in context
            context.set_output(self.name, output)

            logger.info(f"[{self.name}] Execution complete ({len(output)} chars)")
            return output

        except Exception as e:
            logger.error(f"[{self.name}] Execution failed: {e}")
            raise

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', role='{self.role}')"
