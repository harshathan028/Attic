"""
Content Pipeline - Orchestrates the full content generation workflow.

Enhanced with live data ingestion and long-term learning memory.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from agents import (
    ResearchAgent, WriterAgent, FactCheckAgent, OptimizerAgent,
    LiveDataAgent, LiveDataConfig
)
from agents.base_agent import AgentContext
from orchestrator.agent_manager import AgentManager
from tools.llm_client import LLMClient, GeminiClient, OllamaClient, create_llm_client
from tools.search_tool import SearchTool, MockSearchTool
from tools.ddg_search import get_search_tool
from tools.vector_store import VectorStore
from tools.evaluator import ContentEvaluator
from memory.run_history_store import RunHistoryStore, RunRecord
from memory.strategy_memory import StrategyMemory, Strategy
from memory.performance_tracker import PerformanceTracker, PerformanceMetrics

logger = logging.getLogger(__name__)


class ContentPipeline:
    """
    Full content generation pipeline with learning capabilities.
    
    Workflow: 
    [live data (optional)] → research → writer → fact check → optimizer
    
    Features:
    - Live data ingestion from RSS, APIs, PDFs, CSVs
    - Long-term strategy memory for optimization
    - Performance tracking and learning
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        search_tool: Optional[SearchTool] = None,
        vector_store: Optional[VectorStore] = None,
        config_path: Optional[str] = None,
        live_data_enabled: bool = False,
        learning_enabled: bool = True,
    ):
        """
        Initialize the content pipeline.

        Args:
            llm_client: LLM client instance.
            search_tool: Search tool instance.
            vector_store: Vector store instance.
            config_path: Path to configuration file.
            live_data_enabled: Enable live data ingestion.
            learning_enabled: Enable learning memory.
        """
        self.config = self._load_config(config_path)
        self.live_data_enabled = live_data_enabled or self.config.get("live_data_enabled", False)
        self.learning_enabled = learning_enabled and self.config.get("learning_memory_enabled", True)
        
        # Generate run ID
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.run_start_time = None
        
        # Initialize LLM client based on config
        if llm_client:
            self.llm_client = llm_client
        else:
            llm_config = self.config.get("llm", {})
            provider = llm_config.get("provider", "gemini")
            model = llm_config.get("model", "gemini-2.0-flash")
            self.llm_client = create_llm_client(provider=provider, model=model)
        
        self.search_tool = search_tool or get_search_tool(prefer_real=True)
        self.vector_store = vector_store or VectorStore()
        self.evaluator = ContentEvaluator()

        # Tool dictionary for agents
        self.tools = {
            "search": self.search_tool,
            "vector_store": self.vector_store,
            "evaluator": self.evaluator,
        }

        # Initialize memory systems
        if self.learning_enabled:
            self.run_history = RunHistoryStore()
            self.strategy_memory = StrategyMemory()
            self.performance_tracker = PerformanceTracker()
        else:
            self.run_history = None
            self.strategy_memory = None
            self.performance_tracker = None

        # Initialize agent manager and agents
        self.manager = AgentManager(
            max_retries=self.config.get("pipeline", {}).get("max_retries", 3)
        )
        self._initialize_agents()
        
        # Run logging directory
        self.runs_dir = Path(__file__).parent.parent / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        
        logger.info(f"ContentPipeline initialized (run_id: {self.run_id})")
        logger.info(f"Live data: {self.live_data_enabled}, Learning: {self.learning_enabled}")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path:
            path = Path(config_path)
        else:
            path = Path(__file__).parent.parent / "config" / "agent_config.yaml"

        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _initialize_agents(self) -> None:
        """Initialize and register all agents."""
        order = 0
        
        # Live data agent (optional, runs first if enabled)
        if self.live_data_enabled:
            live_data_config = self._get_live_data_config()
            live_data_agent = LiveDataAgent(
                llm_client=self.llm_client,
                tools=self.tools,
                config=live_data_config,
            )
            self.manager.register_agent(live_data_agent, order=order)
            order += 1
        
        # Core agents
        research = ResearchAgent(llm_client=self.llm_client, tools=self.tools)
        writer = WriterAgent(llm_client=self.llm_client, tools=self.tools)
        factcheck = FactCheckAgent(llm_client=self.llm_client, tools=self.tools)
        optimizer = OptimizerAgent(llm_client=self.llm_client, tools=self.tools)

        self.manager.register_agent(research, order=order)
        self.manager.register_agent(writer, order=order + 1)
        self.manager.register_agent(factcheck, order=order + 2)
        self.manager.register_agent(optimizer, order=order + 3)

    def _get_live_data_config(self) -> LiveDataConfig:
        """Get live data configuration."""
        config = LiveDataConfig()
        
        tool_perms = self.config.get("tool_permissions", {}).get("live_data_agent", [])
        # Config can specify default sources
        live_config = self.config.get("live_data", {})
        
        if "rss" in tool_perms or not tool_perms:
            config.rss_feeds = live_config.get("rss_feeds", [])
        if "api" in tool_perms or not tool_perms:
            config.api_endpoints = live_config.get("api_endpoints", [])
        if "pdf" in tool_perms or not tool_perms:
            config.pdf_files = live_config.get("pdf_files", [])
        if "csv" in tool_perms or not tool_perms:
            config.csv_files = live_config.get("csv_files", [])
        
        return config

    def run(
        self,
        topic: str,
        data_source_url: Optional[str] = None,
        data_file: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ) -> str:
        """
        Execute the full pipeline for a topic.

        Args:
            topic: Topic to generate content for.
            data_source_url: Optional URL for live data.
            data_file: Optional file path for live data.
            api_endpoint: Optional API endpoint for live data.

        Returns:
            Generated content string.
        """
        self.run_start_time = time.time()
        logger.info(f"Starting pipeline for topic: {topic}")
        
        # Create run directory
        run_dir = self._create_run_directory()
        
        # Load strategy from memory if available
        strategy = None
        if self.learning_enabled and self.strategy_memory:
            strategy = self._load_strategy(topic)
            if strategy:
                logger.info(f"Loaded strategy: {strategy.strategy_id}")
                self._save_json(run_dir / "strategy_used.json", strategy.to_dict())
        
        # Clear previous data
        self.vector_store.clear()
        
        # Create context with metadata
        context = AgentContext(topic=topic)
        context.metadata["run_id"] = self.run_id
        context.metadata["live_data_enabled"] = self.live_data_enabled
        
        # Add live data source info
        if data_source_url:
            context.metadata["data_source_url"] = data_source_url
        if data_file:
            context.metadata["data_file"] = data_file
        if api_endpoint:
            context.metadata["api_endpoint"] = api_endpoint
        
        # Apply strategy adjustments
        if strategy:
            context.metadata["strategy"] = strategy.to_dict()
        
        # Save run config
        self._save_json(run_dir / "config.json", {
            "run_id": self.run_id,
            "topic": topic,
            "live_data_enabled": self.live_data_enabled,
            "learning_enabled": self.learning_enabled,
            "data_sources": {
                "url": data_source_url,
                "file": data_file,
                "api": api_endpoint,
            },
            "strategy_id": strategy.strategy_id if strategy else None,
        })
        
        # Execute pipeline with progress callback if available
        try:
            self.manager.execute_pipeline(
                context,
                progress_callback=getattr(self, '_progress_callback', None),
            )
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            if self.performance_tracker:
                self.performance_tracker.record_failure(
                    agent_name="pipeline",
                    run_id=self.run_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"topic": topic},
                )
            raise
        
        # Get final output
        final_output = context.optimized_content or context.draft_content or ""
        
        # Evaluate final output
        evaluation = self.evaluator.evaluate(final_output)
        
        # Save agent outputs
        self._save_agent_outputs(run_dir, context)
        
        # Save evaluation
        self._save_json(run_dir / "evaluation.json", {
            "overall_score": evaluation.overall_score,
            "readability_score": evaluation.readability_score,
            "length_score": evaluation.length_score,
            "keyword_density_score": evaluation.keyword_density_score,
            "structure_score": evaluation.structure_score,
            "details": evaluation.details,
        })
        
        # Record run history and performance
        execution_time = time.time() - self.run_start_time
        if self.learning_enabled:
            self._record_run(
                topic=topic,
                context=context,
                evaluation=evaluation,
                execution_time=execution_time,
                strategy=strategy,
            )
        
        # Log summary
        summary = self.manager.get_execution_summary()
        logger.info(f"Pipeline complete: {summary}")
        logger.info(f"Quality score: {evaluation.overall_score}")
        
        return final_output

    def _load_strategy(self, topic: str) -> Optional[Strategy]:
        """Load best strategy for topic."""
        if not self.strategy_memory:
            return None
        
        strategy = self.strategy_memory.get_best_strategy(topic)
        if strategy and strategy.average_score >= 70:
            return strategy
        return None

    def _record_run(
        self,
        topic: str,
        context: AgentContext,
        evaluation: Any,
        execution_time: float,
        strategy: Optional[Strategy],
    ) -> None:
        """Record run to history and update learning."""
        # Build run record
        record = RunRecord(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            topic=topic,
            agents_executed=list(context.agent_outputs.keys()),
            prompts_used={},  # Could track actual prompts used
            tool_usage=context.metadata.get("tool_usage", {}),
            output_length=len(context.optimized_content or context.draft_content or ""),
            evaluation_scores={
                "overall": evaluation.overall_score,
                "readability": evaluation.readability_score,
                "length": evaluation.length_score,
                "keyword_density": evaluation.keyword_density_score,
                "structure": evaluation.structure_score,
            },
            retry_counts=context.metadata.get("retry_counts", {}),
            execution_time=execution_time,
            final_quality_score=evaluation.overall_score,
            live_data_enabled=self.live_data_enabled,
            strategy_id=strategy.strategy_id if strategy else None,
            metadata={"live_data_output": context.metadata.get("live_data_output", {})},
        )
        
        # Store run record
        if self.run_history:
            self.run_history.store(record)
        
        # Record performance metrics per agent
        if self.performance_tracker:
            for agent_name, output in context.agent_outputs.items():
                metrics = PerformanceMetrics(
                    agent_name=agent_name,
                    run_id=self.run_id,
                    timestamp=datetime.now().isoformat(),
                    output_score=evaluation.overall_score,
                    readability_score=evaluation.readability_score,
                    length_score=evaluation.length_score,
                    keyword_density_score=evaluation.keyword_density_score,
                    structure_score=evaluation.structure_score,
                    citation_count=output.count("[") if output else 0,  # Simple citation count
                    execution_time=execution_time / max(1, len(context.agent_outputs)),
                    retry_count=context.metadata.get("retry_counts", {}).get(agent_name, 0),
                    success=True,
                )
                self.performance_tracker.record(metrics)
        
        # Update strategy memory
        if self.strategy_memory and evaluation.overall_score >= 70:
            new_strategy = Strategy(
                strategy_id=strategy.strategy_id if strategy else "",
                topic_category=self._categorize_topic(topic),
                topic_keywords=[],
                agent_order=list(context.agent_outputs.keys()),
                prompt_overrides={},
                tool_preferences={},
                live_data_sources=context.metadata.get("live_data_output", {}).get("sources_used", []),
                average_score=evaluation.overall_score,
                use_count=1,
                last_used=datetime.now().isoformat(),
                created_at=datetime.now().isoformat(),
            )
            self.strategy_memory.store_strategy(topic, new_strategy, evaluation.overall_score)

    def _categorize_topic(self, topic: str) -> str:
        """Simple topic categorization."""
        topic_lower = topic.lower()
        
        categories = {
            "technology": ["ai", "software", "tech", "digital", "computer", "data"],
            "healthcare": ["health", "medical", "medicine", "hospital", "patient"],
            "business": ["business", "market", "finance", "economy", "company"],
            "science": ["science", "research", "study", "experiment", "discovery"],
            "education": ["education", "learning", "school", "university", "teaching"],
        }
        
        for category, keywords in categories.items():
            if any(kw in topic_lower for kw in keywords):
                return category
        
        return "general"

    def _create_run_directory(self) -> Path:
        """Create run output directory."""
        run_dir = self.runs_dir / self.run_id
        run_dir.mkdir(exist_ok=True)
        (run_dir / "agent_outputs").mkdir(exist_ok=True)
        (run_dir / "live_data_raw").mkdir(exist_ok=True)
        return run_dir

    def _save_json(self, path: Path, data: Dict) -> None:
        """Save data as JSON file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _save_agent_outputs(self, run_dir: Path, context: AgentContext) -> None:
        """Save individual agent outputs."""
        outputs_dir = run_dir / "agent_outputs"
        
        for agent_name, output in context.agent_outputs.items():
            output_file = outputs_dir / f"{agent_name}.txt"
            with open(output_file, "w") as f:
                f.write(output or "")

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        summary = self.manager.get_execution_summary()
        summary["run_id"] = self.run_id
        summary["live_data_enabled"] = self.live_data_enabled
        summary["learning_enabled"] = self.learning_enabled
        return summary

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning system statistics."""
        if not self.learning_enabled:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "run_history": self.run_history.get_statistics() if self.run_history else {},
            "performance": self.performance_tracker.get_overall_stats() if self.performance_tracker else {},
            "strategies": len(self.strategy_memory.get_all_strategies()) if self.strategy_memory else 0,
        }

    def run_from_cli_config(
        self,
        topic: str,
        data_source_url: Optional[str] = None,
        data_file: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        live_data_enabled: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """
        Execute pipeline from CLI configuration.
        
        This is the adapter method for ATTIC CLI integration.
        It wraps the existing run() method without duplicating logic.

        Args:
            topic: Topic to generate content for.
            data_source_url: Optional URL for live data (RSS, PDF URL).
            data_file: Optional local file path (CSV, PDF).
            api_endpoint: Optional API endpoint URL.
            live_data_enabled: Whether to enable live data agent.
            progress_callback: Optional callback for progress updates.
                Signature: callback(agent_name: str, status: str)
                status is one of: 'start', 'complete', 'error'

        Returns:
            Generated content string.
        """
        # Update pipeline configuration for this run
        self.live_data_enabled = live_data_enabled
        
        # Re-initialize agents if live data mode changed
        has_live_data_agent = any(
            agent.name == "live_data_agent" 
            for agent in self.manager.agents.values()
        )
        
        if live_data_enabled and not has_live_data_agent:
            self.manager.agents.clear()
            self.manager.execution_order.clear()
            self._initialize_agents()
        elif not live_data_enabled and has_live_data_agent:
            self.manager.agents.clear()
            self.manager.execution_order.clear()
            self._initialize_agents()
        
        # Store progress callback for potential future use
        self._progress_callback = progress_callback
        
        # Call existing run method
        result = self.run(
            topic=topic,
            data_source_url=data_source_url,
            data_file=data_file,
            api_endpoint=api_endpoint,
        )
        
        return result

