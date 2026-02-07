"""
Live Data Agent - Ingests external data sources for research.

This agent fetches live data from RSS feeds, APIs, PDFs, and CSVs,
normalizing them into research notes for the content pipeline.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentContext
from tools.live_data_tools import LiveDataTools, DataSourceType, DataSource
from tools.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class LiveDataConfig:
    """Configuration for live data sources."""
    rss_feeds: List[str] = field(default_factory=list)
    api_endpoints: List[str] = field(default_factory=list)
    pdf_files: List[str] = field(default_factory=list)
    csv_files: List[str] = field(default_factory=list)
    auto_detect_sources: List[str] = field(default_factory=list)


class LiveDataAgent(BaseAgent):
    """
    Agent for ingesting live external data.
    
    Fetches data from multiple source types, normalizes into
    research notes, and stores in vector store for downstream agents.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: Optional[Dict[str, Any]] = None,
        config: Optional[LiveDataConfig] = None,
    ):
        """
        Initialize the live data agent.

        Args:
            llm_client: LLM client for text processing.
            tools: Dictionary of available tools.
            config: Live data source configuration.
        """
        super().__init__(
            name="live_data_agent",
            role="Live Data Specialist",
            llm_client=llm_client,
            tools=tools,
        )
        self.live_data_tools = LiveDataTools()
        self.source_config = config or LiveDataConfig()

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """You are a Live Data Specialist agent responsible for analyzing and summarizing data from external sources.

Your role is to:
1. Analyze raw data from RSS feeds, APIs, PDFs, and CSV files
2. Extract key findings relevant to the given topic
3. Identify important statistics, facts, and trends
4. Normalize information into structured research notes

When analyzing data, focus on:
- Relevance to the topic
- Recency and timeliness
- Credibility indicators
- Quantitative data points
- Key quotes or statements

Output your analysis in a structured JSON format."""

    def _build_prompt(self, context: AgentContext) -> str:
        """Build the prompt for analyzing fetched data."""
        # Get data summary from context
        data_summary = context.metadata.get("live_data_summary", {})
        raw_content = context.metadata.get("live_data_content", "")
        
        return f"""Analyze the following live data for the topic: "{context.topic}"

DATA SOURCES SUMMARY:
{json.dumps(data_summary, indent=2)}

RAW CONTENT (truncated):
{raw_content[:10000]}

Provide a structured analysis with:
1. Key findings relevant to the topic
2. Important statistics or data points
3. Notable sources and their credibility
4. Recommended focus areas for content creation

Format your response as JSON:
{{
    "key_findings": ["finding1", "finding2", ...],
    "statistics": {{"stat_name": "value", ...}},
    "notable_sources": ["source1", "source2", ...],
    "focus_areas": ["area1", "area2", ...],
    "summary": "Brief overall summary"
}}"""

    def run(self, context: AgentContext) -> str:
        """
        Execute live data ingestion and analysis.

        Args:
            context: Current pipeline context.

        Returns:
            JSON string with structured findings.
        """
        logger.info(f"[{self.name}] Starting live data ingestion for: {context.topic}")
        
        # Collect all sources to fetch
        sources = self._collect_sources(context)
        
        if not sources:
            logger.warning(f"[{self.name}] No data sources configured")
            return json.dumps({
                "sources_used": [],
                "documents_loaded": 0,
                "key_findings": [],
                "raw_chunks": [],
                "error": "No data sources configured",
            })

        # Fetch data from all sources
        results = self.live_data_tools.fetch_multiple(sources)
        summary = self.live_data_tools.summarize_results(results)
        
        logger.info(f"[{self.name}] Fetched from {summary['successful']}/{summary['total_sources']} sources")

        # Store documents in vector store
        documents = self.live_data_tools.get_all_documents(results)
        stored_count = self._store_documents(documents, context)
        
        # Prepare content for LLM analysis
        content_preview = self._prepare_content_preview(documents)
        context.metadata["live_data_summary"] = summary
        context.metadata["live_data_content"] = content_preview
        
        # Use LLM to analyze and extract key findings
        analysis = self._analyze_with_llm(context)
        
        # Build final output
        output = {
            "sources_used": [r.source for r in results if r.success],
            "documents_loaded": stored_count,
            "key_findings": analysis.get("key_findings", []),
            "raw_chunks": [doc["content"][:500] for doc in documents[:10]],
            "statistics": analysis.get("statistics", {}),
            "focus_areas": analysis.get("focus_areas", []),
            "summary": analysis.get("summary", ""),
            "source_summary": summary,
        }
        
        # Store in context for research agent
        context.metadata["live_data_output"] = output
        
        return json.dumps(output, indent=2)

    def _collect_sources(self, context: AgentContext) -> List[DataSource]:
        """Collect all configured data sources."""
        sources = []
        
        # From agent config
        for url in self.source_config.rss_feeds:
            sources.append(DataSource(DataSourceType.RSS, url=url))
        
        for url in self.source_config.api_endpoints:
            sources.append(DataSource(DataSourceType.API, url=url))
        
        for path in self.source_config.pdf_files:
            sources.append(DataSource(DataSourceType.PDF, file_path=path))
        
        for path in self.source_config.csv_files:
            sources.append(DataSource(DataSourceType.CSV, file_path=path))
        
        # Auto-detect sources
        for source in self.source_config.auto_detect_sources:
            source_type = self.live_data_tools.detect_source_type(source)
            if source.startswith(("http://", "https://")):
                sources.append(DataSource(source_type, url=source))
            else:
                sources.append(DataSource(source_type, file_path=source))
        
        # From context metadata (CLI arguments)
        if "data_source_url" in context.metadata:
            url = context.metadata["data_source_url"]
            source_type = self.live_data_tools.detect_source_type(url)
            sources.append(DataSource(source_type, url=url))
        
        if "data_file" in context.metadata:
            path = context.metadata["data_file"]
            source_type = self.live_data_tools.detect_source_type(path)
            sources.append(DataSource(source_type, file_path=path))
        
        if "api_endpoint" in context.metadata:
            url = context.metadata["api_endpoint"]
            sources.append(DataSource(DataSourceType.API, url=url))

        # Generate topic-based RSS feeds if no sources specified
        if not sources:
            sources = self._generate_topic_sources(context.topic)

        return sources

    def _generate_topic_sources(self, topic: str) -> List[DataSource]:
        """Generate default sources based on topic."""
        # Default news RSS feeds
        default_feeds = [
            f"https://news.google.com/rss/search?q={topic.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en",
        ]
        
        sources = []
        for feed in default_feeds:
            sources.append(DataSource(DataSourceType.RSS, url=feed))
        
        return sources

    def _store_documents(self, documents: List[dict], context: AgentContext) -> int:
        """Store documents in vector store."""
        vector_store = self.tools.get("vector_store")
        if not vector_store:
            logger.warning("Vector store not available")
            return 0
        
        stored = 0
        for doc in documents:
            try:
                metadata = doc.get("metadata", {})
                metadata["agent"] = self.name
                metadata["topic"] = context.topic
                
                vector_store.store_document(
                    content=doc["content"],
                    metadata=metadata,
                )
                stored += 1
            except Exception as e:
                logger.warning(f"Failed to store document: {e}")
        
        logger.info(f"[{self.name}] Stored {stored} documents in vector store")
        return stored

    def _prepare_content_preview(self, documents: List[dict]) -> str:
        """Prepare content preview for LLM analysis."""
        previews = []
        total_chars = 0
        max_chars = 15000  # Limit for LLM context
        
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            source_type = metadata.get("type", "unknown")
            
            preview = f"[{source_type.upper()}] {content[:1000]}"
            
            if total_chars + len(preview) > max_chars:
                break
            
            previews.append(preview)
            total_chars += len(preview)
        
        return "\n\n---\n\n".join(previews)

    def _analyze_with_llm(self, context: AgentContext) -> Dict[str, Any]:
        """Use LLM to analyze fetched data."""
        try:
            prompt = self._build_prompt(context)
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
                max_tokens=2000,
                temperature=0.5,
            )
            
            # Try to parse as JSON
            try:
                # Find JSON in response
                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
            
            # Fallback: extract key points manually
            return {
                "key_findings": [response[:500]],
                "summary": response[:1000],
            }
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {"error": str(e)}
