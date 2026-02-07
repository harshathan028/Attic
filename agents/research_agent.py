"""
Research Agent - Gathers structured research notes on a topic.

This agent searches for information and stores findings in the vector database
for later retrieval by other agents in the pipeline.
"""

import logging
from typing import Optional

from .base_agent import BaseAgent, AgentContext
from tools.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Research agent that gathers and structures information.
    
    Uses search tools to find relevant information about a topic,
    synthesizes findings using the LLM, and stores results in
    the vector database for other agents to access.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        **kwargs,
    ):
        """
        Initialize the research agent.

        Args:
            llm_client: LLM client for text generation.
            **kwargs: Additional arguments passed to BaseAgent.
        """
        super().__init__(
            name="research_agent",
            role="Research Specialist",
            llm_client=llm_client,
            prompt_file="research.txt",
            **kwargs,
        )

    def run(self, context: AgentContext) -> str:
        """
        Execute research on the given topic.

        Args:
            context: Pipeline context containing the topic.

        Returns:
            Structured research notes.
        """
        topic = context.topic
        logger.info(f"[{self.name}] Researching topic: {topic}")

        # Step 1: Search for information
        search_results = self._gather_search_results(topic)

        # Step 2: Synthesize research using LLM
        research_notes = self._synthesize_research(topic, search_results)

        # Step 3: Store in vector database
        self._store_research(topic, research_notes, context)

        # Update context
        context.research_notes = research_notes

        return research_notes

    def _gather_search_results(self, topic: str) -> str:
        """Gather search results for the topic."""
        if not self.search_tool:
            logger.warning("No search tool available, using topic directly")
            return f"Topic to research: {topic}"

        # Perform multiple searches with different angles
        queries = [
            topic,
            f"{topic} key facts",
            f"{topic} recent developments",
            f"{topic} statistics data",
        ]

        all_results = []
        for query in queries:
            try:
                results = self.search_tool.search(query, num_results=3)
                formatted = self.search_tool.format_results(results)
                all_results.append(f"Query: {query}\n{formatted}")
            except Exception as e:
                logger.warning(f"Search failed for '{query}': {e}")

        return "\n\n".join(all_results) if all_results else f"Topic: {topic}"

    def _synthesize_research(self, topic: str, search_results: str) -> str:
        """Synthesize search results into structured notes."""
        prompt = self.get_prompt(
            topic=topic,
            search_results=search_results,
        )

        if not prompt:
            # Fallback prompt if template not loaded
            prompt = f"""You are a research specialist. Analyze the following search results 
about "{topic}" and create comprehensive, structured research notes.

Search Results:
{search_results}

Create detailed research notes covering:
1. Key Facts and Definitions
2. Current State and Trends
3. Important Statistics and Data
4. Key Players and Stakeholders
5. Challenges and Opportunities
6. Recent Developments

Format your notes clearly with headers and bullet points."""

        system_prompt = (
            "You are an expert research analyst. Synthesize information into "
            "clear, factual, and well-organized research notes. Focus on accuracy "
            "and comprehensiveness."
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2000,
                temperature=0.5,  # Lower temperature for factual content
            )
            return response
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return f"Research Notes for: {topic}\n\n{search_results}"

    def _store_research(
        self,
        topic: str,
        research_notes: str,
        context: AgentContext,
    ) -> None:
        """Store research notes in the vector database."""
        if not self.vector_store:
            logger.warning("No vector store available, skipping storage")
            return

        try:
            # Split research into chunks for better retrieval
            chunks = self._chunk_research(research_notes)
            
            for i, chunk in enumerate(chunks):
                self.vector_store.store_document(
                    content=chunk,
                    metadata={
                        "topic": topic,
                        "agent": self.name,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "type": "research",
                    },
                )
            
            logger.info(f"Stored {len(chunks)} research chunks in vector store")
            
        except Exception as e:
            logger.error(f"Failed to store research: {e}")

    def _chunk_research(self, research_notes: str, max_chunk_size: int = 500) -> list:
        """Split research notes into manageable chunks."""
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in research_notes.split("\n\n") if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para.split())
            
            if current_size + para_size > max_chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks if chunks else [research_notes]
