"""
Search Tool - Pluggable web search interface.

This module provides an abstraction for web search functionality,
with a mock implementation for development and testing.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result."""
    title: str
    url: str
    snippet: str
    source: str = "web"


class SearchTool(ABC):
    """Abstract base class for search tools."""

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Perform a web search.

        Args:
            query: The search query string.
            num_results: Maximum number of results to return.

        Returns:
            List of SearchResult objects.
        """
        pass

    def format_results(self, results: List[SearchResult]) -> str:
        """
        Format search results as a readable string.

        Args:
            results: List of search results.

        Returns:
            Formatted string representation.
        """
        if not results:
            return "No search results found."

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"[{i}] {result.title}\n"
                f"    Source: {result.source}\n"
                f"    URL: {result.url}\n"
                f"    Summary: {result.snippet}\n"
            )
        return "\n".join(formatted)


class MockSearchTool(SearchTool):
    """
    Mock search tool for development and testing.
    
    Returns realistic-looking but synthetic search results
    based on the query topic.
    """

    # Knowledge base for mock responses
    MOCK_KNOWLEDGE = {
        "ai": [
            SearchResult(
                title="Artificial Intelligence: A Modern Approach",
                url="https://example.com/ai-textbook",
                snippet="AI encompasses machine learning, neural networks, natural language processing, and robotics. Modern AI systems use deep learning for tasks like image recognition and language understanding.",
                source="Academic"
            ),
            SearchResult(
                title="The State of AI in 2024",
                url="https://example.com/ai-report-2024",
                snippet="Large language models have revolutionized NLP. Multimodal AI combines vision, language, and reasoning. Edge AI enables on-device intelligence.",
                source="Industry Report"
            ),
            SearchResult(
                title="AI Ethics and Governance",
                url="https://example.com/ai-ethics",
                snippet="Key concerns include bias in AI systems, privacy implications, job displacement, and the need for transparent and explainable AI decisions.",
                source="Research Paper"
            ),
        ],
        "healthcare": [
            SearchResult(
                title="AI in Healthcare: Transforming Patient Care",
                url="https://example.com/healthcare-ai",
                snippet="AI applications in healthcare include diagnostic imaging, drug discovery, personalized treatment plans, and predictive analytics for patient outcomes.",
                source="Medical Journal"
            ),
            SearchResult(
                title="Digital Health Revolution",
                url="https://example.com/digital-health",
                snippet="Telemedicine adoption increased 300% post-pandemic. Wearable devices monitor vital signs continuously. Electronic health records enable data-driven care.",
                source="Healthcare News"
            ),
            SearchResult(
                title="Clinical Decision Support Systems",
                url="https://example.com/cdss",
                snippet="AI-powered CDSS help physicians diagnose diseases, recommend treatments, and identify drug interactions. Studies show 15-30% improvement in diagnostic accuracy.",
                source="Clinical Study"
            ),
        ],
        "technology": [
            SearchResult(
                title="Emerging Technology Trends",
                url="https://example.com/tech-trends",
                snippet="Key trends include quantum computing, 6G networks, sustainable tech, biotechnology convergence, and the metaverse. Investment in these areas exceeds $500B annually.",
                source="Tech Analysis"
            ),
            SearchResult(
                title="Cloud Computing Evolution",
                url="https://example.com/cloud-evolution",
                snippet="Multi-cloud strategies dominate enterprise IT. Serverless computing reduces operational overhead. Edge computing brings processing closer to data sources.",
                source="Enterprise IT"
            ),
        ],
        "default": [
            SearchResult(
                title="Comprehensive Overview",
                url="https://example.com/overview",
                snippet="This topic encompasses multiple disciplines and has significant implications for society, economy, and technology advancement.",
                source="Encyclopedia"
            ),
            SearchResult(
                title="Latest Research and Developments",
                url="https://example.com/research",
                snippet="Recent studies highlight emerging trends, challenges, and opportunities in this field. Interdisciplinary approaches yield promising results.",
                source="Research Database"
            ),
            SearchResult(
                title="Practical Applications and Case Studies",
                url="https://example.com/case-studies",
                snippet="Real-world implementations demonstrate tangible benefits. Organizations report improved efficiency, cost savings, and innovation acceleration.",
                source="Industry Analysis"
            ),
        ],
    }

    def __init__(self):
        """Initialize the mock search tool."""
        logger.info("Initialized MockSearchTool")

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Perform a mock search.

        Args:
            query: The search query string.
            num_results: Maximum number of results to return.

        Returns:
            List of mock SearchResult objects.
        """
        logger.info(f"Mock search for: {query}")
        query_lower = query.lower()

        # Find matching knowledge base entries
        results = []
        for keyword, entries in self.MOCK_KNOWLEDGE.items():
            if keyword in query_lower and keyword != "default":
                results.extend(entries)

        # Use default if no specific matches
        if not results:
            results = self.MOCK_KNOWLEDGE["default"].copy()

        # Customize results based on query
        customized = []
        for result in results[:num_results]:
            customized.append(SearchResult(
                title=f"{result.title} - {query.split()[0].title()}",
                url=result.url,
                snippet=result.snippet,
                source=result.source,
            ))

        logger.info(f"Returning {len(customized)} mock results")
        return customized


class WebSearchTool(SearchTool):
    """
    Real web search implementation.
    
    Placeholder for integration with actual search APIs
    (e.g., Serper, SerpAPI, Brave Search).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ):
        """
        Initialize the web search tool.

        Args:
            api_key: API key for the search service.
            api_endpoint: Base URL for the search API.
        """
        self.api_key = api_key
        self.api_endpoint = api_endpoint or "https://api.search.example.com/v1/search"
        logger.info("Initialized WebSearchTool")

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Perform a real web search.

        Args:
            query: The search query string.
            num_results: Maximum number of results to return.

        Returns:
            List of SearchResult objects.

        Note:
            This is a template implementation. Replace with actual
            API integration for production use.
        """
        logger.info(f"Web search for: {query}")

        if not self.api_key:
            logger.warning("No API key configured, falling back to mock results")
            return MockSearchTool().search(query, num_results)

        try:
            response = requests.get(
                self.api_endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={"q": query, "num": num_results},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", [])[:num_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    source="web",
                ))
            return results

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            raise
