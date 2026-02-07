"""
DuckDuckGo Search Tool - Real web search integration using the ddgs package.
"""

import logging
from typing import List, Optional

try:
    from ddgs import DDGS
    DDG_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDG_AVAILABLE = True
    except ImportError:
        DDG_AVAILABLE = False

from .search_tool import SearchTool, SearchResult, MockSearchTool

logger = logging.getLogger(__name__)


class DuckDuckGoSearchTool(SearchTool):
    """
    Real web search using DuckDuckGo.
    
    Free, no API key required. Returns actual search results from the web.
    """

    def __init__(self, timeout: int = 10):
        """
        Initialize DuckDuckGo search tool.

        Args:
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout
        self._fallback = MockSearchTool()
        
        if not DDG_AVAILABLE:
            logger.warning(
                "ddgs not installed. Install with: pip install ddgs"
            )
        else:
            logger.info("Initialized DuckDuckGoSearchTool")

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Perform a real web search using DuckDuckGo.

        Args:
            query: The search query string.
            num_results: Maximum number of results to return.

        Returns:
            List of SearchResult objects with real web data.
        """
        if not DDG_AVAILABLE:
            logger.warning("DDG not available, using mock results")
            return self._fallback.search(query, num_results)

        logger.info(f"DuckDuckGo search for: {query}")
        
        results = []
        try:
            with DDGS(timeout=self.timeout) as ddgs:
                # Try regular text search first
                raw_results = list(ddgs.text(
                    query,
                    max_results=num_results,
                    safesearch="moderate",
                ))
            
            for item in raw_results:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", item.get("link", "")),
                    snippet=item.get("body", item.get("snippet", "")),
                    source="DuckDuckGo",
                ))
            
            # If no results found, try news search as fallback
            if not results:
                logger.info(f"No text results for {query}, trying news fallback")
                return self.search_news(query, num_results)

            logger.info(f"Found {len(results)} real search results")
            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            logger.info("Falling back to news search")
            try:
                return self.search_news(query, num_results)
            except Exception:
                logger.info("News search also failed, falling back to mock")
                return self._fallback.search(query, num_results)

    def search_news(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Search for recent news articles.

        Args:
            query: The search query string.
            num_results: Maximum number of results.

        Returns:
            List of news SearchResult objects.
        """
        if not DDG_AVAILABLE:
            return self._fallback.search(query, num_results)

        logger.info(f"DuckDuckGo news search for: {query}")
        
        try:
            with DDGS(timeout=self.timeout) as ddgs:
                raw_results = list(ddgs.news(
                    query,
                    max_results=num_results,
                    safesearch="moderate",
                ))
            
            results = []
            for item in raw_results:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", item.get("link", "")),
                    snippet=item.get("body", item.get("excerpt", "")),
                    source=item.get("source", "News"),
                ))
            
            logger.info(f"Found {len(results)} news results")
            return results

        except Exception as e:
            logger.error(f"DuckDuckGo news search failed: {e}")
            return self._fallback.search(query, num_results)


def get_search_tool(prefer_real: bool = True) -> SearchTool:
    """
    Factory function to get the best available search tool.

    Args:
        prefer_real: If True, prefer DuckDuckGo over mock.

    Returns:
        SearchTool instance.
    """
    if prefer_real and DDG_AVAILABLE:
        return DuckDuckGoSearchTool()
    else:
        logger.info("Using MockSearchTool")
        return MockSearchTool()
