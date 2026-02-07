"""
RSS Reader - Fetch and parse RSS feeds.

Provides structured access to RSS feed content for live data ingestion.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from xml.etree import ElementTree

import requests

logger = logging.getLogger(__name__)


@dataclass
class RSSEntry:
    """Structured RSS feed entry."""
    title: str
    link: str
    summary: str
    published: Optional[datetime]
    source: str


class RSSReader:
    """
    RSS feed reader with parsing and normalization.
    
    Fetches RSS/Atom feeds from URLs and returns structured entries.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: str = "AIContentFactory/1.0",
    ):
        """
        Initialize RSS reader.

        Args:
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts on failure.
            user_agent: User agent string for requests.
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {"User-Agent": user_agent}
        logger.info("Initialized RSSReader")

    def fetch(self, url: str, max_entries: int = 20) -> List[RSSEntry]:
        """
        Fetch and parse RSS feed from URL.

        Args:
            url: RSS feed URL.
            max_entries: Maximum entries to return.

        Returns:
            List of RSSEntry objects.
        """
        logger.info(f"Fetching RSS feed: {url}")
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                
                entries = self._parse_feed(response.text, url)
                logger.info(f"Parsed {len(entries)} entries from RSS feed")
                return entries[:max_entries]

            except requests.RequestException as e:
                logger.warning(f"RSS fetch attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to fetch RSS feed: {url}")
                    return []

        return []

    def _parse_feed(self, content: str, source_url: str) -> List[RSSEntry]:
        """Parse RSS/Atom feed content."""
        entries = []
        
        try:
            root = ElementTree.fromstring(content)
        except ElementTree.ParseError as e:
            logger.error(f"Failed to parse RSS XML: {e}")
            return []

        # Try RSS 2.0 format
        for item in root.findall(".//item"):
            entry = self._parse_rss_item(item, source_url)
            if entry:
                entries.append(entry)

        # Try Atom format if no RSS items found
        if not entries:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for item in root.findall(".//atom:entry", ns):
                entry = self._parse_atom_entry(item, ns, source_url)
                if entry:
                    entries.append(entry)

        return entries

    def _parse_rss_item(self, item: ElementTree.Element, source_url: str) -> Optional[RSSEntry]:
        """Parse RSS 2.0 item element."""
        try:
            title = self._get_text(item, "title") or "Untitled"
            link = self._get_text(item, "link") or ""
            
            # Get description/summary
            summary = self._get_text(item, "description") or ""
            summary = self._clean_html(summary)[:500]
            
            # Parse date
            pub_date = self._get_text(item, "pubDate")
            published = self._parse_date(pub_date) if pub_date else None

            return RSSEntry(
                title=title,
                link=link,
                summary=summary,
                published=published,
                source=source_url,
            )
        except Exception as e:
            logger.warning(f"Failed to parse RSS item: {e}")
            return None

    def _parse_atom_entry(
        self,
        entry: ElementTree.Element,
        ns: dict,
        source_url: str,
    ) -> Optional[RSSEntry]:
        """Parse Atom entry element."""
        try:
            title = self._get_text(entry, "atom:title", ns) or "Untitled"
            
            # Get link from href attribute
            link_elem = entry.find("atom:link", ns)
            link = link_elem.get("href", "") if link_elem is not None else ""
            
            # Get summary
            summary = self._get_text(entry, "atom:summary", ns) or ""
            if not summary:
                summary = self._get_text(entry, "atom:content", ns) or ""
            summary = self._clean_html(summary)[:500]
            
            # Parse date
            updated = self._get_text(entry, "atom:updated", ns)
            published = self._parse_date(updated) if updated else None

            return RSSEntry(
                title=title,
                link=link,
                summary=summary,
                published=published,
                source=source_url,
            )
        except Exception as e:
            logger.warning(f"Failed to parse Atom entry: {e}")
            return None

    def _get_text(
        self,
        element: ElementTree.Element,
        path: str,
        ns: Optional[dict] = None,
    ) -> Optional[str]:
        """Get text content from element."""
        child = element.find(path, ns) if ns else element.find(path)
        return child.text.strip() if child is not None and child.text else None

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",  # RFC 822
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        return None

    def to_documents(self, entries: List[RSSEntry]) -> List[dict]:
        """
        Convert RSS entries to document format for vector store.

        Args:
            entries: List of RSSEntry objects.

        Returns:
            List of document dictionaries.
        """
        documents = []
        for entry in entries:
            doc = {
                "content": f"{entry.title}\n\n{entry.summary}",
                "metadata": {
                    "type": "rss",
                    "title": entry.title,
                    "link": entry.link,
                    "source": entry.source,
                    "published": entry.published.isoformat() if entry.published else None,
                },
            }
            documents.append(doc)
        return documents
