"""
Live Data Tools - Unified interface for live data ingestion.

Provides a unified API for accessing RSS, API, PDF, and CSV data sources.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .rss_reader import RSSReader, RSSEntry
from .api_client import APIClient, APIResponse
from .pdf_loader import PDFLoader, PDFChunk
from .csv_loader import CSVLoader, CSVSummary

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Supported data source types."""
    RSS = "rss"
    API = "api"
    PDF = "pdf"
    CSV = "csv"
    UNKNOWN = "unknown"


@dataclass
class DataSource:
    """Configuration for a data source."""
    source_type: DataSourceType
    url: Optional[str] = None
    file_path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveDataResult:
    """Result from live data ingestion."""
    source_type: DataSourceType
    source: str
    success: bool
    documents: List[dict]
    metadata: Dict[str, Any]
    error: Optional[str] = None


class LiveDataTools:
    """
    Unified interface for live data ingestion.
    
    Provides a single API for accessing multiple data source types
    and converting them to documents for vector store ingestion.
    """

    def __init__(self):
        """Initialize all data source tools."""
        self.rss_reader = RSSReader()
        self.api_client = APIClient()
        self.pdf_loader = PDFLoader()
        self.csv_loader = CSVLoader()
        logger.info("Initialized LiveDataTools")

    def detect_source_type(self, source: str) -> DataSourceType:
        """
        Detect data source type from URL or file path.

        Args:
            source: URL or file path.

        Returns:
            Detected DataSourceType.
        """
        source_lower = source.lower()
        
        # Check file extensions
        if source_lower.endswith(".pdf"):
            return DataSourceType.PDF
        elif source_lower.endswith(".csv"):
            return DataSourceType.CSV
        elif source_lower.endswith((".rss", ".xml", ".atom")):
            return DataSourceType.RSS
        elif source_lower.endswith(".json"):
            return DataSourceType.API
        
        # Check URL patterns
        if "rss" in source_lower or "feed" in source_lower:
            return DataSourceType.RSS
        elif "/api/" in source_lower:
            return DataSourceType.API
        elif source_lower.startswith(("http://", "https://")):
            # Default to API for HTTP URLs
            return DataSourceType.API
        elif Path(source).exists():
            # Check local file
            suffix = Path(source).suffix.lower()
            if suffix == ".pdf":
                return DataSourceType.PDF
            elif suffix == ".csv":
                return DataSourceType.CSV
            elif suffix in (".rss", ".xml"):
                return DataSourceType.RSS
        
        return DataSourceType.UNKNOWN

    def fetch(
        self,
        source: str,
        source_type: Optional[DataSourceType] = None,
        **kwargs,
    ) -> LiveDataResult:
        """
        Fetch data from source.

        Args:
            source: URL or file path.
            source_type: Optional explicit source type.
            **kwargs: Additional arguments for specific loaders.

        Returns:
            LiveDataResult with documents and metadata.
        """
        if source_type is None:
            source_type = self.detect_source_type(source)

        logger.info(f"Fetching {source_type.value} data from: {source}")

        try:
            if source_type == DataSourceType.RSS:
                return self._fetch_rss(source, **kwargs)
            elif source_type == DataSourceType.API:
                return self._fetch_api(source, **kwargs)
            elif source_type == DataSourceType.PDF:
                return self._fetch_pdf(source, **kwargs)
            elif source_type == DataSourceType.CSV:
                return self._fetch_csv(source, **kwargs)
            else:
                return LiveDataResult(
                    source_type=source_type,
                    source=source,
                    success=False,
                    documents=[],
                    metadata={},
                    error=f"Unknown source type: {source_type}",
                )
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return LiveDataResult(
                source_type=source_type,
                source=source,
                success=False,
                documents=[],
                metadata={"error": str(e)},
                error=str(e),
            )

    def _fetch_rss(self, url: str, max_entries: int = 20, **kwargs) -> LiveDataResult:
        """Fetch RSS feed data."""
        entries = self.rss_reader.fetch(url, max_entries=max_entries)
        documents = self.rss_reader.to_documents(entries)
        
        return LiveDataResult(
            source_type=DataSourceType.RSS,
            source=url,
            success=len(entries) > 0,
            documents=documents,
            metadata={
                "entry_count": len(entries),
                "titles": [e.title for e in entries[:5]],
            },
        )

    def _fetch_api(self, url: str, params: Optional[dict] = None, **kwargs) -> LiveDataResult:
        """Fetch API data."""
        response = self.api_client.get(url, params=params)
        documents = self.api_client.to_documents(response)
        
        return LiveDataResult(
            source_type=DataSourceType.API,
            source=url,
            success=response.success,
            documents=documents,
            metadata={
                "status_code": response.status_code,
                "response_time": response.response_time,
                "document_count": len(documents),
            },
            error=response.error,
        )

    def _fetch_pdf(self, source: str, **kwargs) -> LiveDataResult:
        """Fetch PDF data."""
        chunks = self.pdf_loader.load(source)
        documents = self.pdf_loader.to_documents(chunks)
        summary = self.pdf_loader.get_summary(chunks)
        
        return LiveDataResult(
            source_type=DataSourceType.PDF,
            source=source,
            success=len(chunks) > 0,
            documents=documents,
            metadata=summary,
        )

    def _fetch_csv(self, source: str, chunk_size: int = 10, **kwargs) -> LiveDataResult:
        """Fetch CSV data."""
        rows = self.csv_loader.load(source)
        documents = self.csv_loader.to_documents(rows, source, chunk_size)
        summary = self.csv_loader.get_summary(rows, source)
        
        return LiveDataResult(
            source_type=DataSourceType.CSV,
            source=source,
            success=len(rows) > 0,
            documents=documents,
            metadata={
                "row_count": summary.row_count,
                "column_count": summary.column_count,
                "columns": summary.columns,
            },
        )

    def fetch_multiple(
        self,
        sources: List[Union[str, DataSource]],
    ) -> List[LiveDataResult]:
        """
        Fetch data from multiple sources.

        Args:
            sources: List of URLs, file paths, or DataSource configs.

        Returns:
            List of LiveDataResults.
        """
        results = []
        
        for source in sources:
            if isinstance(source, DataSource):
                result = self.fetch(
                    source.url or source.file_path or "",
                    source_type=source.source_type,
                    **source.params,
                )
            else:
                result = self.fetch(source)
            
            results.append(result)

        logger.info(f"Fetched data from {len(results)} sources")
        return results

    def get_all_documents(self, results: List[LiveDataResult]) -> List[dict]:
        """
        Combine documents from multiple results.

        Args:
            results: List of LiveDataResults.

        Returns:
            Combined list of documents.
        """
        documents = []
        for result in results:
            if result.success:
                documents.extend(result.documents)
        return documents

    def summarize_results(self, results: List[LiveDataResult]) -> Dict[str, Any]:
        """
        Generate summary of fetch results.

        Args:
            results: List of LiveDataResults.

        Returns:
            Summary dictionary.
        """
        return {
            "total_sources": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "total_documents": sum(len(r.documents) for r in results),
            "by_type": {
                t.value: sum(1 for r in results if r.source_type == t and r.success)
                for t in DataSourceType
                if t != DataSourceType.UNKNOWN
            },
            "sources": [
                {
                    "source": r.source,
                    "type": r.source_type.value,
                    "success": r.success,
                    "documents": len(r.documents),
                    "error": r.error,
                }
                for r in results
            ],
        }
