"""
CSV Loader - Load and analyze CSV datasets.

Provides CSV loading with column detection, summary statistics,
and conversion to documents for vector store.
"""

import csv
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)


@dataclass
class CSVSummary:
    """CSV dataset summary statistics."""
    row_count: int
    column_count: int
    columns: List[str]
    column_types: Dict[str, str]
    sample_rows: List[Dict[str, Any]]
    source: str


class CSVLoader:
    """
    CSV dataset loader with analysis capabilities.
    
    Loads CSV files or URLs, detects column types,
    and converts to documents for vector store ingestion.
    """

    def __init__(
        self,
        max_rows: int = 10000,
        sample_size: int = 5,
        timeout: int = 60,
    ):
        """
        Initialize CSV loader.

        Args:
            max_rows: Maximum rows to load.
            sample_size: Number of sample rows to include in summary.
            timeout: Download timeout for URL-based CSVs.
        """
        self.max_rows = max_rows
        self.sample_size = sample_size
        self.timeout = timeout
        logger.info("Initialized CSVLoader")

    def load(self, source: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load CSV data from file or URL.

        Args:
            source: File path or URL to CSV.

        Returns:
            List of row dictionaries.
        """
        source_str = str(source)
        logger.info(f"Loading CSV: {source_str}")

        try:
            if source_str.startswith(("http://", "https://")):
                return self._load_from_url(source_str)
            else:
                return self._load_from_file(Path(source_str))
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return []

    def _load_from_file(self, path: Path) -> List[Dict[str, Any]]:
        """Load CSV from local file."""
        if not path.exists():
            logger.error(f"CSV file not found: {path}")
            return []

        rows = []
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= self.max_rows:
                    break
                rows.append(dict(row))

        logger.info(f"Loaded {len(rows)} rows from CSV")
        return rows

    def _load_from_url(self, url: str) -> List[Dict[str, Any]]:
        """Load CSV from URL."""
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            content = response.text
            rows = []
            
            reader = csv.DictReader(io.StringIO(content))
            for i, row in enumerate(reader):
                if i >= self.max_rows:
                    break
                rows.append(dict(row))

            logger.info(f"Loaded {len(rows)} rows from URL")
            return rows
        except requests.RequestException as e:
            logger.error(f"Failed to download CSV: {e}")
            return []

    def get_summary(self, rows: List[Dict[str, Any]], source: str = "") -> CSVSummary:
        """
        Generate summary statistics for CSV data.

        Args:
            rows: List of row dictionaries.
            source: Source path/URL for reference.

        Returns:
            CSVSummary object.
        """
        if not rows:
            return CSVSummary(
                row_count=0,
                column_count=0,
                columns=[],
                column_types={},
                sample_rows=[],
                source=source,
            )

        columns = list(rows[0].keys())
        column_types = self._detect_column_types(rows, columns)
        sample_rows = rows[:self.sample_size]

        return CSVSummary(
            row_count=len(rows),
            column_count=len(columns),
            columns=columns,
            column_types=column_types,
            sample_rows=sample_rows,
            source=source,
        )

    def _detect_column_types(
        self,
        rows: List[Dict[str, Any]],
        columns: List[str],
    ) -> Dict[str, str]:
        """Detect data types for each column."""
        types = {}
        sample = rows[:100]  # Sample first 100 rows

        for col in columns:
            values = [r.get(col, "") for r in sample if r.get(col)]
            types[col] = self._infer_type(values)

        return types

    def _infer_type(self, values: List[str]) -> str:
        """Infer data type from sample values."""
        if not values:
            return "empty"

        # Check for numeric
        numeric_count = 0
        float_count = 0
        for v in values:
            v = v.strip().replace(",", "")
            try:
                float(v)
                numeric_count += 1
                if "." in v:
                    float_count += 1
            except ValueError:
                pass

        if numeric_count == len(values):
            return "float" if float_count > 0 else "integer"

        # Check for date patterns
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",
            r"\d{2}/\d{2}/\d{4}",
            r"\d{2}-\d{2}-\d{4}",
        ]
        import re
        for pattern in date_patterns:
            matches = sum(1 for v in values if re.match(pattern, v.strip()))
            if matches >= len(values) * 0.8:
                return "date"

        # Check for boolean
        bool_values = {"true", "false", "yes", "no", "1", "0"}
        if all(v.lower().strip() in bool_values for v in values):
            return "boolean"

        # Default to string
        avg_len = sum(len(v) for v in values) / len(values)
        if avg_len > 100:
            return "text"
        return "string"

    def get_statistics(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics for numeric columns.

        Args:
            rows: List of row dictionaries.

        Returns:
            Statistics dictionary by column.
        """
        if not rows:
            return {}

        stats = {}
        columns = list(rows[0].keys())
        
        for col in columns:
            values = []
            for row in rows:
                val = row.get(col, "")
                if val:
                    try:
                        values.append(float(str(val).replace(",", "")))
                    except ValueError:
                        continue
            
            if values:
                stats[col] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "count": len(values),
                }

        return stats

    def to_documents(
        self,
        rows: List[Dict[str, Any]],
        source: str = "",
        chunk_size: int = 10,
    ) -> List[dict]:
        """
        Convert CSV rows to document format for vector store.

        Args:
            rows: List of row dictionaries.
            source: Source path/URL.
            chunk_size: Number of rows per document chunk.

        Returns:
            List of document dictionaries.
        """
        if not rows:
            return []

        documents = []
        summary = self.get_summary(rows, source)
        
        # First document: summary
        summary_text = self._format_summary(summary)
        documents.append({
            "content": summary_text,
            "metadata": {
                "type": "csv_summary",
                "source": source,
                "row_count": summary.row_count,
                "columns": summary.columns,
            },
        })

        # Create chunks of rows
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i + chunk_size]
            chunk_text = self._format_rows(chunk, summary.columns)
            
            documents.append({
                "content": chunk_text,
                "metadata": {
                    "type": "csv_data",
                    "source": source,
                    "start_row": i,
                    "end_row": i + len(chunk),
                },
            })

        logger.info(f"Created {len(documents)} documents from CSV")
        return documents

    def _format_summary(self, summary: CSVSummary) -> str:
        """Format summary as readable text."""
        lines = [
            f"CSV Dataset Summary",
            f"Source: {summary.source}",
            f"Rows: {summary.row_count}",
            f"Columns: {summary.column_count}",
            "",
            "Column Details:",
        ]
        
        for col in summary.columns:
            col_type = summary.column_types.get(col, "unknown")
            lines.append(f"  - {col} ({col_type})")
        
        lines.append("")
        lines.append("Sample Data:")
        for i, row in enumerate(summary.sample_rows[:3]):
            lines.append(f"  Row {i + 1}: {row}")

        return "\n".join(lines)

    def _format_rows(self, rows: List[Dict[str, Any]], columns: List[str]) -> str:
        """Format rows as readable text."""
        lines = []
        for row in rows:
            row_parts = [f"{col}: {row.get(col, '')}" for col in columns]
            lines.append(" | ".join(row_parts))
        return "\n".join(lines)
