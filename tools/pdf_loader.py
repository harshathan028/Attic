"""
PDF Loader - Extract and chunk text from PDF documents.

Provides text extraction from PDFs with intelligent chunking
for vector store ingestion.
"""

import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import requests

logger = logging.getLogger(__name__)

# Try to import PDF libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. Install with: pip install pymupdf")


@dataclass
class PDFChunk:
    """Structured PDF text chunk."""
    content: str
    page_number: int
    chunk_index: int
    source: str


class PDFLoader:
    """
    PDF document loader with text extraction and chunking.
    
    Extracts text from PDF files or URLs and chunks into
    manageable segments for vector store ingestion.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        timeout: int = 60,
    ):
        """
        Initialize PDF loader.

        Args:
            chunk_size: Target size for text chunks (characters).
            chunk_overlap: Overlap between chunks for context preservation.
            timeout: Download timeout for URL-based PDFs.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.timeout = timeout
        
        if not PYMUPDF_AVAILABLE:
            logger.warning("PDFLoader initialized but PyMuPDF not available")
        else:
            logger.info("Initialized PDFLoader")

    def load(self, source: Union[str, Path]) -> List[PDFChunk]:
        """
        Load and extract text from PDF.

        Args:
            source: File path or URL to PDF.

        Returns:
            List of PDFChunk objects.
        """
        if not PYMUPDF_AVAILABLE:
            logger.error("Cannot load PDF: PyMuPDF not installed")
            return []

        source_str = str(source)
        logger.info(f"Loading PDF: {source_str}")

        try:
            if source_str.startswith(("http://", "https://")):
                return self._load_from_url(source_str)
            else:
                return self._load_from_file(Path(source_str))
        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            return []

    def _load_from_file(self, path: Path) -> List[PDFChunk]:
        """Load PDF from local file."""
        if not path.exists():
            logger.error(f"PDF file not found: {path}")
            return []

        doc = fitz.open(path)
        return self._extract_chunks(doc, str(path))

    def _load_from_url(self, url: str) -> List[PDFChunk]:
        """Load PDF from URL."""
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            pdf_data = io.BytesIO(response.content)
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            return self._extract_chunks(doc, url)
        except requests.RequestException as e:
            logger.error(f"Failed to download PDF: {e}")
            return []

    def _extract_chunks(self, doc, source: str) -> List[PDFChunk]:
        """Extract text and create chunks from PDF document."""
        chunks = []
        chunk_index = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if not text.strip():
                continue

            # Clean text
            text = self._clean_text(text)
            
            # Create chunks from page text
            page_chunks = self._create_chunks(text)
            
            for chunk_text in page_chunks:
                chunks.append(PDFChunk(
                    content=chunk_text,
                    page_number=page_num + 1,
                    chunk_index=chunk_index,
                    source=source,
                ))
                chunk_index += 1

        doc.close()
        logger.info(f"Extracted {len(chunks)} chunks from PDF")
        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean extracted PDF text."""
        # Remove excessive whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        
        # Remove page headers/footers patterns (common patterns)
        text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
        
        return text.strip()

    def _create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 20% of chunk
                search_start = start + int(self.chunk_size * 0.8)
                best_break = end
                
                for pattern in [". ", ".\n", "? ", "!\n", "\n\n"]:
                    pos = text.rfind(pattern, search_start, end)
                    if pos > search_start:
                        best_break = pos + len(pattern)
                        break
                
                end = best_break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def to_documents(self, chunks: List[PDFChunk]) -> List[dict]:
        """
        Convert PDF chunks to document format for vector store.

        Args:
            chunks: List of PDFChunk objects.

        Returns:
            List of document dictionaries.
        """
        documents = []
        for chunk in chunks:
            doc = {
                "content": chunk.content,
                "metadata": {
                    "type": "pdf",
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                },
            }
            documents.append(doc)
        return documents

    def get_summary(self, chunks: List[PDFChunk]) -> dict:
        """
        Get summary statistics for loaded PDF.

        Args:
            chunks: List of PDFChunk objects.

        Returns:
            Summary dictionary.
        """
        if not chunks:
            return {"pages": 0, "chunks": 0, "total_chars": 0}

        pages = set(c.page_number for c in chunks)
        total_chars = sum(len(c.content) for c in chunks)

        return {
            "pages": len(pages),
            "chunks": len(chunks),
            "total_chars": total_chars,
            "avg_chunk_size": total_chars // len(chunks) if chunks else 0,
            "source": chunks[0].source if chunks else None,
        }
