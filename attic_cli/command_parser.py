"""
Command Parser - Natural language to pipeline configuration.

Parses user prompts and extracts intent, data sources, and topics.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class CLIConfig:
    """Pipeline configuration derived from natural language prompt."""
    topic: str
    live_data_enabled: bool = False
    data_source_url: Optional[str] = None
    data_file: Optional[str] = None
    api_endpoint: Optional[str] = None
    learning_enabled: bool = True
    output_file: Optional[str] = None
    
    # Metadata
    detected_intent: str = "research"
    detected_sources: List[str] = field(default_factory=list)
    original_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for pipeline."""
        return {
            "topic": self.topic,
            "live_data_enabled": self.live_data_enabled,
            "data_source_url": self.data_source_url,
            "data_file": self.data_file,
            "api_endpoint": self.api_endpoint,
            "learning_enabled": self.learning_enabled,
        }


class CommandParser:
    """
    Natural language command parser for ATTIC CLI.
    
    Converts freeform user prompts into structured pipeline
    configurations by detecting intent and data sources.
    """

    # Intent detection patterns
    INTENT_PATTERNS = {
        "summarize": [
            r"\bsummar",
            r"\bdigest\b",
            r"\boverview\b",
            r"\bbrief\b",
            r"\btldr\b",
        ],
        "analyze": [
            r"\banalyz",
            r"\bexamine\b",
            r"\binspect\b",
            r"\breview\b",
            r"\baudit\b",
        ],
        "research": [
            r"\bresearch\b",
            r"\binvestigat",
            r"\bexplor",
            r"\bfind\b",
            r"\bdiscover\b",
            r"\blearn about\b",
        ],
        "write": [
            r"\bwrite\b",
            r"\bcreate\b",
            r"\bgenerate\b",
            r"\bdraft\b",
            r"\bcompose\b",
            r"\breport\b",
        ],
        "explain": [
            r"\bexplain\b",
            r"\bdescrib",
            r"\bwhat is\b",
            r"\bhow does\b",
            r"\bwhy\b",
        ],
    }

    # Live data trigger keywords
    LIVE_DATA_KEYWORDS = [
        "rss", "news", "latest", "feed", "api", "csv", "pdf",
        "dataset", "file", "report", "data", "from", "source",
        "fetch", "load", "import", "read", "ingest",
    ]

    # Source type patterns
    SOURCE_PATTERNS = {
        "rss": [r"\brss\b", r"\bfeed\b", r"\bnews\b", r"\bxml\b"],
        "api": [r"\bapi\b", r"\bendpoint\b", r"\bjson\b", r"\brest\b"],
        "csv": [r"\.csv\b", r"\bcsv\b", r"\bspreadsheet\b", r"\bdataset\b"],
        "pdf": [r"\.pdf\b", r"\bpdf\b", r"\bdocument\b"],
    }

    # Words to strip from topic
    STRIP_WORDS = [
        "summarize", "analyze", "research", "write", "explain",
        "create", "generate", "about", "from", "using", "with",
        "the", "a", "an", "this", "that", "these", "those",
        "rss", "feed", "api", "csv", "pdf", "file", "data",
        "latest", "news", "report", "please", "can", "you",
        "dataset", "source", "load", "fetch", "read", "import",
    ]

    def __init__(self):
        """Initialize the command parser."""
        logger.debug("Initialized CommandParser")

    def parse(self, prompt: str) -> CLIConfig:
        """
        Parse a natural language prompt into pipeline config.

        Args:
            prompt: User's freeform prompt.

        Returns:
            CLIConfig with detected settings.
        """
        prompt_lower = prompt.lower().strip()
        
        # Detect intent
        intent = self._detect_intent(prompt_lower)
        
        # Check for live data mode
        live_data = self._needs_live_data(prompt_lower)
        
        # Extract sources
        sources = self._detect_sources(prompt_lower)
        
        # Extract URLs
        url, url_type = self._extract_url(prompt)
        
        # Extract file paths
        file_path, file_type = self._extract_file_path(prompt)
        
        # Extract the topic
        topic = self._extract_topic(prompt, sources)
        
        # Build config
        config = CLIConfig(
            topic=topic,
            live_data_enabled=live_data or bool(url) or bool(file_path),
            detected_intent=intent,
            detected_sources=sources,
            original_prompt=prompt,
            learning_enabled=True,
        )
        
        # Set appropriate source field
        if url:
            if url_type == "api" or "api" in sources:
                config.api_endpoint = url
            else:
                config.data_source_url = url
        
        if file_path:
            config.data_file = file_path
        
        logger.info(
            f"Parsed prompt: intent={intent}, topic='{topic}', "
            f"live_data={config.live_data_enabled}, sources={sources}"
        )
        
        return config

    def _detect_intent(self, prompt: str) -> str:
        """
        Detect the user's intent from the prompt.

        Args:
            prompt: Lowercase prompt string.

        Returns:
            Intent category string.
        """
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, prompt):
                    return intent
        
        # Default to research if no clear intent
        return "research"

    def _needs_live_data(self, prompt: str) -> bool:
        """
        Check if the prompt requires live data ingestion.

        Args:
            prompt: Lowercase prompt string.

        Returns:
            True if live data is needed.
        """
        return any(kw in prompt for kw in self.LIVE_DATA_KEYWORDS)

    def _detect_sources(self, prompt: str) -> List[str]:
        """
        Detect data source types mentioned in prompt.

        Args:
            prompt: Lowercase prompt string.

        Returns:
            List of detected source types.
        """
        sources = []
        
        for source_type, patterns in self.SOURCE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, prompt):
                    if source_type not in sources:
                        sources.append(source_type)
                    break
        
        return sources

    def _extract_url(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract URL from prompt if present.

        Args:
            prompt: Original prompt string.

        Returns:
            Tuple of (url, url_type) or (None, None).
        """
        # URL regex
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        
        match = re.search(url_pattern, prompt)
        if match:
            url = match.group(0)
            
            # Determine URL type
            parsed = urlparse(url)
            path_lower = parsed.path.lower()
            
            if path_lower.endswith(".pdf"):
                return url, "pdf"
            elif path_lower.endswith(".csv"):
                return url, "csv"
            elif "rss" in url.lower() or path_lower.endswith(".xml"):
                return url, "rss"
            elif "api" in url.lower() or parsed.path.startswith("/api"):
                return url, "api"
            else:
                # Assume RSS for generic URLs in data context
                return url, "rss"
        
        return None, None

    def _extract_file_path(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract file path from prompt if present.

        Args:
            prompt: Original prompt string.

        Returns:
            Tuple of (path, file_type) or (None, None).
        """
        # File path patterns
        patterns = [
            r'(["\'])(.+?\.(csv|pdf|json|txt|xlsx))\1',  # Quoted paths
            r'\b(\S+\.(csv|pdf|json|txt|xlsx))\b',  # Unquoted paths
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                # Get the path (different groups for different patterns)
                if match.lastindex >= 2:
                    path = match.group(2) if match.group(1) in ["'", '"'] else match.group(1)
                else:
                    path = match.group(1)
                
                # Get extension
                ext = Path(path).suffix.lower().lstrip(".")
                
                # Check if file exists
                if not Path(path).exists():
                    # Try in current directory
                    if not Path(Path.cwd() / path).exists():
                        logger.warning(f"File not found: {path}")
                
                return path, ext
        
        return None, None

    def _extract_topic(self, prompt: str, detected_sources: List[str]) -> str:
        """
        Extract the core topic from the prompt.

        Args:
            prompt: Original prompt string.
            detected_sources: List of detected source types.

        Returns:
            Extracted topic string.
        """
        # Remove URLs
        topic = re.sub(r'https?://\S+', '', prompt)
        
        # Remove file paths
        topic = re.sub(r'["\']?\S+\.(csv|pdf|json|txt|xlsx)["\']?', '', topic, flags=re.IGNORECASE)
        
        # Remove strip words
        words = topic.split()
        filtered = []
        
        for word in words:
            clean = word.lower().strip(".,!?;:'\"")
            if clean not in self.STRIP_WORDS and len(clean) > 1:
                filtered.append(word)
        
        topic = " ".join(filtered).strip()
        
        # Clean up extra spaces
        topic = re.sub(r'\s+', ' ', topic)
        
        # If topic is too short, use original without URLs
        if len(topic) < 5:
            topic = re.sub(r'https?://\S+', '', prompt)
            topic = re.sub(r'["\']?\S+\.(csv|pdf|json|txt|xlsx)["\']?', '', topic, flags=re.IGNORECASE)
            topic = topic.strip()
        
        return topic if topic else "General content"

    def validate_config(self, config: CLIConfig) -> Tuple[bool, Optional[str]]:
        """
        Validate a parsed configuration.

        Args:
            config: CLIConfig to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        # Check topic
        if not config.topic or len(config.topic) < 2:
            return False, "Could not extract a valid topic from your prompt"
        
        # Check file exists if specified
        if config.data_file:
            path = Path(config.data_file)
            if not path.exists() and not (Path.cwd() / path).exists():
                return False, f"File not found: {config.data_file}"
        
        # Check URL is valid if specified
        if config.data_source_url:
            try:
                result = urlparse(config.data_source_url)
                if not all([result.scheme, result.netloc]):
                    return False, f"Invalid URL: {config.data_source_url}"
            except Exception:
                return False, f"Invalid URL: {config.data_source_url}"
        
        if config.api_endpoint:
            try:
                result = urlparse(config.api_endpoint)
                if not all([result.scheme, result.netloc]):
                    return False, f"Invalid API endpoint: {config.api_endpoint}"
            except Exception:
                return False, f"Invalid API endpoint: {config.api_endpoint}"
        
        return True, None

    def get_help_text(self) -> str:
        """
        Get help text for prompt formatting.

        Returns:
            Help string.
        """
        return """
╭─ ATTIC Prompt Guide ─────────────────────────────────────────╮
│                                                              │
│  Just type naturally! ATTIC understands:                     │
│                                                              │
│  📝 ACTIONS:                                                 │
│     summarize, analyze, research, write, explain             │
│                                                              │
│  📡 DATA SOURCES:                                            │
│     • RSS feeds:  "summarize AI news from rss"               │
│     • CSV files:  "analyze sales.csv"                        │
│     • PDF docs:   "summarize report.pdf"                     │
│     • APIs:       "fetch data from https://api.example.com"  │
│                                                              │
│  💡 EXAMPLES:                                                │
│     summarize latest ai chip news                            │
│     analyze this csv sales.csv                               │
│     research climate policy from rss                         │
│     write report from pdf market.pdf                         │
│     explain quantum computing                                │
│                                                              │
╰──────────────────────────────────────────────────────────────╯
"""
