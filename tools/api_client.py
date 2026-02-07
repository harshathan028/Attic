"""
API Client - REST API wrapper for live data ingestion.

Provides a simple interface for fetching JSON data from REST APIs.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Structured API response."""
    success: bool
    status_code: int
    data: Optional[Union[Dict, List]]
    error: Optional[str]
    url: str
    response_time: float


class APIClient:
    """
    REST API client with retry handling and response parsing.
    
    Supports GET requests to JSON APIs with configurable timeouts
    and automatic retry logic.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize API client.

        Args:
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts on failure.
            retry_delay: Base delay between retries (exponential backoff).
            default_headers: Default headers for all requests.
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.default_headers = default_headers or {
            "Accept": "application/json",
            "User-Agent": "AIContentFactory/1.0",
        }
        logger.info("Initialized APIClient")

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> APIResponse:
        """
        Make GET request to API endpoint.

        Args:
            url: API endpoint URL.
            params: Query parameters.
            headers: Additional headers (merged with defaults).

        Returns:
            APIResponse with parsed data or error.
        """
        logger.info(f"API GET request: {url}")
        
        request_headers = {**self.default_headers, **(headers or {})}
        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=request_headers,
                    timeout=self.timeout,
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        logger.info(f"API request successful: {response.status_code}")
                        return APIResponse(
                            success=True,
                            status_code=response.status_code,
                            data=data,
                            error=None,
                            url=url,
                            response_time=response_time,
                        )
                    except json.JSONDecodeError as e:
                        return APIResponse(
                            success=False,
                            status_code=response.status_code,
                            data=None,
                            error=f"JSON decode error: {e}",
                            url=url,
                            response_time=response_time,
                        )
                else:
                    last_error = f"HTTP {response.status_code}"
                    logger.warning(f"API request failed: {last_error}")
                    
                    # Don't retry on client errors (4xx)
                    if 400 <= response.status_code < 500:
                        return APIResponse(
                            success=False,
                            status_code=response.status_code,
                            data=None,
                            error=last_error,
                            url=url,
                            response_time=response_time,
                        )

            except requests.Timeout:
                last_error = "Request timeout"
                logger.warning(f"API timeout (attempt {attempt + 1})")
            except requests.ConnectionError as e:
                last_error = f"Connection error: {e}"
                logger.warning(f"API connection error (attempt {attempt + 1})")
            except requests.RequestException as e:
                last_error = str(e)
                logger.warning(f"API request error (attempt {attempt + 1}): {e}")

            if attempt < self.max_retries - 1:
                sleep_time = self.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        response_time = time.time() - start_time
        return APIResponse(
            success=False,
            status_code=0,
            data=None,
            error=last_error or "Unknown error",
            url=url,
            response_time=response_time,
        )

    def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> APIResponse:
        """
        Make POST request to API endpoint.

        Args:
            url: API endpoint URL.
            data: Form data.
            json_data: JSON body data.
            headers: Additional headers.

        Returns:
            APIResponse with parsed data or error.
        """
        logger.info(f"API POST request: {url}")
        
        request_headers = {**self.default_headers, **(headers or {})}
        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    data=data,
                    json=json_data,
                    headers=request_headers,
                    timeout=self.timeout,
                )
                response_time = time.time() - start_time
                
                if response.status_code in (200, 201):
                    try:
                        resp_data = response.json()
                        return APIResponse(
                            success=True,
                            status_code=response.status_code,
                            data=resp_data,
                            error=None,
                            url=url,
                            response_time=response_time,
                        )
                    except json.JSONDecodeError:
                        return APIResponse(
                            success=True,
                            status_code=response.status_code,
                            data={"text": response.text},
                            error=None,
                            url=url,
                            response_time=response_time,
                        )
                else:
                    last_error = f"HTTP {response.status_code}"
                    if 400 <= response.status_code < 500:
                        return APIResponse(
                            success=False,
                            status_code=response.status_code,
                            data=None,
                            error=last_error,
                            url=url,
                            response_time=response_time,
                        )

            except requests.RequestException as e:
                last_error = str(e)
                logger.warning(f"API POST error (attempt {attempt + 1}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (2 ** attempt))

        response_time = time.time() - start_time
        return APIResponse(
            success=False,
            status_code=0,
            data=None,
            error=last_error,
            url=url,
            response_time=response_time,
        )

    def to_documents(self, response: APIResponse, flatten: bool = True) -> List[dict]:
        """
        Convert API response to document format for vector store.

        Args:
            response: APIResponse object.
            flatten: Whether to flatten nested structures.

        Returns:
            List of document dictionaries.
        """
        if not response.success or not response.data:
            return []

        documents = []
        data = response.data

        # Handle list of items
        if isinstance(data, list):
            for i, item in enumerate(data):
                doc = self._item_to_document(item, response.url, i)
                if doc:
                    documents.append(doc)
        # Handle single object
        elif isinstance(data, dict):
            # Check for common result patterns
            items = data.get("results") or data.get("data") or data.get("items") or [data]
            if isinstance(items, list):
                for i, item in enumerate(items):
                    doc = self._item_to_document(item, response.url, i)
                    if doc:
                        documents.append(doc)
            else:
                doc = self._item_to_document(data, response.url, 0)
                if doc:
                    documents.append(doc)

        return documents

    def _item_to_document(self, item: Any, source: str, index: int) -> Optional[dict]:
        """Convert single item to document."""
        if isinstance(item, dict):
            # Extract text content
            content_parts = []
            for key in ["title", "name", "description", "content", "text", "body", "summary"]:
                if key in item and item[key]:
                    content_parts.append(str(item[key]))
            
            if not content_parts:
                content_parts = [json.dumps(item, indent=2)]
            
            return {
                "content": "\n\n".join(content_parts),
                "metadata": {
                    "type": "api",
                    "source": source,
                    "index": index,
                    "raw_data": item,
                },
            }
        elif isinstance(item, str):
            return {
                "content": item,
                "metadata": {
                    "type": "api",
                    "source": source,
                    "index": index,
                },
            }
        return None
