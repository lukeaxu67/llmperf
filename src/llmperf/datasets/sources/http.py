"""HTTP API dataset source for LLMPerf."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from ..types import TestCase, Message
from ..dataset_source import DatasetSource
from ..dataset_source_registry import register_source

logger = logging.getLogger(__name__)


@register_source("http")
class HTTPDatasetSource(DatasetSource):
    """DatasetSource backed by an HTTP API.

    Fetches test cases from an HTTP endpoint.

    Configuration:
        url: API URL (required)
        method: HTTP method (default: GET)
        headers: Request headers
        params: Query parameters
        body: Request body for POST
        timeout: Request timeout in seconds (default: 30)
        data_path: JSONPath to extract data array (default: "")
        pagination_type: Pagination type (offset, cursor, none)
        pagination_param: Pagination parameter name
        page_size: Number of items per page
        max_pages: Maximum pages to fetch
        transform: Transformation rules for API response
    """

    def __init__(self, *, name: str, config: dict[str, object] | None = None):
        super().__init__(name=name, config=config)

        cfg = config or {}
        url_value = cfg.get("url")
        if not url_value:
            raise ValueError("HTTPDatasetSource requires a 'url' configuration value")

        self.url = str(url_value)
        self.method = str(cfg.get("method", "GET")).upper()
        self.headers = cfg.get("headers", {})
        self.params = cfg.get("params", {})
        self.body = cfg.get("body")
        self.timeout = float(cfg.get("timeout", 30))
        self.data_path = cfg.get("data_path", "")
        self.limit = int(cfg["limit"]) if cfg.get("limit") else None

        # Pagination settings
        self.pagination_type = cfg.get("pagination_type", "none")
        self.pagination_param = cfg.get("pagination_param", "offset")
        self.page_size = int(cfg.get("page_size", 100))
        self.max_pages = int(cfg.get("max_pages", 10))

        # Transform settings
        self.transform = cfg.get("transform", {})
        self.id_field = self.transform.get("id_field", "id")
        self.messages_field = self.transform.get("messages_field", "messages")
        self.metadata_fields = self.transform.get("metadata_fields", [])

    def _extract_data(self, response_data: Any) -> List[Dict]:
        """Extract data array from response using data_path.

        Args:
            response_data: Parsed JSON response.

        Returns:
            List of data items.
        """
        if not self.data_path:
            if isinstance(response_data, list):
                return response_data
            return []

        # Navigate data_path
        current = response_data
        for part in str(self.data_path).split("."):
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                current = current[idx] if 0 <= idx < len(current) else None
            else:
                return []

            if current is None:
                return []

        return current if isinstance(current, list) else []

    def _transform_item(self, item: Dict[str, Any]) -> TestCase:
        """Transform API item to TestCase.

        Args:
            item: Raw item from API.

        Returns:
            TestCase instance.
        """
        # Get ID
        test_id = item.get(self.id_field, "")

        # Get messages
        messages_raw = item.get(self.messages_field, [])
        if isinstance(messages_raw, str):
            try:
                messages_raw = json.loads(messages_raw)
            except json.JSONDecodeError:
                messages_raw = []

        # Convert to Message objects
        messages = []
        for m in messages_raw:
            if isinstance(m, dict):
                messages.append(Message(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                ))
            elif isinstance(m, str):
                messages.append(Message(role="user", content=m))

        # Get metadata
        metadata = {}
        if self.metadata_fields:
            for field in self.metadata_fields:
                if field in item:
                    metadata[field] = item[field]
        else:
            # Include all fields except id and messages
            for key, value in item.items():
                if key not in [self.id_field, self.messages_field]:
                    metadata[key] = value

        return TestCase(
            id=str(test_id) if test_id else None,
            messages=messages,
            metadata=metadata if metadata else None,
        )

    def _fetch_page(
        self,
        page: int,
        cursor: Optional[str] = None,
    ) -> tuple[List[Dict], Optional[str]]:
        """Fetch a single page of data.

        Args:
            page: Page number.
            cursor: Pagination cursor.

        Returns:
            Tuple of (items, next_cursor).
        """
        params = dict(self.params)

        if self.pagination_type == "offset":
            params[self.pagination_param] = page * self.page_size
            params["limit"] = self.page_size
        elif self.pagination_type == "cursor" and cursor:
            params[self.pagination_param] = cursor

        try:
            with httpx.Client(timeout=self.timeout) as client:
                if self.method == "GET":
                    response = client.get(
                        self.url,
                        params=params,
                        headers=self.headers,
                    )
                else:
                    response = client.request(
                        self.method,
                        self.url,
                        params=params,
                        json=self.body,
                        headers=self.headers,
                    )

                response.raise_for_status()
                data = response.json()

                items = self._extract_data(data)

                # Get next cursor for cursor-based pagination
                next_cursor = None
                if self.pagination_type == "cursor":
                    next_cursor = data.get("next_cursor") or data.get("cursor")
                    if not next_cursor and isinstance(data.get("pagination"), dict):
                        next_cursor = data["pagination"].get("next_cursor")

                return items, next_cursor

        except httpx.HTTPError as e:
            logger.error("HTTP request failed: %s", e)
            return [], None

    def load(self) -> List[TestCase]:
        """Load test cases from HTTP API.

        Returns:
            List of TestCase instances.
        """
        all_items: List[Dict] = []
        cursor = None

        if self.pagination_type == "none":
            # Single request
            items, _ = self._fetch_page(0)
            all_items.extend(items)
        else:
            # Paginated requests
            for page in range(self.max_pages):
                items, cursor = self._fetch_page(page, cursor)
                if not items:
                    break

                all_items.extend(items)

                if self.limit and len(all_items) >= self.limit:
                    all_items = all_items[:self.limit]
                    break

                if self.pagination_type == "cursor" and not cursor:
                    break

        # Transform items
        test_cases = []
        for item in all_items:
            try:
                test_case = self._transform_item(item)
                if test_case.messages:
                    test_cases.append(test_case)
            except Exception as e:
                logger.warning("Failed to transform item: %s", e)
                continue

        return test_cases
