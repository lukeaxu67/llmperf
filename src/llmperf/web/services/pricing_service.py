"""Pricing service for auto-fetching current prices."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from llmperf.config.runtime import load_runtime_config
from llmperf.records.storage import Storage

logger = logging.getLogger(__name__)


@dataclass
class PriceInfo:
    """Price information for a provider/model."""
    provider: str
    model: str
    input_price: float  # CNY per million tokens
    output_price: float  # CNY per million tokens
    cache_read_price: float = 0.0  # CNY per million tokens
    cache_write_price: float = 0.0  # CNY per million tokens
    currency: str = "CNY"

    @classmethod
    def default(cls, provider: str, model: str) -> "PriceInfo":
        """Create default price info (zero pricing)."""
        return cls(
            provider=provider,
            model=model,
            input_price=0.0,
            output_price=0.0,
            cache_read_price=0.0,
            cache_write_price=0.0,
            currency="CNY",
        )


class PricingService:
    """Service for fetching and caching pricing information.

    This service automatically fetches the latest pricing from the database
    based on provider and model combination. Results are cached to avoid
    frequent database queries.
    """

    # Cache TTL in seconds (5 minutes)
    CACHE_TTL = 300

    def __init__(self, db_path: Optional[str] = None):
        """Initialize pricing service.

        Args:
            db_path: Path to database file. If None, uses runtime config.
        """
        if db_path is None:
            runtime = load_runtime_config()
            db_path = str(runtime.db_path)

        self._storage = Storage(db_path)
        self._cache: Dict[Tuple[str, str], Tuple[PriceInfo, float]] = {}
        # Cache stores: (provider, model) -> (PriceInfo, timestamp)

    def get_current_price(self, provider: str, model: str) -> PriceInfo:
        """Get the current effective price for a provider/model.

        This method checks the cache first, and if the cached entry is stale
        or missing, queries the database for the latest pricing.

        Args:
            provider: Provider name (e.g., "openai", "zhipu")
            model: Model name (e.g., "gpt-4", "glm-4")

        Returns:
            PriceInfo with current pricing, or default zero pricing if not found.
        """
        key = (provider, model)
        now = time.time()

        # Check cache
        if key in self._cache:
            price_info, cached_at = self._cache[key]
            if now - cached_at < self.CACHE_TTL:
                return price_info

        # Cache miss or expired, fetch from database
        price_record = self._storage.get_pricing_at_time(
            provider=provider,
            model=model,
            timestamp=int(now),
        )

        if price_record:
            price_info = PriceInfo(
                provider=provider,
                model=model,
                input_price=price_record.input_price,
                output_price=price_record.output_price,
                cache_read_price=price_record.cache_read_price,
                cache_write_price=price_record.cache_write_price,
                currency="CNY",
            )
        else:
            logger.warning(
                "No pricing found for provider=%s model=%s, using default (zero) pricing",
                provider,
                model,
            )
            price_info = PriceInfo.default(provider, model)

        # Update cache
        self._cache[key] = (price_info, now)
        return price_info

    def invalidate_cache(self, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        """Invalidate cached pricing entries.

        Args:
            provider: If specified, only invalidate entries for this provider.
            model: If specified with provider, only invalidate this specific entry.
        """
        if provider is None and model is None:
            self._cache.clear()
        elif model is not None and provider is not None:
            self._cache.pop((provider, model), None)
        elif provider is not None:
            keys_to_remove = [k for k in self._cache.keys() if k[0] == provider]
            for key in keys_to_remove:
                del self._cache[key]

    def compute_cost(
        self,
        provider: str,
        model: str,
        qtokens: int,
        atokens: int,
        ctokens: int = 0,
    ) -> Tuple[float, float, float, float, str]:
        """Compute cost breakdown based on current pricing.

        Args:
            provider: Provider name
            model: Model name
            qtokens: Query/input tokens
            atokens: Answer/output tokens
            ctokens: Cached tokens

        Returns:
            Tuple of (prompt_cost, completion_cost, cache_cost, total_cost, currency)
        """
        price = self.get_current_price(provider, model)

        # Prices are per million tokens
        prompt_tokens = max(qtokens - ctokens, 0)
        prompt_cost = (prompt_tokens / 1_000_000.0) * price.input_price
        completion_cost = (atokens / 1_000_000.0) * price.output_price
        cache_cost = (ctokens / 1_000_000.0) * price.cache_read_price
        total_cost = prompt_cost + completion_cost + cache_cost

        return prompt_cost, completion_cost, cache_cost, total_cost, price.currency


# Global singleton instance
_pricing_service: Optional[PricingService] = None


def get_pricing_service() -> PricingService:
    """Get the global pricing service instance."""
    global _pricing_service
    if _pricing_service is None:
        _pricing_service = PricingService()
    return _pricing_service
