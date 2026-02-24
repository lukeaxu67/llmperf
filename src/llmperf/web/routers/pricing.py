"""Pricing management API router."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llmperf.config.runtime import load_runtime_config
from llmperf.records.storage import Storage

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response models
class PricingCreateRequest(BaseModel):
    """Request to create a new pricing record."""
    provider: str = Field(..., description="Provider name (e.g., openai, zhipu)")
    model: str = Field(..., description="Model name")
    input_price: float = Field(..., description="Input price per million tokens (CNY)")
    output_price: float = Field(..., description="Output price per million tokens (CNY)")
    cache_read_price: float = Field(0.0, description="Cache read price per million tokens (CNY)")
    cache_write_price: float = Field(0.0, description="Cache write price per million tokens (CNY)")
    effective_at: Optional[int] = Field(None, description="Effective timestamp (defaults to now)")
    note: str = Field("", description="Optional note")


class PricingResponse(BaseModel):
    """Response for pricing operations."""
    id: int
    provider: str
    model: str
    input_price: float
    output_price: float
    cache_read_price: float = 0.0
    cache_write_price: float = 0.0
    effective_at: int
    created_at: int
    note: str = ""


class PricingListResponse(BaseModel):
    """Response for pricing list."""
    items: List[PricingResponse]
    total: int


class ProviderModelSummary(BaseModel):
    """Summary of a provider/model with latest pricing."""
    provider: str
    model: str
    input_price: float
    output_price: float
    effective_at: int


class CostSummaryItem(BaseModel):
    """Cost summary for a provider/model."""
    provider: str
    model: str
    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float


class TotalCostResponse(BaseModel):
    """Total cost across all runs."""
    total_cost: float
    run_count: int
    currency: str


def get_storage() -> Storage:
    """Get storage instance."""
    runtime = load_runtime_config()
    return Storage(str(runtime.db_path))


@router.post(
    "",
    response_model=PricingResponse,
    summary="Add pricing record",
)
async def add_pricing(request: PricingCreateRequest):
    """Add a new pricing record for a provider/model."""
    storage = get_storage()

    try:
        record = storage.add_pricing(
            provider=request.provider,
            model=request.model,
            input_price=request.input_price,
            output_price=request.output_price,
            cache_read_price=request.cache_read_price,
            cache_write_price=request.cache_write_price,
            effective_at=request.effective_at,
            note=request.note,
        )
        return PricingResponse(
            id=record.id,
            provider=record.provider,
            model=record.model,
            input_price=record.input_price,
            output_price=record.output_price,
            cache_read_price=record.cache_read_price,
            cache_write_price=record.cache_write_price,
            effective_at=record.effective_at,
            created_at=record.created_at,
            note=record.note,
        )
    except Exception as e:
        logger.error("Failed to add pricing: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "",
    response_model=PricingListResponse,
    summary="List pricing records",
)
async def list_pricing(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    limit: int = 100,
):
    """List pricing records, optionally filtered by provider/model."""
    storage = get_storage()
    records = storage.list_pricing(provider=provider, model=model, limit=limit)

    return PricingListResponse(
        items=[
            PricingResponse(
                id=r.id,
                provider=r.provider,
                model=r.model,
                input_price=r.input_price,
                output_price=r.output_price,
                cache_read_price=r.cache_read_price,
                cache_write_price=r.cache_write_price,
                effective_at=r.effective_at,
                created_at=r.created_at,
                note=r.note,
            )
            for r in records
        ],
        total=len(records),
    )


@router.get(
    "/providers",
    response_model=List[ProviderModelSummary],
    summary="List providers and models",
)
async def list_providers():
    """List all unique provider/model combinations with latest pricing."""
    storage = get_storage()
    return storage.get_providers_models()


@router.get(
    "/{pricing_id}",
    response_model=PricingResponse,
    summary="Get pricing record",
)
async def get_pricing(pricing_id: int):
    """Get a specific pricing record."""
    storage = get_storage()
    record = storage.get_pricing(pricing_id)

    if not record:
        raise HTTPException(status_code=404, detail="Pricing record not found")

    return PricingResponse(
        id=record.id,
        provider=record.provider,
        model=record.model,
        input_price=record.input_price,
        output_price=record.output_price,
        cache_read_price=record.cache_read_price,
        cache_write_price=record.cache_write_price,
        effective_at=record.effective_at,
        created_at=record.created_at,
        note=record.note,
    )


@router.delete(
    "/{pricing_id}",
    summary="Delete pricing record",
)
async def delete_pricing(pricing_id: int):
    """Delete a pricing record."""
    storage = get_storage()
    success = storage.delete_pricing(pricing_id)

    if not success:
        raise HTTPException(status_code=404, detail="Pricing record not found")

    return {"message": "Pricing record deleted", "id": pricing_id}


@router.get(
    "/history/chart",
    summary="Get pricing history for charts",
)
async def get_pricing_history(
    provider: Optional[str] = None,
    days: int = 30,
):
    """Get pricing history for visualization."""
    storage = get_storage()
    return storage.get_pricing_history_by_provider(provider=provider, days=days)


@router.get(
    "/cost/summary",
    response_model=List[CostSummaryItem],
    summary="Get cost summary by provider/model",
)
async def get_cost_summary(days: int = 30):
    """Get cost summary grouped by provider and model."""
    storage = get_storage()
    return storage.get_cost_summary_by_provider(days=days)


@router.get(
    "/cost/total",
    response_model=TotalCostResponse,
    summary="Get total cost",
)
async def get_total_cost():
    """Get total cost across all runs."""
    storage = get_storage()
    return storage.get_total_cost()


class CurrentPriceQueryResponse(BaseModel):
    """Response for current price query."""
    provider: str
    model: str
    input_price: float  # CNY per million tokens
    output_price: float  # CNY per million tokens
    cache_read_price: float = 0.0
    cache_write_price: float = 0.0
    currency: str = "CNY"
    found: bool = True  # Whether pricing was found


@router.get(
    "/current",
    response_model=CurrentPriceQueryResponse,
    summary="Get current price for provider/model",
)
async def get_current_price(
    provider: str,
    model: str,
):
    """Get the current effective price for a provider/model combination.

    This returns the latest pricing from the pricing_history table.
    If no pricing is found, returns default zero pricing with found=False.
    """
    storage = get_storage()

    price_record = storage.get_pricing_at_time(
        provider=provider,
        model=model,
        timestamp=int(time.time()),
    )

    if price_record:
        return CurrentPriceQueryResponse(
            provider=provider,
            model=model,
            input_price=price_record.input_price,
            output_price=price_record.output_price,
            cache_read_price=price_record.cache_read_price,
            cache_write_price=price_record.cache_write_price,
            currency="CNY",
            found=True,
        )
    else:
        return CurrentPriceQueryResponse(
            provider=provider,
            model=model,
            input_price=0.0,
            output_price=0.0,
            cache_read_price=0.0,
            cache_write_price=0.0,
            currency="CNY",
            found=False,
        )
