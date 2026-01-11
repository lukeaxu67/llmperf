from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, AliasChoices


class DatasetIteratorOptions(BaseModel):
    mutation_chain: Optional[List[str]] = Field(default_factory=list)
    max_total_seconds: Optional[int] = Field(None, ge=0)
    max_rounds: Optional[int] = Field(None, ge=1)


class DatasetSourceConfig(BaseModel):
    type: str = Field(min_length=1)
    name: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    source: DatasetSourceConfig
    iterator: Optional[DatasetIteratorOptions] = None


class RateConfig(BaseModel):
    qps: float | None = Field(None, gt=0)
    interval_seconds: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _validate_oneof(self) -> "ExecutorConfig.RateConfig":
        if self.qps is not None and self.interval_seconds is not None:
            raise ValueError("Specify only one of qps or interval_seconds")
        return self


class ExecutorConfig(BaseModel):

    id: str
    name: str
    type: str
    impl: str = "chat"
    after: List[str] = Field(default_factory=list)
    concurrency: int = 1
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    param: Dict[str, Any] = Field(default_factory=dict)
    rate: RateConfig | None = Field(default=None, validation_alias=AliasChoices("rate", "limiter"))


class PricingEntry(BaseModel):
    provider: str
    model: str
    unit: Literal["per_1k", "per_1m"] = "per_1k"
    input_price: float = Field(..., ge=0)
    output_price: float = Field(..., ge=0)
    cache_input_discount: float = Field(0.0, ge=0.0, le=1.0)
    cache_output_discount: float = Field(0.0, ge=0.0, le=1.0)
    currency: str = "CNY"

    @model_validator(mode="after")
    def _validate_cache_discounts(self) -> "PricingEntry":
        fields_set = getattr(self, "model_fields_set", set())
        has_input = "cache_input_discount" in fields_set
        has_output = "cache_output_discount" in fields_set
        if has_input ^ has_output:
            raise ValueError("cache discounts must define both input and output ratios")
        return self


class MultiprocessConfig(BaseModel):
    """Configuration for multi-process execution of executors."""

    # If true, run each executor in its own process (default); otherwise, single-process mode.
    per_executor: bool = True
    # Optional cap on the number of worker processes; defaults to CPU count when omitted.
    max_workers: Optional[int] = None


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    info: str = ""
    db_path: Optional[str] = None
    dataset: DatasetConfig
    executors: List[ExecutorConfig]
    pricing: List[PricingEntry] = Field(default_factory=list)
    multiprocess: Optional[MultiprocessConfig] = None
