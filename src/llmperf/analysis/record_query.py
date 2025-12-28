from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from llmperf.config.runtime import load_runtime_config
from llmperf.records.storage import Storage


class RecordQuery(BaseModel):

    db_path: Optional[str] = None

    run_ids: list[str] = Field(default_factory=list)

    provider: Optional[str] = None
    model: Optional[str] = None

    start_ts: Optional[int] = None
    end_ts: Optional[int] = None

    def storage(self) -> Storage:
        runtime = load_runtime_config()
        return Storage(self.db_path or str(runtime.db_path))
