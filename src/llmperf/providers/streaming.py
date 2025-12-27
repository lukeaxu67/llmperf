from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional

from ..records.model import RunRecord, now_ms


StreamEventKind = Literal["reasoning", "content", "tool"]


def is_stream_debug_enabled() -> bool:
    return os.environ.get("LLMPERF_STREAM_DEBUG", "").strip().lower() in {"1", "true", "yes"}


@dataclass
class ProviderStreamEvent:
    kind: StreamEventKind
    text: str
    ts_ms: int
    seq: int = 0


@dataclass
class StreamAccumulator:
    record: RunRecord
    debug: Optional[bool] = None
    merge_window_ms: int = 5
    events: List[ProviderStreamEvent] = field(default_factory=list)
    start_ms: int = field(default_factory=now_ms)
    _next_seq: int = 0
    _closed: bool = False
    _debug_logger: Optional[Callable[[str], None]] = None

    def __post_init__(self) -> None:
        if self.debug is None:
            self.debug = is_stream_debug_enabled()
        self.record.action_times = [self.start_ms]
        self.record.reasoning_times = [self.start_ms]
        self.record.content_times = [self.start_ms]
        if self._debug_logger is None:
            self._debug_logger = lambda msg: print(f"[stream-debug] {msg}")
        if self.debug:
            self._log_debug(f"accumulator-start provider={self.record.provider} model={self.record.model}")

    def append_reasoning(self, text: str, ts_ms: Optional[int] = None, seq: Optional[int] = None) -> None:
        self._append_event("reasoning", text, ts_ms, seq)

    def append_content(self, text: str, ts_ms: Optional[int] = None, seq: Optional[int] = None) -> None:
        self._append_event("content", text, ts_ms, seq)

    def finalize(self, success: bool = True, final_ts: Optional[int] = None) -> None:
        if self._closed:
            return
        self._closed = True
        final_ts = max(final_ts or (self.events[-1].ts_ms if self.events else self.start_ms), self.start_ms)
        action_times = [self.start_ms]
        reasoning_times = [self.start_ms]
        content_times = [self.start_ms]
        reasoning_segments: List[str] = []
        content_segments: List[str] = []

        for event in self.events:
            action_times.append(event.ts_ms)
            if event.kind == "reasoning":
                reasoning_segments.append(event.text)
                reasoning_times.append(event.ts_ms)
            elif event.kind == "content":
                content_segments.append(event.text)
                content_times.append(event.ts_ms)

        if not success:
            reasoning_segments = []
            content_segments = []
            reasoning_times = [self.start_ms]
            content_times = [self.start_ms]
        if not action_times or action_times[-1] != final_ts:
            action_times.append(final_ts)

        self.record.reasoning = reasoning_segments
        self.record.content = content_segments
        self.record.reasoning_times = reasoning_times
        self.record.content_times = content_times
        self.record.action_times = action_times

        if self.debug:
            snapshot = self.snapshot()
            snapshot["status"] = "success" if success else "discarded"
            self._log_debug(json.dumps(snapshot, ensure_ascii=False))
        self.events.clear()

    def snapshot(self) -> dict:
        duration = (self.events[-1].ts_ms - self.start_ms) if self.events else 0
        return {
            "run_id": self.record.run_id,
            "executor": self.record.executor_id,
            "provider": self.record.provider,
            "model": self.record.model,
            "chunks": len(self.events),
            "reasoning_chunks": len([e for e in self.events if e.kind == "reasoning"]),
            "content_chunks": len([e for e in self.events if e.kind == "content"]),
            "duration_ms": max(duration, 0),
        }

    def _append_event(
        self,
        kind: StreamEventKind,
        text: str,
        ts_ms: Optional[int],
        seq: Optional[int],
    ) -> None:
        if not text:
            return
        if self._closed:
            return
        ts = ts_ms or now_ms()
        if self.events and ts <= self.events[-1].ts_ms:
            ts = self.events[-1].ts_ms + 1
        if self.events:
            last = self.events[-1]
            if last.kind == kind and ts - last.ts_ms <= self.merge_window_ms:
                last.text += text
                last.ts_ms = ts
                return
        event = ProviderStreamEvent(
            kind=kind,
            text=text,
            ts_ms=ts,
            seq=seq if seq is not None else self._next_seq,
        )
        self.events.append(event)
        self._next_seq = event.seq + 1
        if self.debug:
            self._log_debug(f"append kind={kind} seq={event.seq} ts={event.ts_ms} len={len(text)}")

    def _log_debug(self, message: str) -> None:
        if self._debug_logger:
            self._debug_logger(message)
