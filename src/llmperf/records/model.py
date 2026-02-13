from __future__ import annotations

import json
import time
import statistics
from dataclasses import fields
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
from ..utils.counter import get_word_num


def aliased_property(alias: str):
    def _decorator(func):
        setattr(func, "__llmperf_alias__", alias)
        return property(func)

    return _decorator


@dataclass
class RunRecord:
    run_id: str = field(metadata={"alias": "任务ID"})
    executor_id: str = field(metadata={"alias": "执行器ID"})
    dataset_row_id: str = field(metadata={"alias": "样本ID"})
    provider: str = field(metadata={"alias": "提供方"})
    model: str = field(metadata={"alias": "模型"})
    status: int = field(default=0, metadata={"alias": "状态码"})
    info: str = field(default="{}", metadata={"alias": "信息"})
    qtokens: int = field(default=0, metadata={"alias": "输入tokens"})
    atokens: int = field(default=0, metadata={"alias": "输出tokens"})
    ctokens: int = field(default=0, metadata={"alias": "缓存tokens"})
    prompt_cost: float = field(default=0.0, metadata={"alias": "输入费用"})
    completion_cost: float = field(default=0.0, metadata={"alias": "输出费用"})
    cache_cost: float = field(default=0.0, metadata={"alias": "缓存费用"})
    total_cost: float = field(default=0.0, metadata={"alias": "总费用"})
    currency: str = field(default="CNY", metadata={"alias": "币种"})
    usage: Dict[str, Any] = field(default_factory=dict)
    request_params: Dict[str, Any] = field(default_factory=dict, metadata={"alias": "请求参数"})
    action_times: List[int] = field(default_factory=list)
    reasoning_times: List[int] = field(default_factory=list)
    content_times: List[int] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    content: List[str] = field(default_factory=list)
    output_items: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    created_at: int = field(default_factory=lambda: int(time.time()))

    @aliased_property("首响(ms)")
    def first_resp_time(self) -> int:
        if len(self.action_times) < 2:
            return 0
        return self.action_times[1] - self.action_times[0]

    @aliased_property("总耗时(ms)")
    def last_resp_time(self) -> int:
        if len(self.action_times) < 2:
            return 0
        return self.action_times[-1] - self.action_times[0]

    @aliased_property("字符数")
    def char_count(self) -> int:
        return get_word_num("".join(self.reasoning + self.content))

    @aliased_property("推理字符数")
    def reasoning_char_count(self) -> int:
        return get_word_num("".join(self.reasoning))

    @aliased_property("内容字符数")
    def content_char_count(self) -> int:
        return get_word_num("".join(self.content))

    @aliased_property("会话耗时(ms)")
    def session_time(self) -> int:
        if len(self.action_times) < 2:
            return 1
        t = self.action_times[-1] - self.action_times[1]
        return 1 if t == 0 else t

    @property
    def reasoning_session_time(self) -> int:
        if len(self.reasoning_times) < 2:
            return 1
        t = self.reasoning_times[-1] - self.reasoning_times[0]
        return 1 if t == 0 else t

    @property
    def content_session_time(self) -> int:
        if len(self.content_times) < 2:
            return 1
        t = self.content_times[-1] - self.content_times[0]
        return 1 if t == 0 else t

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def _usage_dict(self) -> Dict[str, Any]:
        usage = self.usage or {}
        if isinstance(usage, str):
            try:
                usage = json.loads(usage)
            except Exception:
                usage = {}
        return usage if isinstance(usage, dict) else {}

    @aliased_property("字符速度(char/s)")
    def char_per_second(self) -> float:
        return self.char_count * 1e3 / self.session_time

    @aliased_property("字符速度(含首响)(char/s)")
    def char_per_second_with_calltime(self) -> float:
        if len(self.action_times) < 2:
            return 0.0
        return self.char_count * 1e3 / (self.action_times[-1] - self.action_times[0])

    def _split_completion_tokens(self) -> tuple[int, int]:
        usage = self._usage_dict()
        details = (
            usage.get("completion_tokens_details")
            or usage.get("output_tokens_details")
            or usage.get("completion", {})
        )
        if isinstance(details, dict):
            r = details.get("reasoning_tokens") or details.get("thinking_tokens")
            c = details.get("content_tokens") or details.get("output_tokens")
            if isinstance(r, int) and isinstance(c, int):
                return max(r, 0), max(c, 0)
        total_chars = self.reasoning_char_count + self.content_char_count
        if total_chars <= 0 or self.atokens <= 0:
            return 0, 0
        r_tokens = int(self.atokens * (self.reasoning_char_count / total_chars))
        c_tokens = max(self.atokens - r_tokens, 0)
        return r_tokens, c_tokens

    def _cached_prompt_tokens(self, usage: Dict[str, Any]) -> int:
        candidates = [
            usage.get("prompt_cache_hit_tokens"),
            usage.get("cache_creation_input_tokens"),
            usage.get("cached_tokens"),
        ]
        prompt_details = usage.get("prompt_tokens_details") or usage.get("prompt_tokens_detail")
        if isinstance(prompt_details, dict):
            candidates.append(prompt_details.get("cached_tokens"))
        for value in candidates:
            if isinstance(value, int) and value > 0:
                return value
        return 0

    def _prompt_token_total(self, usage: Dict[str, Any]) -> int:
        for key in ("prompt_tokens", "input_tokens", "total_prompt_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return 0

    @aliased_property("缓存命中")
    def cache_hit(self) -> bool:
        usage = self._usage_dict()
        cached = max(self.ctokens, self._cached_prompt_tokens(usage))
        return cached > 0

    @aliased_property("缓存比例")
    def cache_ratio(self) -> float:
        usage = self._usage_dict()
        cached_tokens = max(self.ctokens, self._cached_prompt_tokens(usage))
        total_prompt_tokens = max(self.qtokens, self._prompt_token_total(usage))
        if total_prompt_tokens <= 0:
            return 0.0
        cached_tokens = min(cached_tokens, total_prompt_tokens)
        return cached_tokens / total_prompt_tokens

    @property
    def reasoning_char_per_second(self) -> float:
        return self.reasoning_char_count * 1e3 / self.reasoning_session_time

    @property
    def content_char_per_second(self) -> float:
        return self.content_char_count * 1e3 / self.content_session_time

    @aliased_property("token速度(tok/s)")
    def token_per_second(self) -> float:
        return self.atokens * 1e3 / self.session_time if self.session_time else 0

    @property
    def reasoning_token_per_second(self) -> float:
        r_tokens, _ = self._split_completion_tokens()
        return r_tokens * 1e3 / self.reasoning_session_time

    @property
    def content_token_per_second(self) -> float:
        _, c_tokens = self._split_completion_tokens()
        return c_tokens * 1e3 / self.content_session_time

    @property
    def token_per_second_with_calltime(self) -> float:
        if len(self.action_times) < 2:
            return 0
        return self.atokens * 1e3 / (self.action_times[-1] - self.action_times[0])

    @aliased_property("吞吐(tok/s)")
    def token_throughput(self) -> float:
        return (self.atokens + self.qtokens) * 1e3 / self.session_time

    @property
    def token_throughput_with_calltime(self) -> float:
        if len(self.action_times) < 2:
            return 0
        return (
            (self.atokens + self.qtokens)
            * 1e3
            / (self.action_times[-1] - self.action_times[0])
        )

    @property
    def average_interval(self) -> float:
        if len(self.action_times) < 2:
            return 0.0
        diffs = [
            (self.action_times[i] - self.action_times[i - 1])
            for i in range(1, len(self.action_times))
        ]
        return float(statistics.mean(diffs)) if diffs else 0.0

    @property
    def variance_interval(self) -> float:
        if len(self.action_times) <= 3:
            return 0.0
        diffs = [
            (self.action_times[i] - self.action_times[i - 1])
            for i in range(2, len(self.action_times))
        ]
        if len(diffs) <= 1:
            return 0.0
        return float(statistics.pvariance(diffs))

    @property
    def resp_times(self) -> int:
        if len(self.action_times) < 2:
            return 0
        return len(self.action_times) - 1

    def is_valid(self) -> bool:
        return len(self.action_times) > 2 and self.atokens > 0


def now_ms() -> int:
    return int(time.time() * 1e3)


def field_aliases(model_cls=RunRecord) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for f in fields(model_cls):
        alias = f.metadata.get("alias")
        if isinstance(alias, str) and alias:
            mapping[f.name] = alias
    for name in dir(model_cls):
        attr = getattr(model_cls, name, None)
        if isinstance(attr, property) and attr.fget is not None:
            alias = getattr(attr.fget, "__llmperf_alias__", None)
            if isinstance(alias, str) and alias:
                mapping[name] = alias
    return mapping
