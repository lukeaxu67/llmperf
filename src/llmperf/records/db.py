from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sqlalchemy import Float, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class RunORM(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    task_type: Mapped[str] = mapped_column(String, nullable=False)
    info: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[int] = mapped_column(Integer, default=lambda: int(time.time()))
    config_path: Mapped[str] = mapped_column(Text, default="")
    config_content: Mapped[str] = mapped_column(Text, default="")
    pricing_path: Mapped[str] = mapped_column(Text, default="")
    pricing_content: Mapped[str] = mapped_column(Text, default="")


class ExecutionORM(Base):
    __tablename__ = "executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    executor_id: Mapped[str] = mapped_column(String)
    dataset_row_id: Mapped[str] = mapped_column(String)
    provider: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    status: Mapped[int] = mapped_column(Integer)
    info: Mapped[str] = mapped_column(Text)
    qtokens: Mapped[int] = mapped_column(Integer)
    atokens: Mapped[int] = mapped_column(Integer)
    ctokens: Mapped[int] = mapped_column(Integer)
    prompt_cost: Mapped[float] = mapped_column(Float)
    completion_cost: Mapped[float] = mapped_column(Float)
    cache_cost: Mapped[float] = mapped_column(Float, default=0.0)
    total_cost: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String)
    usage_json: Mapped[str] = mapped_column(Text)
    request_params_json: Mapped[str] = mapped_column(Text, default="{}")
    action_times: Mapped[str] = mapped_column(Text)
    reasoning_times: Mapped[str] = mapped_column(Text)
    content_times: Mapped[str] = mapped_column(Text)
    reasoning_json: Mapped[str] = mapped_column(Text)
    content_json: Mapped[str] = mapped_column(Text)
    extra_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[int] = mapped_column(Integer, default=lambda: int(time.time()))


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def loads(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


class Database:
    def __init__(self, db_path: str = "data.db"):
        self.db_path = Path(db_path)
        if self.db_path.parent and not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", future=True)
        self._legacy_responses_column = self._detect_legacy_responses_column()
        Base.metadata.create_all(self.engine)
        self._ensure_cache_cost_column()
        self._ensure_run_snapshot_columns()
        self._ensure_request_params_column()
        self._ensure_provider_column()
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def session(self) -> Session:
        return self.Session()

    def close(self) -> None:
        self.engine.dispose()

    def has_legacy_responses_column(self) -> bool:
        return self._legacy_responses_column

    def export_legacy_responses(self, export_path: str) -> int:
        """Dump legacy responses_json content to a JSONL file for manual reconciliation."""
        if not self.has_legacy_responses_column():
            return 0
        target = Path(export_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows_exported = 0
        with self.engine.connect() as conn, target.open("w", encoding="utf-8") as handle:
            result = conn.execute(
                text(
                    "SELECT id, run_id, responses_json FROM executions "
                    "WHERE responses_json IS NOT NULL AND responses_json <> ''"
                )
            )
            for row in result:
                handle.write(
                    json.dumps(
                        {
                            "execution_id": row.id,
                            "run_id": row.run_id,
                            "responses": loads(row.responses_json, []),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows_exported += 1
        return rows_exported

    def _detect_legacy_responses_column(self) -> bool:
        try:
            with self.engine.connect() as conn:
                pragma = conn.execute(text("PRAGMA table_info(executions);")).fetchall()
        except Exception:
            return False
        return any(col[1] == "responses_json" for col in pragma)

    def _ensure_cache_cost_column(self) -> None:
        try:
            with self.engine.connect() as conn:
                if not self._table_exists(conn, "executions"):
                    return
                pragma = conn.execute(text("PRAGMA table_info(executions);")).fetchall()
                if any(col[1] == "cache_cost" for col in pragma):
                    return
                conn.execute(text("ALTER TABLE executions ADD COLUMN cache_cost REAL DEFAULT 0"))
        except Exception:
            # Fallback silently; inserts will still work because SQLAlchemy reflects column definitions.
            pass

    def _ensure_request_params_column(self) -> None:
        try:
            with self.engine.connect() as conn:
                if not self._table_exists(conn, "executions"):
                    return
                pragma = conn.execute(text("PRAGMA table_info(executions);")).fetchall()
                if any(col[1] == "request_params_json" for col in pragma):
                    return
                conn.execute(
                    text(
                        "ALTER TABLE executions ADD COLUMN request_params_json TEXT DEFAULT '{}'"
                    )
                )
        except Exception:
            pass

    def _ensure_provider_column(self) -> None:
        try:
            with self.engine.connect() as conn:
                if not self._table_exists(conn, "executions"):
                    return
                pragma = conn.execute(text("PRAGMA table_info(executions);")).fetchall()
                existing = {col[1] for col in pragma}
                if "provider" in existing:
                    return
                if "vendor" in existing:
                    conn.execute(text("ALTER TABLE executions RENAME COLUMN vendor TO provider"))
                    return
                conn.execute(text("ALTER TABLE executions ADD COLUMN provider TEXT DEFAULT ''"))
        except Exception:
            pass

    def _ensure_run_snapshot_columns(self) -> None:
        try:
            with self.engine.connect() as conn:
                if not self._table_exists(conn, "runs"):
                    return
                pragma = conn.execute(text("PRAGMA table_info(runs);")).fetchall()
                existing = {col[1] for col in pragma}
                if "config_content" not in existing:
                    conn.execute(
                        text("ALTER TABLE runs ADD COLUMN config_content TEXT DEFAULT ''")
                    )
                if "pricing_path" not in existing:
                    conn.execute(
                        text("ALTER TABLE runs ADD COLUMN pricing_path TEXT DEFAULT ''")
                    )
                if "pricing_content" not in existing:
                    conn.execute(
                        text("ALTER TABLE runs ADD COLUMN pricing_content TEXT DEFAULT ''")
                    )
        except Exception:
            pass

    def _table_exists(self, conn, table: str) -> bool:
        row = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name = :table"),
            {"table": table},
        ).first()
        return row is not None
