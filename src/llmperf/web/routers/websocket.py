"""WebSocket router for real-time updates."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        # Map of run_id -> set of websocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}
        # Map of websocket -> run_id
        self._run_mapping: Dict[WebSocket, str] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, run_id: str):
        """Accept a new WebSocket connection for a specific run.

        Args:
            websocket: The WebSocket connection.
            run_id: The run ID to subscribe to.
        """
        await websocket.accept()

        async with self._lock:
            if run_id not in self._connections:
                self._connections[run_id] = set()
            self._connections[run_id].add(websocket)
            self._run_mapping[websocket] = run_id

        logger.debug("WebSocket connected for run %s", run_id)

    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection.

        Args:
            websocket: The WebSocket connection to disconnect.
        """
        async with self._lock:
            run_id = self._run_mapping.pop(websocket, None)
            if run_id and run_id in self._connections:
                self._connections[run_id].discard(websocket)
                if not self._connections[run_id]:
                    del self._connections[run_id]

        logger.debug("WebSocket disconnected from run %s", run_id)

    async def broadcast_to_run(self, run_id: str, message: dict):
        """Broadcast a message to all connections for a run.

        Args:
            run_id: The run ID to broadcast to.
            message: The message to broadcast.
        """
        async with self._lock:
            connections = self._connections.get(run_id, set()).copy()

        if not connections:
            return

        message_str = json.dumps(message, ensure_ascii=False, default=str)

        # Send to all connections
        disconnected = []
        for connection in connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.warning("Failed to send WebSocket message: %s", e)
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            await self.disconnect(conn)

    async def broadcast_progress(
        self,
        run_id: str,
        progress_percent: float,
        completed: int,
        total: int,
        elapsed_seconds: float,
        eta_seconds: Optional[float] = None,
        success_count: int = 0,
        error_count: int = 0,
        current_cost: float = 0.0,
        currency: str = "CNY",
        status: str = "running",
        current_rate: float = 0.0,
        concurrency: int = 1,
        paused_at: Optional[datetime] = None,
        executors: Optional[list[dict]] = None,
        topology: Optional[dict] = None,
    ):
        """Broadcast progress update for a run.

        Args:
            run_id: The run ID.
            progress_percent: Progress percentage (0-100).
            completed: Number of completed requests.
            total: Total number of requests.
            elapsed_seconds: Elapsed time in seconds.
            eta_seconds: Estimated time remaining.
            success_count: Number of successful requests.
            error_count: Number of failed requests.
            current_cost: Current total cost.
            currency: Currency code.
            status: Current task status.
            current_rate: Current request rate (requests/second).
            concurrency: Current concurrency level.
            paused_at: When the task was paused.
        """
        message = {
            "type": "progress",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "progress_percent": progress_percent,
                "completed": completed,
                "total": total,
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": eta_seconds,
                "success_count": success_count,
                "error_count": error_count,
                "current_cost": current_cost,
                "currency": currency,
                "status": status,
                "current_rate": current_rate,
                "concurrency": concurrency,
                "paused_at": paused_at.isoformat() if paused_at else None,
                "executors": executors or [],
                "topology": topology or {"nodes": [], "edges": [], "layers": []},
            },
        }
        await self.broadcast_to_run(run_id, message)

    async def broadcast_event(
        self,
        run_id: str,
        event_type: str,
        data: dict,
    ):
        """Broadcast a custom event for a run.

        Args:
            run_id: The run ID.
            event_type: Type of event (e.g., 'error', 'alert', 'completed').
            data: Event-specific data.
        """
        message = {
            "type": event_type,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        await self.broadcast_to_run(run_id, message)


# Global connection manager
manager = ConnectionManager()


def get_manager() -> ConnectionManager:
    """Get the global connection manager."""
    return manager


@router.websocket("/tasks/{run_id}/progress")
async def websocket_progress(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for real-time progress updates.

    Connect to this endpoint to receive progress updates for a specific run.

    Message format:
    {
        "type": "progress" | "error" | "alert" | "completed",
        "run_id": "xxx",
        "timestamp": "2024-01-01T00:00:00",
        "data": { ... }
    }
    """
    await manager.connect(websocket, run_id)

    try:
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "message": f"Connected to run {run_id}",
        }))

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for any message from client (ping/pong)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )

                # Handle ping
                if data == "ping":
                    await websocket.send_text("pong")

            except asyncio.TimeoutError:
                # Send keepalive
                try:
                    await websocket.send_text(json.dumps({
                        "type": "keepalive",
                        "timestamp": datetime.now().isoformat(),
                    }))
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.debug("WebSocket disconnected for run %s", run_id)
    except Exception as e:
        logger.error("WebSocket error for run %s: %s", run_id, e)
    finally:
        await manager.disconnect(websocket)


@router.websocket("/tasks/{run_id}/logs")
async def websocket_logs(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for real-time log streaming.

    Connect to this endpoint to receive log messages for a specific run.
    """
    await websocket.accept()

    try:
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "message": f"Connected to logs for run {run_id}",
        }))

        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({
                    "type": "keepalive",
                    "timestamp": datetime.now().isoformat(),
                }))
            except Exception:
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket logs error for run %s: %s", run_id, e)
    finally:
        logger.debug("WebSocket logs disconnected for run %s", run_id)
