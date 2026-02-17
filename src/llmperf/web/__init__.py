"""Web API module for LLMPerf.

This module provides a FastAPI-based web interface for:
- Task management (create, list, cancel tasks)
- Real-time progress monitoring via WebSocket
- Data analysis and visualization
- Configuration management
"""

from .main import create_app, app

__all__ = ["create_app", "app"]
