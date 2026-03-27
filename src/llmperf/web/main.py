"""Main FastAPI application for LLMPerf Web UI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .routers import analysis, tasks, websocket, config, datasets, pricing, mock_openai
from .services.analysis_service import AnalysisService
from .services.task_service import TaskService

logger = logging.getLogger(__name__)

# Global service instances
_task_service: Optional[TaskService] = None
_analysis_service: Optional[AnalysisService] = None


def get_task_service() -> TaskService:
    """Get or create task service instance."""
    global _task_service
    if _task_service is None:
        _task_service = TaskService()
    return _task_service


def get_analysis_service() -> AnalysisService:
    """Get or create analysis service instance."""
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AnalysisService()
    return _analysis_service


def get_dataset_service():
    """Get or create dataset service instance."""
    from .services.dataset_service import get_dataset_service as _get
    return _get()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting LLMPerf Web API")

    # Initialize services
    get_task_service()

    # Initialize dataset service and scan datasets
    try:
        dataset_service = get_dataset_service()
        scanned = dataset_service.scan()
        logger.info(f"Dataset service initialized with {len(scanned)} datasets")
    except Exception as e:
        logger.warning("Failed to initialize dataset service: %s", e)

    # Import notification channels to register them
    try:
        from ..notifications import channels  # noqa: F401
    except ImportError:
        pass

    # Import exporters to register them
    try:
        from ..export import CSVExporter, JSONLExporter, HTMLReportExporter  # noqa: F401
    except ImportError:
        pass

    # Import providers to register them (especially mock for testing)
    try:
        from ..providers import mock  # noqa: F401
    except ImportError:
        pass

    yield

    # Shutdown
    logger.info("Shutting down LLMPerf Web API")


def create_app(
    title: str = "LLMPerf API",
    version: str = "0.1.0",
    cors_origins: Optional[list] = None,
    static_dir: Optional[str] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        title: API title.
        version: API version.
        cors_origins: List of allowed CORS origins.
        static_dir: Directory to serve static files from.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title=title,
        version=version,
        description="API for LLMPerf - Unified benchmarking toolkit for LLM providers",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # Configure CORS
    if cors_origins is None:
        cors_origins = ["*"]  # Allow all origins for development

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(
        tasks.router,
        prefix="/api/tasks",
        tags=["tasks"],
    )
    app.include_router(
        analysis.router,
        prefix="/api/analysis",
        tags=["analysis"],
    )
    app.include_router(
        websocket.router,
        prefix="/ws",
        tags=["websocket"],
    )
    app.include_router(
        config.router,
        prefix="/api/config",
        tags=["config"],
    )
    app.include_router(
        datasets.router,
        prefix="/api/datasets",
        tags=["datasets"],
    )
    app.include_router(
        pricing.router,
        prefix="/api/pricing",
        tags=["pricing"],
    )
    app.include_router(mock_openai.router)

    # Health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": version}

    # API info endpoint
    @app.get("/api", tags=["root"])
    async def api_info():
        """API info endpoint."""
        return {
            "name": "LLMPerf API",
            "version": version,
            "docs": "/api/docs",
        }

    # Serve static files for frontend
    static_path = None
    if static_dir:
        static_path = Path(static_dir)
    else:
        # Check common locations
        possible_paths = [
            Path(__file__).parent.parent.parent.parent.parent / "frontend" / "dist",
            Path("/app/static"),
            Path(__file__).parent / "static",
        ]
        for p in possible_paths:
            if p.exists() and (p / "index.html").exists():
                static_path = p
                break

    if static_path and static_path.exists():
        # Mount assets directory
        assets_path = static_path / "assets"
        if assets_path.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

        # Serve index.html for all non-API routes (SPA support)
        @app.get("/{path:path}", include_in_schema=False)
        async def serve_spa(request: Request, path: str):
            """Serve the SPA for all non-API routes."""
            # Skip API and WebSocket routes
            if path.startswith("api/") or path.startswith("ws/") or path == "health":
                return JSONResponse({"detail": "Not Found"}, status_code=404)

            # Try to serve static file
            file_path = static_path / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)

            # Fall back to index.html for SPA routing
            index_path = static_path / "index.html"
            if index_path.exists():
                return FileResponse(index_path)

            return JSONResponse({"detail": "Not Found"}, status_code=404)

        logger.info(f"Serving frontend from: {static_path}")
    else:
        # No static files, just serve API
        @app.get("/", tags=["root"])
        async def root():
            """Root endpoint."""
            return {
                "name": "LLMPerf API",
                "version": version,
                "docs": "/api/docs",
                "message": "Frontend not built. Run 'npm run build' in frontend directory.",
            }

    return app


# Default app instance
app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info",
    static_dir: Optional[str] = None,
):
    """Run the API server.

    Args:
        host: Server host.
        port: Server port.
        reload: Enable auto-reload for development.
        log_level: Log level.
        static_dir: Directory for static frontend files.
    """
    # Create app with static dir
    global app
    app = create_app(static_dir=static_dir)

    uvicorn.run(
        "llmperf.web.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    run_server()
