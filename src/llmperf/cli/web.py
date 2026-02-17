"""CLI entry point for running the web server."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def main():
    """Run the LLMPerf web server."""
    parser = argparse.ArgumentParser(
        description="Run the LLMPerf web server"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("LLMPERF_WEB_HOST", "0.0.0.0"),
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("LLMPERF_WEB_PORT", "8000")),
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level (default: info)",
    )
    parser.add_argument(
        "--static-dir",
        default=os.environ.get("STATIC_DIR"),
        help="Directory to serve static files from (optional)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    )

    # Import and run server
    from llmperf.web.main import run_server

    print(f"Starting LLMPerf web server on {args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/api/docs")
    print(f"OpenAPI spec: http://{args.host}:{args.port}/api/openapi.json")

    if args.static_dir:
        print(f"Static files: {args.static_dir}")

    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        static_dir=args.static_dir,
    )


if __name__ == "__main__":
    main()
