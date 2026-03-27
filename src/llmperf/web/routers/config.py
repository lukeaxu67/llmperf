"""Configuration management API router."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import yaml
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Default template directory
TEMPLATE_DIR = Path(__file__).parent.parent.parent.parent.parent / "template"


class ConfigTemplate(BaseModel):
    """A configuration template."""
    name: str
    path: str
    description: str = ""


class ConfigValidationResult(BaseModel):
    """Result of config validation."""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class RuntimeConfigResponse(BaseModel):
    """Response for runtime configuration."""
    db_path: str
    log_level: str
    log_dir: str


@router.get(
    "/templates",
    response_model=List[ConfigTemplate],
    summary="List config templates",
    description="Get a list of available configuration templates.",
)
async def list_templates():
    """List available configuration templates."""
    templates = []

    if TEMPLATE_DIR.exists():
        for path in TEMPLATE_DIR.glob("*.yaml"):
            # Skip test templates
            if path.name.startswith("test_"):
                continue

            content = path.read_text(encoding="utf-8")
            description = ""

            # Extract info from YAML
            try:
                data = yaml.safe_load(content)
                if isinstance(data, dict):
                    description = data.get("info", "")
            except Exception:
                pass

            templates.append(ConfigTemplate(
                name=path.stem,
                path=str(path.relative_to(TEMPLATE_DIR.parent.parent)),
                description=description,
            ))

    return templates


@router.get(
    "/templates/{name}",
    summary="Get config template",
    description="Get the content of a specific configuration template.",
)
async def get_template(name: str):
    """Get a configuration template by name."""
    # Security: prevent path traversal
    safe_name = name.replace("..", "").replace("/", "").replace("\\", "")
    template_path = TEMPLATE_DIR / f"{safe_name}.yaml"

    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")

    content = template_path.read_text(encoding="utf-8")

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=500, detail=f"Invalid YAML: {e}")

    return {
        "name": name,
        "path": str(template_path),
        "content": content,
        "parsed": data,
    }


@router.post(
    "/validate",
    response_model=ConfigValidationResult,
    summary="Validate configuration",
    description="Validate a YAML configuration.",
)
async def validate_config(
    request: Request,
):
    """Validate a YAML configuration."""
    raw_body = await request.body()
    config_content = raw_body.decode("utf-8") if raw_body else ""

    errors = []
    warnings = []

    # Parse YAML
    try:
        data = yaml.safe_load(config_content)
    except yaml.YAMLError as e:
        return ConfigValidationResult(
            valid=False,
            errors=[f"YAML parsing error: {e}"],
        )

    if not isinstance(data, dict):
        return ConfigValidationResult(
            valid=False,
            errors=["Configuration must be a YAML dictionary"],
        )

    # Validate required fields
    if "dataset" not in data:
        errors.append("Missing required field: dataset")

    if "executors" not in data:
        errors.append("Missing required field: executors")
    elif not isinstance(data["executors"], list):
        errors.append("executors must be a list")
    elif len(data["executors"]) == 0:
        errors.append("executors list cannot be empty")

    # Validate dataset
    if "dataset" in data:
        dataset = data["dataset"]
        if not isinstance(dataset, dict):
            errors.append("dataset must be a dictionary")
        elif "source" not in dataset:
            errors.append("dataset.source is required")
        elif not isinstance(dataset["source"], dict):
            errors.append("dataset.source must be a dictionary")
        elif "type" not in dataset["source"]:
            errors.append("dataset.source.type is required")

    # Validate executors
    if isinstance(data.get("executors"), list):
        for i, executor in enumerate(data["executors"]):
            if not isinstance(executor, dict):
                errors.append(f"executors[{i}] must be a dictionary")
                continue

            if "id" not in executor:
                errors.append(f"executors[{i}].id is required")
            if "type" not in executor:
                errors.append(f"executors[{i}].type is required")

            # Check for common issues
            if executor.get("concurrency", 1) > 100:
                warnings.append(
                    f"executors[{i}].concurrency is very high ({executor['concurrency']}), "
                    "this may cause rate limiting issues"
                )

            # Validate rate configuration
            rate = executor.get("rate", {})
            if isinstance(rate, dict):
                if "qps" in rate and "interval_seconds" in rate:
                    errors.append(
                        f"executors[{i}].rate: specify only one of qps or interval_seconds"
                    )

    # Check for circular dependencies in executors
    if isinstance(data.get("executors"), list):
        executor_ids = {e.get("id") for e in data["executors"] if isinstance(e, dict)}
        for executor in data["executors"]:
            if isinstance(executor, dict):
                after = executor.get("after", [])
                if isinstance(after, list):
                    for dep_id in after:
                        if dep_id not in executor_ids:
                            errors.append(
                                f"executors[{executor.get('id', '?')}].after references "
                                f"unknown executor '{dep_id}'"
                            )

    return ConfigValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


@router.post(
    "/upload",
    summary="Upload configuration",
    description="Upload a YAML configuration file.",
)
async def upload_config(file: UploadFile = File(...)):
    """Upload a configuration file."""
    if not file.filename or not file.filename.endswith((".yaml", ".yml")):
        raise HTTPException(
            status_code=400,
            detail="File must be a YAML file (.yaml or .yml)",
        )

    content = await file.read()
    try:
        text = content.decode("utf-8")
        data = yaml.safe_load(text)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    return {
        "filename": file.filename,
        "size": len(content),
        "parsed": data,
    }


@router.get(
    "/runtime",
    response_model=RuntimeConfigResponse,
    summary="Get runtime configuration",
    description="Get current runtime configuration settings.",
)
async def get_runtime_config():
    """Get runtime configuration."""
    from llmperf.config.runtime import load_runtime_config

    config = load_runtime_config()

    return RuntimeConfigResponse(
        db_path=str(config.db_path),
        log_level=config.log_level,
        log_dir=str(config.log_dir),
    )


@router.get(
    "/pricing",
    summary="Get pricing catalog",
    description="Get current pricing information.",
)
async def get_pricing():
    """Get pricing catalog."""
    pricing_dir = Path(__file__).parent.parent.parent.parent.parent / "pricings"

    pricing_files = []
    if pricing_dir.exists():
        for path in pricing_dir.glob("*.yaml"):
            content = path.read_text(encoding="utf-8")
            try:
                data = yaml.safe_load(content)
                pricing_files.append({
                    "name": path.stem,
                    "path": str(path),
                    "entries": data.get("pricing", []),
                })
            except Exception as e:
                logger.warning("Failed to load pricing file %s: %s", path, e)

    return {
        "pricing_files": pricing_files,
    }
