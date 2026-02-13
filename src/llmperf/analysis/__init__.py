from .analysis_registry import create_analysis, load_analysis_config

# Import all analysis implementations to register them
from .impl import summary_export as _excel  # noqa: F401
from .impl import cross as _summary  # noqa: F401
from .impl import history as _history  # noqa: F401
from .impl import export as _export  # noqa: F401
from .impl import stability as _stability  # noqa: F401
from .impl import health as _health  # noqa: F401
from .impl import resource as _resource  # noqa: F401
from .impl import ratelimit as _ratelimit  # noqa: F401
from .impl import monitoring_report as _monitoring_report  # noqa: F401

__all__ = ["create_analysis", "load_analysis_config"]
