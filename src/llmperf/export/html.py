"""HTML report exporter for LLMPerf records."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import Exporter, ExportConfig, ExportResult
from .registry import register_exporter
from llmperf.records.model import RunRecord

logger = logging.getLogger(__name__)


class HTMLReportConfig(ExportConfig):
    """Configuration for HTML report export."""
    title: str = "LLMPerf Report"
    """Report title."""

    include_charts: bool = True
    """Whether to include chart placeholders (for JS rendering)."""

    include_details: bool = True
    """Whether to include detailed record table."""

    theme: str = "light"
    """Color theme: 'light' or 'dark'."""

    max_detail_rows: int = 1000
    """Maximum number of rows in detail table."""


# HTML template for report
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg-primary: {bg_primary};
            --bg-secondary: {bg_secondary};
            --text-primary: {text_primary};
            --text-secondary: {text_secondary};
            --border-color: {border_color};
            --accent-color: #0d6efd;
            --success-color: #198754;
            --warning-color: #fd7e14;
            --danger-color: #dc3545;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .header {{
            background: linear-gradient(135deg, var(--accent-color), #6610f2);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}

        .header h1 {{
            font-size: 2rem;
            margin-bottom: 10px;
        }}

        .header .meta {{
            opacity: 0.9;
            font-size: 0.9rem;
        }}

        .card {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}

        .card h2 {{
            font-size: 1.25rem;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}

        .stat-item {{
            text-align: center;
            padding: 15px;
            background-color: var(--bg-primary);
            border-radius: 8px;
        }}

        .stat-value {{
            font-size: 1.75rem;
            font-weight: bold;
            color: var(--accent-color);
        }}

        .stat-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 5px;
        }}

        .progress-bar {{
            background-color: var(--bg-primary);
            border-radius: 10px;
            overflow: hidden;
            height: 20px;
        }}

        .progress-bar .fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--success-color), var(--accent-color));
            transition: width 0.3s;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background-color: var(--bg-primary);
            font-weight: 600;
            position: sticky;
            top: 0;
        }}

        tr:hover {{
            background-color: var(--bg-primary);
        }}

        .status-success {{
            color: var(--success-color);
        }}

        .status-error {{
            color: var(--danger-color);
        }}

        .table-container {{
            max-height: 500px;
            overflow-y: auto;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}

        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}

        .chart-placeholder {{
            background-color: var(--bg-primary);
            border: 2px dashed var(--border-color);
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            color: var(--text-secondary);
        }}

        @media print {{
            body {{
                padding: 0;
            }}
            .card {{
                break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="meta">
                <div>Run ID: {run_id}</div>
                <div>Generated: {generated_at}</div>
            </div>
        </div>

        {summary_section}

        {executors_section}

        {details_section}

        <div class="footer">
            <p>Generated by LLMPerf</p>
        </div>
    </div>
</body>
</html>
'''


@register_exporter("html")
class HTMLReportExporter(Exporter):
    """HTML report exporter.

    Generates a self-contained HTML report with summary statistics
    and detailed records.

    Configuration:
        output_dir: Output directory (default: ".")
        title: Report title (default: "LLMPerf Report")
        include_charts: Include chart placeholders (default: True)
        include_details: Include detail table (default: True)
        theme: Color theme (default: "light")
        max_detail_rows: Max rows in detail table (default: 1000)
    """

    format_name = "html"
    file_extension = ".html"

    def __init__(self, config: ExportConfig):
        """Initialize HTML exporter.

        Args:
            config: Export configuration.
        """
        if not isinstance(config, HTMLReportConfig):
            config = HTMLReportConfig(**config.model_dump())
        super().__init__(config)

    @property
    def html_config(self) -> HTMLReportConfig:
        """Get HTML-specific configuration."""
        return self.config  # type: ignore

    def _get_theme_colors(self) -> Dict[str, str]:
        """Get theme color variables.

        Returns:
            Dictionary of CSS color variables.
        """
        if self.html_config.theme == "dark":
            return {
                "bg_primary": "#1a1a2e",
                "bg_secondary": "#16213e",
                "text_primary": "#eaeaea",
                "text_secondary": "#a0a0a0",
                "border_color": "#2a2a4a",
            }
        else:
            return {
                "bg_primary": "#f8f9fa",
                "bg_secondary": "#ffffff",
                "text_primary": "#212529",
                "text_secondary": "#6c757d",
                "border_color": "#dee2e6",
            }

    def _calculate_summary(self, records: List[RunRecord]) -> Dict[str, Any]:
        """Calculate summary statistics.

        Args:
            records: List of records.

        Returns:
            Summary statistics dictionary.
        """
        if not records:
            return {}

        total = len(records)
        successful = [r for r in records if r.status == 200]
        failed = [r for r in records if r.status != 200]

        total_cost = sum(r.total_cost for r in records)
        currency = records[0].currency if records else "CNY"

        first_resp_times = [r.first_resp_time for r in successful if r.first_resp_time > 0]
        char_per_sec = [r.char_per_second for r in successful if r.char_per_second > 0]
        token_throughput = [r.token_throughput for r in successful if r.token_throughput > 0]

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        return {
            "total_requests": total,
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / total * 100 if total > 0 else 0,
            "total_cost": total_cost,
            "currency": currency,
            "avg_first_resp_ms": avg(first_resp_times),
            "p95_first_resp_ms": self._percentile(first_resp_times, 0.95),
            "avg_char_per_sec": avg(char_per_sec),
            "avg_token_throughput": avg(token_throughput),
            "total_input_tokens": sum(r.qtokens for r in records),
            "total_output_tokens": sum(r.atokens for r in records),
            "total_cached_tokens": sum(r.ctokens for r in records),
        }

    def _percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile.

        Args:
            data: List of values.
            p: Percentile (0-1).

        Returns:
            Percentile value.
        """
        if not data:
            return 0
        sorted_data = sorted(data)
        k = int(len(sorted_data) * p)
        return sorted_data[min(k, len(sorted_data) - 1)]

    def _group_by_executor(self, records: List[RunRecord]) -> Dict[str, List[RunRecord]]:
        """Group records by executor ID.

        Args:
            records: List of records.

        Returns:
            Dictionary mapping executor ID to records.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for r in records:
            groups[r.executor_id].append(r)
        return dict(groups)

    def _render_summary_section(self, summary: Dict[str, Any]) -> str:
        """Render summary statistics section.

        Args:
            summary: Summary statistics.

        Returns:
            HTML string.
        """
        if not summary:
            return ""

        return f'''
        <div class="card">
            <h2>Summary</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{summary['total_requests']:,}</div>
                    <div class="stat-label">Total Requests</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{summary['success_rate']:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{summary['total_cost']:.4f}</div>
                    <div class="stat-label">Total Cost ({summary['currency']})</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{summary['avg_first_resp_ms']:.0f}ms</div>
                    <div class="stat-label">Avg First Response</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{summary['avg_char_per_sec']:.1f}</div>
                    <div class="stat-label">Avg Char/sec</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{summary['avg_token_throughput']:.1f}</div>
                    <div class="stat-label">Avg Token/sec</div>
                </div>
            </div>
        </div>
        '''

    def _render_executors_section(
        self,
        grouped_records: Dict[str, List[RunRecord]],
    ) -> str:
        """Render per-executor statistics section.

        Args:
            grouped_records: Records grouped by executor.

        Returns:
            HTML string.
        """
        if not grouped_records:
            return ""

        rows = []
        for executor_id, records in grouped_records.items():
            summary = self._calculate_summary(records)
            rows.append(f'''
            <tr>
                <td>{executor_id}</td>
                <td>{records[0].provider if records else '-'}</td>
                <td>{records[0].model if records else '-'}</td>
                <td>{summary.get('total_requests', 0):,}</td>
                <td>{summary.get('success_rate', 0):.1f}%</td>
                <td>{summary.get('avg_first_resp_ms', 0):.0f}ms</td>
                <td>{summary.get('total_cost', 0):.4f}</td>
            </tr>
            ''')

        return f'''
        <div class="card">
            <h2>Executors</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Executor</th>
                            <th>Provider</th>
                            <th>Model</th>
                            <th>Requests</th>
                            <th>Success Rate</th>
                            <th>Avg First Response</th>
                            <th>Cost</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(rows)}
                    </tbody>
                </table>
            </div>
        </div>
        '''

    def _render_details_section(
        self,
        records: List[RunRecord],
        max_rows: int,
    ) -> str:
        """Render detailed records section.

        Args:
            records: List of records.
            max_rows: Maximum rows to display.

        Returns:
            HTML string.
        """
        if not self.html_config.include_details or not records:
            return ""

        display_records = records[:max_rows]
        truncated = len(records) > max_rows

        rows = []
        for r in display_records:
            status_class = "status-success" if r.status == 200 else "status-error"
            rows.append(f'''
            <tr>
                <td>{r.executor_id}</td>
                <td>{r.model}</td>
                <td class="{status_class}">{r.status}</td>
                <td>{r.qtokens:,}</td>
                <td>{r.atokens:,}</td>
                <td>{r.first_resp_time:.0f}ms</td>
                <td>{r.char_per_second:.1f}</td>
                <td>{r.total_cost:.4f}</td>
            </tr>
            ''')

        truncation_notice = ""
        if truncated:
            truncation_notice = f'''
            <p style="color: var(--text-secondary); margin-top: 10px;">
                Showing {max_rows} of {len(records)} records.
            </p>
            '''

        return f'''
        <div class="card">
            <h2>Detailed Records</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Executor</th>
                            <th>Model</th>
                            <th>Status</th>
                            <th>Input Tokens</th>
                            <th>Output Tokens</th>
                            <th>First Response</th>
                            <th>Char/sec</th>
                            <th>Cost</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(rows)}
                    </tbody>
                </table>
            </div>
            {truncation_notice}
        </div>
        '''

    def export(
        self,
        records: List[RunRecord],
        run_id: str,
        task_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExportResult:
        """Export records to HTML report.

        Args:
            records: List of records to export.
            run_id: Run identifier.
            task_name: Task name for filename.
            metadata: Optional additional metadata.

        Returns:
            ExportResult indicating success or failure.
        """
        if not records:
            return ExportResult(
                success=False,
                format=self.format_name,
                message="No records to export",
            )

        output_path = self._build_output_path(task_name)
        theme_colors = self._get_theme_colors()

        try:
            # Calculate statistics
            summary = self._calculate_summary(records)
            grouped = self._group_by_executor(records)

            # Generate report title
            title = self.html_config.title
            if task_name:
                title = f"{title} - {task_name}"

            # Render sections
            summary_section = self._render_summary_section(summary)
            executors_section = self._render_executors_section(grouped)
            details_section = self._render_details_section(
                records,
                self.html_config.max_detail_rows,
            )

            # Fill template
            html_content = HTML_TEMPLATE.format(
                title=title,
                run_id=run_id,
                generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                summary_section=summary_section,
                executors_section=executors_section,
                details_section=details_section,
                **theme_colors,
            )

            # Write file
            with output_path.open("w", encoding="utf-8") as f:
                f.write(html_content)

            file_size = output_path.stat().st_size

            logger.info(
                "Exported HTML report to %s (%.2f KB)",
                output_path,
                file_size / 1024,
            )

            return ExportResult(
                success=True,
                output_path=str(output_path),
                format=self.format_name,
                records_exported=len(records),
                message="Successfully exported HTML report",
                metadata={
                    "theme": self.html_config.theme,
                    "file_size_bytes": file_size,
                    "truncated": len(records) > self.html_config.max_detail_rows,
                },
            )

        except Exception as e:
            logger.error("Failed to export HTML report: %s", e)
            return ExportResult(
                success=False,
                format=self.format_name,
                message=f"Export failed: {e}",
            )
