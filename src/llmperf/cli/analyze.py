from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from ..analysis import create_analysis, load_analysis_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze historical benchmarking data.")
    parser.add_argument("--type", required=True, help="Analysis type (e.g. cost, summary)")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config")
    parser.add_argument("--output", help="Optional path to write JSON output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config_data = load_analysis_config(args.config)
    analysis = create_analysis(args.type, config_data)
    result: Dict[str, Any] = analysis.run()

    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(str(out_path))
        return
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

