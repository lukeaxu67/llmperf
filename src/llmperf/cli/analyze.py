from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

from llmperf.analysis import create_analysis, load_analysis_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze historical benchmarking data.")
    parser.add_argument("--type", required=True, help="Analysis type (e.g. cost, summary)")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config_data = load_analysis_config(args.config)
    analysis = create_analysis(args.type, config_data)
    result: Dict[str, Any] = analysis.run()

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
