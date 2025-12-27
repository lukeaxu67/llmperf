from __future__ import annotations

import argparse
import logging

from llmperf.config.loader import load_config
from llmperf.pricing.loader import load_pricing_entries
from llmperf.runner import execute_from_yaml


def main():
    parser = argparse.ArgumentParser(description="Run benchmarking tasks using YAML configuration.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    parser.add_argument("--pricing-file", help="Path to price catalog YAML")
    parser.add_argument("--run-id", help="Optional run identifier")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        cfg = load_config(args.config)
        pricing_path = args.pricing_file
        if pricing_path:
            cfg.pricing = load_pricing_entries(pricing_path)
        run_id = execute_from_yaml(args.config, pricing_path=pricing_path, config=cfg, run_id=args.run_id)
    except KeyboardInterrupt:
        print("Run interrupted by user (Ctrl-C).")
        raise SystemExit(130)
    print(f"Run completed. run_id={run_id}")


if __name__ == "__main__":
    main()
