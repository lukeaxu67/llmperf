from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "datasets"

DATASETS = [
    ("CC_testset-query.jsonl", "CC_testset_query"),
    ("Math_testset-query.jsonl", "Math_testset_query"),
    ("OC_testset-query.jsonl", "OC_testset_query"),
]


def convert_dataset(source_name: str, output_name: str) -> tuple[Path, Path, int]:
    source_path = DATASET_DIR / source_name
    output_path = DATASET_DIR / f"{output_name}.jsonl"
    meta_path = DATASET_DIR / f"{output_name}.meta.json"

    row_count = 0
    with source_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as output:
        for row_count, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            messages = json.loads(stripped)
            payload = {
                "id": str(row_count),
                "messages": messages,
                "metadata": {
                    "source_file": source_name,
                    "source_line": row_count,
                },
            }
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")

    file_size = output_path.stat().st_size
    now_ts = int(time.time())
    meta = {
        "name": output_name,
        "description": f"Converted from {source_name} to LLMPerf JSONL testcase format.",
        "file_path": output_path.name,
        "file_type": "jsonl",
        "row_count": row_count,
        "columns": ["id", "messages", "metadata"],
        "created_at": now_ts,
        "updated_at": now_ts,
        "file_size": file_size,
        "encoding": "utf-8",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path, meta_path, row_count


def main() -> None:
    for source_name, output_name in DATASETS:
        output_path, meta_path, row_count = convert_dataset(source_name, output_name)
        print(f"converted {source_name} -> {output_path.name} ({row_count} rows)")
        print(f"wrote metadata -> {meta_path.name}")


if __name__ == "__main__":
    main()
