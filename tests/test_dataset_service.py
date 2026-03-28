from __future__ import annotations

import json

from llmperf.web.services.dataset_service import DatasetService


def test_dataset_service_uses_meta_id_for_lookup_and_file_path_for_storage(tmp_path):
    data_path = tmp_path / "physical-file.jsonl"
    data_path.write_text(
        json.dumps(
            {"id": "row-1", "messages": [{"role": "user", "content": "hello"}]},
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    metadata_path = tmp_path / "stable-id.meta.json"
    metadata_path.write_text(
        json.dumps(
            {
                "name": "展示名称已修改",
                "description": "来自 meta 的描述",
                "file_path": data_path.name,
                "file_type": "jsonl",
                "row_count": 1,
                "columns": ["id", "messages"],
                "created_at": 1,
                "updated_at": 2,
                "file_size": data_path.stat().st_size,
                "encoding": "utf-8",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    service = DatasetService(tmp_path)
    datasets = service.scan()

    assert len(datasets) == 1
    dataset = service.get_dataset("stable-id")
    assert dataset is not None
    assert dataset.id == "stable-id"
    assert dataset.name == "展示名称已修改"
    assert dataset.file_path == "physical-file.jsonl"

    preview = service.preview_dataset("stable-id", limit=1)
    assert preview["id"] == "stable-id"
    assert preview["name"] == "展示名称已修改"
    assert preview["file_path"] == "physical-file.jsonl"
    assert preview["preview_rows"] == 1
