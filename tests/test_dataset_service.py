from __future__ import annotations

import json

from fastapi.testclient import TestClient

from llmperf.web.main import create_app
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
                "name": "Display Name Updated",
                "description": "From meta description",
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
    assert dataset.name == "Display Name Updated"
    assert dataset.file_path == "physical-file.jsonl"

    preview = service.preview_dataset("stable-id", limit=1)
    assert preview["id"] == "stable-id"
    assert preview["name"] == "Display Name Updated"
    assert preview["file_path"] == "physical-file.jsonl"
    assert preview["preview_rows"] == 1


def test_dataset_service_supports_message_array_jsonl_lines(tmp_path):
    runtime_dir = tmp_path / "runtime"
    service = DatasetService(runtime_dir)

    metadata = service.upload_dataset(
        filename="list.jsonl",
        content=(
            json.dumps([{"role": "user", "content": "first"}], ensure_ascii=False) + "\n"
            + json.dumps([{"role": "system", "content": "s"}, {"role": "user", "content": "second"}], ensure_ascii=False) + "\n"
        ).encode("utf-8"),
        name="List Format",
        description="message array lines",
    )

    assert metadata.row_count == 2
    preview = service.preview_dataset(metadata.id, limit=2)
    assert preview["records"][0]["id"] == "row-1"
    assert preview["records"][0]["messages"][0]["content"] == "first"
    assert preview["records"][1]["id"] == "row-2"
    assert preview["records"][1]["messages"][1]["content"] == "second"


def test_dataset_service_separates_runtime_and_builtin_datasets(tmp_path):
    runtime_dir = tmp_path / "runtime"
    preset_dir = tmp_path / "preset"
    preset_dir.mkdir()

    builtin_data = preset_dir / "builtin.jsonl"
    builtin_data.write_text(
        json.dumps({"id": "row-1", "messages": [{"role": "user", "content": "builtin"}]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (preset_dir / "builtin.meta.json").write_text(
        json.dumps(
            {
                "name": "Builtin Dataset",
                "description": "preset",
                "file_path": builtin_data.name,
                "file_type": "jsonl",
                "row_count": 1,
                "columns": ["id", "messages"],
                "created_at": 1,
                "updated_at": 1,
                "file_size": builtin_data.stat().st_size,
                "encoding": "utf-8",
            }
        ),
        encoding="utf-8",
    )

    service = DatasetService(runtime_dir, preset_datasets_dir=preset_dir)
    uploaded = service.upload_dataset(
        filename="runtime.jsonl",
        content=(
            json.dumps({"id": "row-1", "messages": [{"role": "user", "content": "runtime"}]}, ensure_ascii=False) + "\n"
        ).encode("utf-8"),
        name="Runtime Dataset",
    )

    datasets = {item.id: item for item in service.scan()}
    assert datasets["builtin"].source == "builtin"
    assert datasets["builtin"].read_only is True
    assert datasets[uploaded.id].source == "runtime"
    assert datasets[uploaded.id].read_only is False


def test_dataset_upload_validate_endpoint_accepts_message_array_jsonl():
    content = (
        json.dumps([{"role": "user", "content": "hello"}], ensure_ascii=False) + "\n"
        + json.dumps([{"role": "user", "content": "world"}], ensure_ascii=False) + "\n"
    ).encode("utf-8")

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/datasets/validate",
            files={"file": ("list.jsonl", content, "application/x-ndjson")},
            data={"encoding": "utf-8"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["row_count"] == 2
    assert body["preview_records"][0]["id"] == "row-1"


def test_dataset_upload_endpoint_persists_runtime_metadata():
    content = (
        json.dumps([{"role": "user", "content": "hello"}], ensure_ascii=False) + "\n"
    ).encode("utf-8")

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/datasets/upload",
            files={"file": ("list.jsonl", content, "application/x-ndjson")},
            data={
                "name": "Uploaded Runtime Dataset",
                "description": "from upload api",
                "encoding": "utf-8",
            },
        )
        assert response.status_code == 200
        dataset_id = response.json()["id"]

        detail_response = client.get(f"/api/datasets/{dataset_id}")

    assert detail_response.status_code == 200
    body = detail_response.json()
    assert body["name"] == "Uploaded Runtime Dataset"
    assert body["description"] == "from upload api"
    assert body["source"] == "runtime"
    assert body["read_only"] is False
