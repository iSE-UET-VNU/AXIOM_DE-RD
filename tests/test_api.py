"""Tests for the REST API request contracts and artifact-backed responses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.api.output import build_dataeng_output
from src.api.schemas import DataEngRequest, PresignedFileRequest
from src.models import DataObject, PipelineState


# ── request contracts ─────────────────────────────────────────────────────────

def test_presigned_file_accepts_http_and_https_urls() -> None:
    for url in ("https://bucket.s3.amazonaws.com/k?sig=abc", "http://localhost:9000/k"):
        assert PresignedFileRequest(key="k", presigned_url=url).presigned_url == url


def test_presigned_url_is_preserved_byte_for_byte() -> None:
    """Normalizing a signed URL invalidates the signature, so it must pass through."""
    url = "https://bucket.s3.amazonaws.com/a%2Fb?X-Amz-Signature=deadbeef&x=1+2"
    assert PresignedFileRequest(key="k", presigned_url=url).presigned_url == url


@pytest.mark.parametrize(
    "url",
    ["ftp://host/key", "s3://bucket/key", "not-a-url", "https://", "//host/key", ""],
)
def test_presigned_url_rejects_non_http_schemes_and_missing_host(url: str) -> None:
    with pytest.raises(ValidationError):
        PresignedFileRequest(key="k", presigned_url=url)


def test_blank_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PresignedFileRequest(key="   ", presigned_url="https://h/k")


def test_request_requires_bucket_and_at_least_one_file() -> None:
    with pytest.raises(ValidationError):
        DataEngRequest(bucket="b", files=[])
    with pytest.raises(ValidationError):
        DataEngRequest(bucket="  ", files=[{"key": "k", "presigned_url": "https://h/k"}])


def test_bucket_is_trimmed() -> None:
    request = DataEngRequest(
        bucket="  my-bucket  ",
        files=[{"key": "k", "presigned_url": "https://h/k"}],
    )
    assert request.bucket == "my-bucket"


def test_to_inventory_drops_unset_optionals_and_keeps_supplied_ones() -> None:
    request = DataEngRequest(
        bucket="b",
        files=[
            {"key": "a", "presigned_url": "https://h/a"},
            {"key": "b", "presigned_url": "https://h/b", "size": 10, "etag": "e"},
        ],
    )
    inventory = request.to_inventory()
    assert inventory["bucket"] == "b"
    assert inventory["files"][0] == {"key": "a", "presigned_url": "https://h/a"}
    assert inventory["files"][1]["size"] == 10
    assert inventory["files"][1]["etag"] == "e"


def test_unknown_request_fields_are_ignored_rather_than_rejected() -> None:
    request = DataEngRequest(
        bucket="b",
        files=[{"key": "k", "presigned_url": "https://h/k", "future_field": 1}],
        another_future_field="x",
    )
    assert "future_field" not in request.to_inventory()["files"][0]


# ── artifact-backed response ──────────────────────────────────────────────────

def _state(tmp_path: Path, artifact_paths: dict[str, str], object_ids: list[str]) -> PipelineState:
    return PipelineState(
        run_id="run-1",
        input_source="test",
        embedded_dir=str(tmp_path / "embedded"),
        output_dir=str(tmp_path / "output"),
        data_objects=[DataObject(object_id=oid, uri=f"s3://b/{oid}") for oid in object_ids],
        artifact_paths=artifact_paths,
    )


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_output_bundles_metadata_and_one_document_per_data_object(tmp_path: Path) -> None:
    _write(tmp_path / "meta.json", {"run_id": "run-1"})
    _write(tmp_path / "docs" / "a.json", {"document": {"id": "a"}})
    _write(tmp_path / "docs" / "b.json", {"document": {"id": "b"}})

    result = build_dataeng_output(
        _state(
            tmp_path,
            {
                "output_metadata": str(tmp_path / "meta.json"),
                "output_document:a": str(tmp_path / "docs" / "a.json"),
                "output_document:b": str(tmp_path / "docs" / "b.json"),
            },
            ["a", "b"],
        ),
        project_root=tmp_path,
    )
    assert result["metadata"] == {"run_id": "run-1"}
    assert [d["document"]["id"] for d in result["documents"]] == ["a", "b"]


def test_relative_artifact_paths_resolve_against_the_project_root(tmp_path: Path) -> None:
    _write(tmp_path / "meta.json", {"ok": True})
    result = build_dataeng_output(
        _state(tmp_path, {"output_metadata": "meta.json"}, []),
        project_root=tmp_path,
    )
    assert result["metadata"] == {"ok": True}


def test_missing_artifact_registration_names_the_artifacts_module(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="artifacts module enabled"):
        build_dataeng_output(_state(tmp_path, {}, []), project_root=tmp_path)


def test_registered_but_absent_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        build_dataeng_output(
            _state(tmp_path, {"output_metadata": str(tmp_path / "gone.json")}, []),
            project_root=tmp_path,
        )


def test_artifact_path_outside_the_project_root_is_refused(tmp_path: Path) -> None:
    """Guards against a poisoned state leaking arbitrary files over HTTP."""
    outside = _write(tmp_path / "outside" / "secret.json", {"secret": True})
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    with pytest.raises(RuntimeError, match="outside the project root"):
        build_dataeng_output(
            _state(tmp_path, {"output_metadata": str(outside)}, []),
            project_root=root,
        )


def test_traversal_via_relative_path_is_refused(tmp_path: Path) -> None:
    _write(tmp_path / "secret.json", {"secret": True})
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    with pytest.raises(RuntimeError, match="outside the project root"):
        build_dataeng_output(
            _state(tmp_path, {"output_metadata": "../secret.json"}, []),
            project_root=root,
        )


def test_invalid_json_artifact_is_reported_clearly(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        build_dataeng_output(
            _state(tmp_path, {"output_metadata": str(path)}, []),
            project_root=tmp_path,
        )


def test_non_object_json_artifact_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path / "meta.json", [1, 2, 3])
    with pytest.raises(RuntimeError, match="must contain a JSON object"):
        build_dataeng_output(
            _state(tmp_path, {"output_metadata": str(tmp_path / "meta.json")}, []),
            project_root=tmp_path,
        )
