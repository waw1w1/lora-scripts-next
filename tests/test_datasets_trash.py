from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mikazuki.app.application import app
from mikazuki.app.config import app_config


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    dataset_dir = root / "ds"
    nested = dataset_dir / "sub"
    nested.mkdir(parents=True)
    (dataset_dir / "a.png").write_bytes(b"png-a")
    (dataset_dir / "a.txt").write_text("1girl", encoding="utf-8")
    (nested / "b.jpg").write_bytes(b"jpg-b")
    monkeypatch.setitem(app_config._stored, "datasets_root", str(root))
    monkeypatch.setattr(app_config, "save_config", lambda: None)
    return root, dataset_dir


def delete(client: TestClient, paths: list[str]):
    return client.request("DELETE", "/api/datasets/ds/files", json={"paths": paths})


def test_soft_delete_moves_image_with_linked_caption(workspace):
    root, dataset_dir = workspace
    client = TestClient(app)
    response = delete(client, ["a.png"])

    assert response.status_code == 200
    data = response.json()["data"]
    assert sorted(data["deleted"]) == ["a.png", "a.txt"]
    assert not (dataset_dir / "a.png").exists()
    assert not (dataset_dir / "a.txt").exists()
    batch_dir = root / ".trash" / data["batch"]
    assert (batch_dir / "files" / "a.png").is_file()
    assert (batch_dir / "files" / "a.txt").is_file()
    assert (batch_dir / "manifest.json").is_file()


def test_soft_delete_nested_and_missing(workspace):
    _root, dataset_dir = workspace
    client = TestClient(app)
    response = delete(client, ["sub/b.jpg", "ghost.png"])

    data = response.json()["data"]
    assert data["deleted"] == ["sub/b.jpg"]
    assert data["missing"] == ["ghost.png"]
    assert not (dataset_dir / "sub/b.jpg").exists()


def test_trash_list_and_restore_roundtrip(workspace):
    _root, dataset_dir = workspace
    client = TestClient(app)
    batch = delete(client, ["a.png"]).json()["data"]["batch"]

    listing = client.get("/api/datasets/ds/trash").json()["data"]["batches"]
    assert len(listing) == 1
    assert listing[0]["id"] == batch
    assert sorted(listing[0]["paths"]) == ["a.png", "a.txt"]

    restored = client.post("/api/datasets/ds/trash/restore", json={"id": batch}).json()["data"]
    assert sorted(restored["restored"]) == ["a.png", "a.txt"]
    assert restored["conflicts"] == []
    assert (dataset_dir / "a.png").read_bytes() == b"png-a"
    assert client.get("/api/datasets/ds/trash").json()["data"]["batches"] == []


def test_restore_conflict_does_not_overwrite(workspace):
    root, dataset_dir = workspace
    client = TestClient(app)
    batch = delete(client, ["a.png"]).json()["data"]["batch"]
    (dataset_dir / "a.png").write_bytes(b"new-content")

    result = client.post("/api/datasets/ds/trash/restore", json={"id": batch}).json()["data"]
    assert result["restored"] == ["a.txt"]
    assert result["conflicts"] == ["a.png"]
    assert (dataset_dir / "a.png").read_bytes() == b"new-content"
    assert (root / ".trash" / batch / "files" / "a.png").is_file()

    remaining = client.get("/api/datasets/ds/trash").json()["data"]["batches"]
    assert remaining[0]["paths"] == ["a.png"]


def test_trash_empty_requires_confirm(workspace):
    root, _dataset_dir = workspace
    client = TestClient(app)
    delete(client, ["a.png"])

    assert client.post("/api/datasets/ds/trash/empty", json={}).status_code == 400

    result = client.post("/api/datasets/ds/trash/empty", json={"confirm": True}).json()["data"]
    assert result["removed"] == 1
    assert not list((root / ".trash").iterdir())


def test_trash_empty_single_batch(workspace):
    _root, _dataset_dir = workspace
    client = TestClient(app)
    first = delete(client, ["a.png"]).json()["data"]["batch"]
    second = delete(client, ["sub/b.jpg"]).json()["data"]["batch"]

    result = client.post("/api/datasets/ds/trash/empty", json={"confirm": True, "id": first}).json()["data"]
    assert result["removed"] == 1
    batches = client.get("/api/datasets/ds/trash").json()["data"]["batches"]
    assert [b["id"] for b in batches] == [second]


def test_trash_isolated_between_datasets(workspace):
    root, dataset_dir = workspace
    other = root / "other"
    other.mkdir()
    (other / "x.png").write_bytes(b"x")
    client = TestClient(app)
    delete(client, ["a.png"])

    assert client.get("/api/datasets/other/trash").json()["data"]["batches"] == []
    ghost = client.post("/api/datasets/other/trash/restore", json={"id": "whatever"})
    assert ghost.status_code == 404
    assert dataset_dir.name == "ds"


def test_trash_hidden_from_listing_and_export(workspace):
    root, _dataset_dir = workspace
    client = TestClient(app)
    delete(client, ["a.png"])

    names = [d["name"] for d in client.get("/api/datasets").json()["data"]["datasets"]]
    assert names == ["ds"]

    import io
    import zipfile

    response = client.get("/api/datasets/ds/download")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert ".trash" not in "".join(archive.namelist())


def test_delete_dataset_moves_everything_to_trash(workspace):
    root, dataset_dir = workspace
    client = TestClient(app)
    response = client.delete("/api/datasets/ds")

    assert response.status_code == 200
    data = response.json()["data"]
    assert sorted(data["deleted"]) == ["a.png", "a.txt", "sub/b.jpg"]
    assert not dataset_dir.exists()
    assert [d["name"] for d in client.get("/api/datasets").json()["data"]["datasets"]] == []

    batches = client.get("/api/datasets-trash").json()["data"]["batches"]
    assert len(batches) == 1
    assert batches[0]["dataset"] == "ds"
    assert batches[0]["count"] == 3


def test_global_restore_recreates_deleted_dataset(workspace):
    root, dataset_dir = workspace
    client = TestClient(app)
    batch = client.delete("/api/datasets/ds").json()["data"]["batch"]

    result = client.post("/api/datasets-trash/restore", json={"id": batch}).json()["data"]
    assert result["dataset"] == "ds"
    assert sorted(result["restored"]) == ["a.png", "a.txt", "sub/b.jpg"]
    assert (dataset_dir / "sub" / "b.jpg").read_bytes() == b"jpg-b"
    assert [d["name"] for d in client.get("/api/datasets").json()["data"]["datasets"]] == ["ds"]


def test_global_restore_conflict_keeps_batch(workspace):
    root, dataset_dir = workspace
    client = TestClient(app)
    batch = client.delete("/api/datasets/ds").json()["data"]["batch"]
    (dataset_dir / "sub").mkdir(parents=True)
    (dataset_dir / "sub" / "b.jpg").write_bytes(b"newer")

    result = client.post("/api/datasets-trash/restore", json={"id": batch}).json()["data"]
    assert sorted(result["restored"]) == ["a.png", "a.txt"]
    assert result["conflicts"] == ["sub/b.jpg"]
    assert (dataset_dir / "sub" / "b.jpg").read_bytes() == b"newer"
    batches = client.get("/api/datasets-trash").json()["data"]["batches"]
    assert batches[0]["paths"] == ["sub/b.jpg"]


def test_global_empty_trash(workspace):
    root, _dataset_dir = workspace
    client = TestClient(app)
    client.delete("/api/datasets/ds")

    assert client.post("/api/datasets-trash/empty", json={}).status_code == 400
    result = client.post("/api/datasets-trash/empty", json={"confirm": True}).json()["data"]
    assert result["removed"] == 1
    assert not (root / ".trash").exists() or not list((root / ".trash").iterdir())
