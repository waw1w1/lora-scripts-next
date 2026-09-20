from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mikazuki.app.application import app
from mikazuki.app.config import app_config
from mikazuki.datasets import root as root_module


@pytest.fixture
def datasets_root(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    monkeypatch.setitem(app_config._stored, "datasets_root", str(root))
    monkeypatch.setattr(app_config, "save_config", lambda: None)
    return root


def make_dataset(root: Path, name: str):
    dataset_dir = root / name
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "alpha.png").write_bytes(b"png")
    return dataset_dir


def test_default_root_resolves_against_repo_root(monkeypatch):
    monkeypatch.delitem(app_config._stored, "datasets_root", raising=False)
    assert root_module.get_datasets_root() == (root_module.REPO_ROOT / "datasets").resolve()


def test_relative_root_resolves_against_repo_root_not_cwd(monkeypatch, tmp_path):
    monkeypatch.setitem(app_config._stored, "datasets_root", "./my-datasets")
    monkeypatch.chdir(tmp_path)
    assert root_module.get_datasets_root() == (root_module.REPO_ROOT / "my-datasets").resolve()


def test_get_root_reports_configured_path(datasets_root):
    client = TestClient(app)
    response = client.get("/api/datasets/root")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["root"] == str(datasets_root.resolve()).replace("\\", "/")
    assert data["default"] == "./datasets"
    assert data["exists"] is False


def test_update_root_creates_directory_without_touching_old_data(datasets_root, tmp_path):
    old_dir = make_dataset(tmp_path / "old-root", "keep-me")
    new_root = tmp_path / "new-root"

    client = TestClient(app)
    response = client.put("/api/datasets/root", json={"path": str(new_root)})

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["root"] == str(new_root.resolve()).replace("\\", "/")
    assert data["exists"] is True
    assert new_root.is_dir()
    assert (old_dir / "alpha.png").is_file()
    assert app_config["datasets_root"] == str(new_root)


def test_update_root_rejects_empty_path(datasets_root):
    client = TestClient(app)
    response = client.put("/api/datasets/root", json={"path": "   "})
    assert response.status_code == 400


def test_list_datasets_discovers_first_level_dirs(datasets_root):
    make_dataset(datasets_root, "beta")
    make_dataset(datasets_root, "Alpha")
    (datasets_root / ".trash").mkdir(parents=True)
    (datasets_root / "loose-file.txt").write_text("x", encoding="utf-8")

    client = TestClient(app)
    response = client.get("/api/datasets")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["exists"] is True
    assert [d["name"] for d in data["datasets"]] == ["Alpha", "beta"]
    assert all("\\" not in d["path"] for d in data["datasets"])


def test_list_datasets_missing_root_returns_empty(datasets_root):
    client = TestClient(app)
    response = client.get("/api/datasets")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["exists"] is False
    assert data["datasets"] == []


def test_list_datasets_excludes_symlinked_directories(datasets_root, tmp_path):
    make_dataset(datasets_root, "real")
    outside = tmp_path / "outside"
    outside.mkdir()
    (datasets_root / "linked").symlink_to(outside, target_is_directory=True)

    client = TestClient(app)
    data = client.get("/api/datasets").json()["data"]
    assert [d["name"] for d in data["datasets"]] == ["real"]


def test_create_dataset(datasets_root):
    client = TestClient(app)
    response = client.post("/api/datasets", json={"name": "my-dataset"})

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["name"] == "my-dataset"
    assert (datasets_root / "my-dataset").is_dir()


def test_create_dataset_conflict(datasets_root):
    make_dataset(datasets_root, "dup")

    client = TestClient(app)
    response = client.post("/api/datasets", json={"name": "dup"})
    assert response.status_code == 409


@pytest.mark.parametrize("name", ["../escape", "a/b", ".hidden", "", "  "])
def test_create_dataset_rejects_invalid_names(datasets_root, name):
    client = TestClient(app)
    response = client.post("/api/datasets", json={"name": name})
    assert response.status_code == 400
