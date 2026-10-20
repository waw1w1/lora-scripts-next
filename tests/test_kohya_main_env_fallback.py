"""Forward compatibility: pre-standalone releases trained in the main
environment (portable python_embeded / GUI interpreter). When the standalone
extensions/kohya env is absent or broken, a passing main-env audit must report
ready instead of forcing a reinstall.
"""
import sys
from pathlib import Path

import pytest

from mikazuki.engines.kohya.settings import Runtime, main_env_python_candidates
from mikazuki.engines.kohya import extension_state
from mikazuki.engines.kohya.extension_state import read_status, write_state, fingerprint


@pytest.fixture(autouse=True)
def clear_audit_cache():
    extension_state._MAIN_ENV_AUDIT_CACHE.clear()
    yield
    extension_state._MAIN_ENV_AUDIT_CACHE.clear()


@pytest.fixture
def main_python(tmp_path):
    python = tmp_path / "python_embeded" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("fake", encoding="utf-8")
    return python


def patch_main_env(monkeypatch, python, audit):
    monkeypatch.setattr(extension_state, "main_env_python_candidates", lambda root: [python])
    monkeypatch.setattr(extension_state, "audit_python", lambda p: audit)


def test_candidates_include_portable_embedded_and_interpreter(tmp_path):
    project_root = tmp_path / "Next-Trainer"
    project_root.mkdir()
    embedded = tmp_path / "python_embeded" / "python.exe"
    embedded.parent.mkdir()
    embedded.write_text("fake", encoding="utf-8")

    candidates = main_env_python_candidates(project_root)

    assert embedded in candidates
    assert Path(sys.executable) in candidates
    assert len(candidates) == len({str(p.resolve()) for p in candidates})


def test_candidates_skip_missing_portable_layout(tmp_path):
    project_root = tmp_path / "Next-Trainer"
    project_root.mkdir()

    assert main_env_python_candidates(project_root) == [Path(sys.executable)]


def test_fallback_ready_when_standalone_missing(tmp_path, monkeypatch, main_python):
    patch_main_env(monkeypatch, main_python, {"ok": True, "torch": "2.7.0", "gpu": True})

    status = read_status(Runtime(tmp_path / "Next-Trainer"))

    assert status["state"] == "ready"
    assert status["facts"]["mode"] == "main_env"
    assert status["facts"]["standalone_state"] == "not_installed"
    assert status["runtime"]["python"] == str(main_python)


def test_fallback_failure_keeps_not_installed(tmp_path, monkeypatch, main_python):
    patch_main_env(monkeypatch, main_python, {"ok": False, "errors": ["no torch"]})

    status = read_status(Runtime(tmp_path / "Next-Trainer"))

    assert status["state"] == "not_installed"
    assert status["facts"]["main_env_rejected"]["python"] == str(main_python)
    assert status["facts"]["main_env_rejected"]["errors"] == ["no torch"]


def test_standalone_ready_wins_without_main_env_audit(tmp_path, monkeypatch, main_python):
    runtime = Runtime(tmp_path / "Next-Trainer")
    runtime.python.parent.mkdir(parents=True)
    runtime.python.write_text("fake", encoding="utf-8")
    write_state(runtime, "ready", {"fingerprint": fingerprint(runtime)})

    def forbidden_audit(python):
        raise AssertionError("main env audit must not run when standalone is ready")

    monkeypatch.setattr(extension_state, "main_env_python_candidates", lambda root: [main_python])
    monkeypatch.setattr(extension_state, "audit_python", forbidden_audit)

    status = read_status(runtime)

    assert status["state"] == "ready"
    assert status["facts"]["mode"] == "standalone"
    assert status["runtime"]["python"] == str(runtime.python)


def test_broken_standalone_falls_back_to_main_env(tmp_path, monkeypatch, main_python):
    runtime = Runtime(tmp_path / "Next-Trainer")
    write_state(runtime, "ready", {"fingerprint": "stale"})
    patch_main_env(monkeypatch, main_python, {"ok": True, "gpu": True})

    status = read_status(runtime)

    assert status["state"] == "ready"
    assert status["facts"]["mode"] == "main_env"
    assert status["facts"]["standalone_state"] == "broken"


def test_interrupted_install_falls_back(tmp_path, monkeypatch, main_python):
    runtime = Runtime(tmp_path / "Next-Trainer")
    write_state(runtime, "installing", {"task_id": "kohya-install-missing"})
    patch_main_env(monkeypatch, main_python, {"ok": True, "gpu": True})

    status = read_status(runtime)

    assert status["state"] == "ready"
    assert status["facts"]["mode"] == "main_env"
    assert status["facts"]["standalone_state"] == "broken"


def test_main_env_audit_results_are_cached(tmp_path, monkeypatch, main_python):
    calls = []

    def audit(python):
        calls.append(str(python))
        return {"ok": True, "gpu": True}

    monkeypatch.setattr(extension_state, "main_env_python_candidates", lambda root: [main_python])
    monkeypatch.setattr(extension_state, "audit_python", audit)
    runtime = Runtime(tmp_path / "Next-Trainer")

    read_status(runtime)
    read_status(runtime)

    assert calls == [str(main_python)]


def test_main_env_audit_failure_cached_briefly(tmp_path, monkeypatch, main_python):
    calls = []

    def audit(python):
        calls.append(str(python))
        return {"ok": False, "errors": ["no torch"]}

    monkeypatch.setattr(extension_state, "main_env_python_candidates", lambda root: [main_python])
    monkeypatch.setattr(extension_state, "audit_python", audit)
    runtime = Runtime(tmp_path / "Next-Trainer")

    read_status(runtime)
    read_status(runtime)

    assert calls == [str(main_python)]


def test_remove_extension_never_touches_main_env(tmp_path, main_python):
    runtime = Runtime(tmp_path / "Next-Trainer")
    runtime.root.mkdir(parents=True)
    write_state(runtime, "ready", {"fingerprint": "x"})

    from mikazuki.engines.kohya.installer import remove_extension

    remove_extension(runtime)

    assert not runtime.root.exists()
    assert main_python.is_file()
