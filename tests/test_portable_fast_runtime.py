import json
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]


def runtime():
    path = ROOT / "mikazuki/engines/anima_fast/portable_runtime.py"
    assert path.is_file(), "portable relocation support missing"
    return runpy.run_path(str(path))


def fixture(root):
    (root / ".python").mkdir(parents=True)
    (root / ".python/python.exe").touch()
    (root / ".venv").mkdir()
    (root / ".venv/pyvenv.cfg").write_text(
        "home = C:\\outside\\Python\ninclude-system-site-packages = false\n"
        "version = 3.13.9\nexecutable = C:\\outside\\Python\\python.exe\n")
    (root / "portable-runtime.json").write_text(json.dumps({"version": 1}))


def test_relocate_repairs_only_marked_runtime(tmp_path):
    fixture(tmp_path)
    runtime()["repair"](tmp_path)
    text = (tmp_path / ".venv/pyvenv.cfg").read_text()
    assert str(tmp_path / ".python") in text
    assert "outside" not in text
    assert "include-system-site-packages = false" in text
    moved = tmp_path.parent / (tmp_path.name + " moved")
    tmp_path.rename(moved)
    runtime()["repair"](moved)
    assert str(moved / ".python") in (moved / ".venv/pyvenv.cfg").read_text()


def test_unmarked_developer_environment_is_untouched(tmp_path):
    fixture(tmp_path)
    (tmp_path / "portable-runtime.json").unlink()
    before = (tmp_path / ".venv/pyvenv.cfg").read_bytes()
    runtime()["repair"](tmp_path)
    assert (tmp_path / ".venv/pyvenv.cfg").read_bytes() == before


def test_missing_bundled_base_fails_without_rewriting(tmp_path):
    fixture(tmp_path)
    (tmp_path / ".python/python.exe").unlink()
    before = (tmp_path / ".venv/pyvenv.cfg").read_bytes()
    with pytest.raises(RuntimeError, match="base Python"):
        runtime()["repair"](tmp_path)
    assert (tmp_path / ".venv/pyvenv.cfg").read_bytes() == before


def test_fast_profile_launcher_does_not_require_host_torch():
    text = (ROOT / "scripts/portable/launch_portable.bat").read_text(encoding="utf-8")
    assert 'portable-profile.json" goto :fast_profile' in text
    profile = text.split(":fast_profile\n")[-1].split(":first_run")[0]
    assert "verify_fast_package.py" in profile
    assert "goto :launch" in profile
    assert "setup_environment.py" not in profile


def test_fast_builder_installs_gui_after_temporary_cleanup():
    text = (ROOT / "build-scripts/build_portable.ps1").read_text(encoding="utf-8-sig")
    assert "bundle_fast_runtime.py" in text
    install = text.index("# Complete Fast host")
    assert install > text.index("Cleared build-time pip packages")
    assert "verify_fast_package.py" in text[install:]
    assert "portable-profile.json" in text[install:]


def test_verifier_rejects_external_runtime_paths(tmp_path):
    path = ROOT / "scripts/portable/verify_fast_package.py"
    checker = runpy.run_path(str(path))["require_inside"]
    checker(tmp_path / "python.exe", tmp_path)
    with pytest.raises(RuntimeError, match="escapes"):
        checker(tmp_path.parent / "external/python.exe", tmp_path)


def test_main_audit_does_not_import_training_torch(monkeypatch):
    import builtins
    from mikazuki.engines.anima_fast.environment import _main_facts_in_process
    original = builtins.__import__
    calls = []

    def importing(name, *args, **kwargs):
        if name == "torch":
            calls.append(name)
            raise ImportError("GUI intentionally has no torch")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    facts = _main_facts_in_process()
    assert calls == []
    assert "torch" not in facts["imports"]


def test_fast_builder_skips_unrelated_tokenizer_downloads():
    text = (ROOT / "build-scripts/build_portable.ps1").read_text(encoding="utf-8-sig")
    assert "if (-not $BundleAnimaFast) {\n# SD-family tokenizer prefetch" in text


def test_gui_without_transformers_does_not_partially_patch_modelscope(monkeypatch):
    import builtins
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    import mikazuki.china_hub as hub

    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("GUI-only environment")
        return original(name, *args, **kwargs)

    patch = Mock(side_effect=ImportError("transformers missing inside patch_hub"))
    aliases = Mock()
    monkeypatch.setitem(sys.modules, "modelscope.utils.hf_util", SimpleNamespace(patch_hub=patch))
    monkeypatch.setattr(builtins, "__import__", importing)
    monkeypatch.setattr(hub, "_PATCHED", False)
    monkeypatch.setattr(hub, "_patch_modelscope_download_aliases", aliases)
    assert hub.enable_china_hub(force=True) is False
    patch.assert_not_called()
    aliases.assert_not_called()
