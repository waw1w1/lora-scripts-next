"""#410: the DiffSynth runtime fingerprint must not depend on file timestamps."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from mikazuki.engines.diffsynth.extension_state import (
    FINGERPRINT_SCHEME,
    fingerprint,
    read_status,
    write_state,
)
from mikazuki.engines.diffsynth.settings import TRAIN_SCRIPT, Runtime


def make_runtime(project_root: Path) -> Runtime:
    runtime = Runtime(project_root)
    (runtime.source / "diffsynth" / "models").mkdir(parents=True, exist_ok=True)
    (runtime.source / TRAIN_SCRIPT).parent.mkdir(parents=True, exist_ok=True)
    (runtime.source / TRAIN_SCRIPT).write_text("print('train')\n", encoding="utf-8")
    (runtime.source / "pyproject.toml").write_text("[project]\nname = 'diffsynth'\n", encoding="utf-8")
    (runtime.source / "diffsynth" / "models" / "model.py").write_text("x = 1\n", encoding="utf-8")
    runtime.python.parent.mkdir(parents=True, exist_ok=True)
    runtime.python.write_text("#!/python\n", encoding="utf-8")
    metadata = runtime.root / ".venv" / "lib" / "torch-2.8.0.dist-info"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "METADATA").write_text("Name: torch\nVersion: 2.8.0\n", encoding="utf-8")
    return runtime


class FingerprintContentTests(unittest.TestCase):
    def test_mtime_only_changes_keep_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            runtime = make_runtime(Path(td))
            before = fingerprint(runtime)

            for path in (
                runtime.source / TRAIN_SCRIPT,
                runtime.source / "diffsynth" / "models" / "model.py",
                runtime.root / ".venv" / "lib" / "torch-2.8.0.dist-info" / "METADATA",
            ):
                stat = path.stat()
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

            self.assertEqual(fingerprint(runtime), before)

    def test_same_length_content_change_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            runtime = make_runtime(Path(td))
            source = runtime.source / "diffsynth" / "models" / "model.py"
            before = fingerprint(runtime)

            source.write_text("x = 2\n", encoding="utf-8")

            self.assertNotEqual(fingerprint(runtime), before)

    def test_symlinked_python_is_fingerprinted_by_target(self):
        with tempfile.TemporaryDirectory() as td:
            runtime = make_runtime(Path(td))
            target = Path(td) / "outside-python"
            target.write_text("#!/python\n", encoding="utf-8")
            runtime.python.unlink()
            try:
                runtime.python.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable on this platform")

            self.assertEqual(len(fingerprint(runtime)), 64)


class FingerprintMigrationTests(unittest.TestCase):
    def test_legacy_state_is_refreshed_instead_of_broken(self):
        with tempfile.TemporaryDirectory() as td:
            runtime = make_runtime(Path(td))
            # State written before the content scheme: hash of the old ingredients,
            # no scheme marker at all.
            write_state(runtime, "ready", {"fingerprint": "0" * 64, "source_commit": "abc"}, "audit passed")

            status = read_status(runtime)
            persisted = json.loads(runtime.state_file.read_text(encoding="utf-8"))

        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["facts"]["fingerprint_scheme"], FINGERPRINT_SCHEME)
        self.assertEqual(persisted["facts"]["fingerprint_scheme"], FINGERPRINT_SCHEME)
        self.assertEqual(persisted["facts"]["source_commit"], "abc")

    def test_matching_state_stays_ready_without_rewriting(self):
        with tempfile.TemporaryDirectory() as td:
            runtime = make_runtime(Path(td))
            write_state(runtime, "ready", {"fingerprint": fingerprint(runtime), "fingerprint_scheme": FINGERPRINT_SCHEME}, "")
            before = runtime.state_file.read_text(encoding="utf-8")

            status = read_status(runtime)
            after = runtime.state_file.read_text(encoding="utf-8")

        self.assertEqual(status["state"], "ready")
        self.assertEqual(after, before)

    def test_current_scheme_mismatch_is_still_broken(self):
        with tempfile.TemporaryDirectory() as td:
            runtime = make_runtime(Path(td))
            write_state(runtime, "ready", {"fingerprint": "0" * 64, "fingerprint_scheme": FINGERPRINT_SCHEME}, "")

            status = read_status(runtime)

        self.assertEqual(status["state"], "broken")
        self.assertIn("已变化", status["reason"])

    def test_missing_runtime_files_are_still_broken(self):
        with tempfile.TemporaryDirectory() as td:
            runtime = make_runtime(Path(td))
            write_state(runtime, "ready", {"fingerprint": fingerprint(runtime), "fingerprint_scheme": FINGERPRINT_SCHEME}, "")
            (runtime.source / TRAIN_SCRIPT).unlink()

            status = read_status(runtime)

        self.assertEqual(status["state"], "broken")
        self.assertIn("训练入口", status["reason"])


if __name__ == "__main__":
    unittest.main()
