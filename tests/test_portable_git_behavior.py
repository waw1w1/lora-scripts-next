"""Exercise the shipped Git helper against real temporary repos; no downloads."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/portable/portable_git.py"
DATA = {
    "sd-models/model.safetensors": b"model\x00weights",
    "train/images/image.png": b"training image",
    "output/lora.safetensors": b"trained weights",
    "logs/run.log": b"training log",
    "toml/autosave/run.toml": b"toml config",
    "config/autosave/run.toml": b"saved config",
    "custom-dataset/image.png": b"untracked user dataset",
}


class PortableGitBehavior(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="portable git tests ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.origin = self.root / "origin.git"
        self.source = self.root / "source"
        self.package = self.root / "Next-Trainer"
        self.env = os.environ.copy()
        self.env.update(
            {
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_AUTHOR_NAME": "Portable test",
                "GIT_AUTHOR_EMAIL": "portable@example.invalid",
                "GIT_COMMITTER_NAME": "Portable test",
                "GIT_COMMITTER_EMAIL": "portable@example.invalid",
            }
        )
        self.run_cmd("git", "init", "--bare", str(self.origin))
        self.run_cmd("git", "clone", self.origin.as_uri(), str(self.source))
        self.git(self.source, "checkout", "-b", "portable-test")
        self.write(self.source, ".gitignore", (ROOT / ".gitignore").read_bytes())
        self.write(self.source, ".gitattributes", b"*.bat text eol=crlf\n")
        self.write(self.source, "gui.py", b"old code\n")
        self.write(self.source, "docs/guide.md", b"old guide\n")
        self.write(self.source, "tests/example.py", b"test file\n")
        self.write(self.source, "scripts/portable/portable_git.py", HELPER.read_bytes())
        self.commit("initial")
        self.helper("seed", source=True)

    def run_cmd(self, *args, ok=True):
        result = subprocess.run(args, env=self.env, capture_output=True)
        if ok:
            self.assertEqual(
                result.returncode, 0, result.stderr.decode(errors="replace")
            )
        return result

    def git(self, root, *args, ok=True):
        return self.run_cmd("git", "-C", str(root), *args, ok=ok).stdout

    def helper(self, action, ok=True, source=False):
        args = [sys.executable, str(HELPER), action, "--trainer-dir", str(self.package)]
        if source:
            args += ["--source", str(self.source)]
        return self.run_cmd(*args, ok=ok)

    def write(self, root, path, data):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def commit(self, message):
        self.git(self.source, "add", "-A")
        self.git(self.source, "commit", "-m", message)
        self.git(self.source, "push", "origin", "HEAD")

    def incoming(self, path="gui.py", data=b"new code\n"):
        self.write(self.source, path, data)
        self.commit("incoming")
        self.git(self.package, "fetch", "origin", "portable-test", "--deepen=50")

    def put_data(self):
        for path, data in DATA.items():
            self.write(self.package, path, data)

    def legacy_bootstrap(self):
        path = "scripts/portable/portable_git.py"
        self.git(self.source, "rm", path)
        self.commit("old version without helper")
        self.package = self.root / "legacy-package"
        self.helper("seed", source=True)
        self.put_data()
        self.write(self.package, "docs/guide.md", b"saved user notes\n")
        self.git(self.package, "stash", "push", "-m", "existing backup")
        self.incoming(path, HELPER.read_bytes())
        self.write(self.package, path, HELPER.read_bytes())
        return path

    def test_first_upgrade_adopts_new_bootstrap_file(self):
        path = self.legacy_bootstrap()
        before = self.git(self.package, "stash", "list")
        self.helper("update")
        self.assert_data()
        self.assertEqual(self.git(self.package, "stash", "list"), before)
        self.assertEqual(
            self.git(self.package, "status", "--porcelain", "--untracked-files=no"), b""
        )
        self.assertEqual((self.package / path).read_bytes(), HELPER.read_bytes())

    def test_new_bootstrap_with_different_content_is_not_overwritten(self):
        path = self.legacy_bootstrap()
        self.write(self.package, path, b"user-owned content\n")
        before = self.git(self.package, "ls-files", "--stage")
        self.assertNotEqual(self.helper("update", ok=False).returncode, 0)
        self.assertEqual((self.package / path).read_bytes(), b"user-owned content\n")
        self.assertEqual(self.git(self.package, "ls-files", "--stage"), before)
        self.assert_data()

    def test_failed_first_upgrade_restores_index_and_keeps_download(self):
        path = self.legacy_bootstrap()
        self.incoming("gui.py", b"incoming code\n")
        self.write(self.package, "gui.py", b"user edit\n")
        before = self.git(self.package, "ls-files", "--stage")
        stash = self.git(self.package, "stash", "list")
        self.assertNotEqual(self.helper("update", ok=False).returncode, 0)
        self.assertEqual(self.git(self.package, "ls-files", "--stage"), before)
        self.assertEqual(self.git(self.package, "stash", "list"), stash)
        self.assertEqual((self.package / path).read_bytes(), HELPER.read_bytes())
        self.assertEqual((self.package / "gui.py").read_bytes(), b"user edit\n")
        self.assert_data()

    def assert_data(self):
        for path, data in DATA.items():
            self.assertEqual((self.package / path).read_bytes(), data, path)

    def test_seed_is_complete_clean_shallow_checkout(self):
        self.assertTrue((self.package / "docs/guide.md").is_file())
        self.assertTrue((self.package / "tests/example.py").is_file())
        self.assertEqual(self.git(self.package, "status", "--porcelain"), b"")
        self.assertEqual(
            self.git(self.package, "rev-parse", "--is-shallow-repository").strip(),
            b"true",
        )
        self.helper("verify")

    def test_update_preserves_ignored_and_untracked_data(self):
        self.put_data()
        self.incoming()
        self.helper("update")
        self.assert_data()
        self.assertEqual((self.package / "gui.py").read_bytes(), b"new code\n")
        self.assertEqual(self.git(self.package, "stash", "list"), b"")

    def test_missing_ignore_does_not_collect_user_data(self):
        (self.package / ".gitignore").unlink()
        self.put_data()
        self.incoming()
        self.helper("update")
        self.assert_data()
        self.assertEqual(self.git(self.package, "stash", "list"), b"")

    def assert_conflict_preserved(self, path):
        self.put_data()
        self.write(self.package, "gui.py", b"saved old edit\n")
        self.git(self.package, "stash", "push", "-m", "existing user stash")
        self.git(self.package, "config", "merge.autostash", "true")
        self.write(self.package, path, b"user data\n")
        self.write(self.source, path, b"incoming conflicting data\n")
        self.git(self.source, "add", "--force", "--", path)
        self.commit("conflict")
        self.git(self.package, "fetch", "origin", "portable-test", "--deepen=50")
        before = {
            name: self.git(self.package, *cmd)
            for name, cmd in {
                "head": ("rev-parse", "HEAD"),
                "index": ("ls-files", "--stage"),
                "stash": ("stash", "list"),
            }.items()
        }
        self.assertNotEqual(self.helper("update", ok=False).returncode, 0)
        self.assertEqual((self.package / path).read_bytes(), b"user data\n")
        self.assertEqual(self.git(self.package, "rev-parse", "HEAD"), before["head"])
        self.assertEqual(self.git(self.package, "ls-files", "--stage"), before["index"])
        self.assertEqual(self.git(self.package, "stash", "list"), before["stash"])

    def test_tracked_conflict_preserves_user_edits_and_stash(self):
        self.assert_conflict_preserved("gui.py")

    def test_untracked_conflict_preserves_user_data_and_stash(self):
        self.assert_conflict_preserved("custom-dataset/image.png")

    def test_ignored_conflict_preserves_user_data_and_stash(self):
        self.assert_conflict_preserved("sd-models/model.safetensors")

    def test_ignored_directory_blocking_incoming_file_is_preserved(self):
        self.write(self.package, "output/result/private.bin", b"private output")
        self.write(self.source, "output/result", b"incoming file")
        self.git(self.source, "add", "--force", "output/result")
        self.commit("file directory collision")
        self.git(self.package, "fetch", "origin", "portable-test", "--deepen=50")
        self.assertNotEqual(self.helper("update", ok=False).returncode, 0)
        self.assertEqual(
            (self.package / "output/result/private.bin").read_bytes(), b"private output"
        )

    def test_old_cropped_package_is_repaired_and_updated_without_stashing(self):
        self.put_data()
        (self.package / "docs/guide.md").unlink()
        self.incoming("docs/guide.md", b"changed guide\n")
        self.helper("update")
        self.assert_data()
        self.assertEqual(
            (self.package / "docs/guide.md").read_bytes(), b"changed guide\n"
        )
        self.assertEqual(self.git(self.package, "stash", "list"), b"")

    def test_bootstrap_file_already_matching_incoming_is_allowed(self):
        path = "scripts/portable/portable_git.py"
        content = HELPER.read_bytes() + b"\n# updater revision\n"
        self.incoming(path, content)
        self.write(self.package, path, content)
        self.helper("update")
        self.assertEqual(self.git(self.package, "status", "--porcelain"), b"")

    def test_bootstrap_staging_is_undone_when_merge_fails(self):
        path = "scripts/portable/portable_git.py"
        content = HELPER.read_bytes() + b"\n# bootstrap revision\n"
        self.write(self.source, path, content)
        self.incoming()
        self.write(self.package, path, content)
        self.write(self.package, "gui.py", b"user edit\n")
        before = self.git(self.package, "ls-files", "--stage")
        self.assertNotEqual(self.helper("update", ok=False).returncode, 0)
        self.assertEqual(self.git(self.package, "ls-files", "--stage"), before)
        self.assertEqual((self.package / path).read_bytes(), content)
        self.assertEqual((self.package / "gui.py").read_bytes(), b"user edit\n")

    def test_missing_program_path_blocked_by_user_file_is_preserved(self):
        (self.package / "docs/guide.md").unlink()
        (self.package / "docs").rmdir()
        self.write(self.package, "docs", b"user document")
        self.incoming()
        self.assertNotEqual(self.helper("update", ok=False).returncode, 0)
        self.assertEqual((self.package / "docs").read_bytes(), b"user document")

    def test_success_preserves_existing_stash_and_unrelated_edit(self):
        self.write(self.package, "gui.py", b"saved edit\n")
        self.git(self.package, "stash", "push", "-m", "user backup")
        before = self.git(self.package, "stash", "list")
        self.write(self.package, "docs/guide.md", b"user notes\n")
        self.incoming()
        self.helper("update")
        self.assertEqual(self.git(self.package, "stash", "list"), before)
        self.assertEqual((self.package / "docs/guide.md").read_bytes(), b"user notes\n")

    def test_no_successful_fetch_leaves_data_untouched(self):
        self.put_data()
        self.assertNotEqual(self.helper("update", ok=False).returncode, 0)
        self.assert_data()

    @unittest.skipUnless(os.name == "nt", "requires Windows cmd.exe")
    def test_windows_batch_entrypoint_updates_with_user_data(self):
        self.put_data()
        self.write(self.source, "gui.py", b"batch update\n")
        self.commit("batch update")
        launcher = self.root / "Update-Next-Trainer.bat"
        launcher.write_bytes(
            (ROOT / "build-scripts/templates/Update-Next-Trainer.bat").read_bytes()
        )
        self.run_cmd(
            "cmd",
            "/c",
            "mklink",
            "/J",
            str(self.root / "python_embeded"),
            sys.base_prefix,
        )
        result = subprocess.run(
            ["cmd", "/c", str(launcher), "--no-bootstrap"],
            env=self.env,
            input=b"\n",
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            (result.stdout + result.stderr).decode(errors="replace"),
        )
        self.assert_data()
        self.assertEqual((self.package / "gui.py").read_bytes(), b"batch update\n")
        self.assertEqual(self.git(self.package, "stash", "list"), b"")

    def test_mirror_fetch_uses_fetch_head_not_stale_origin(self):
        old = self.git(self.package, "rev-parse", "origin/portable-test")
        self.write(self.source, "gui.py", b"mirror update\n")
        self.commit("mirror")
        self.git(
            self.package, "fetch", self.origin.as_uri(), "portable-test", "--deepen=50"
        )
        self.assertEqual(
            self.git(self.package, "rev-parse", "origin/portable-test"), old
        )
        self.helper("update")
        self.assertEqual((self.package / "gui.py").read_bytes(), b"mirror update\n")

    def test_diverged_history_stops_without_touching_data(self):
        self.put_data()
        self.write(self.package, "local.txt", b"local commit")
        self.git(self.package, "add", "local.txt")
        self.git(self.package, "commit", "-m", "local")
        self.incoming()
        before = self.git(self.package, "rev-parse", "HEAD")
        self.assertNotEqual(self.helper("update", ok=False).returncode, 0)
        self.assertEqual(self.git(self.package, "rev-parse", "HEAD"), before)
        self.assert_data()

    def test_verifier_rejects_missing_tracked_files_and_ignore_rules(self):
        (self.package / "docs/guide.md").unlink()
        self.assertNotEqual(self.helper("verify", ok=False).returncode, 0)
        self.git(self.package, "restore", "docs/guide.md")
        (self.package / ".gitignore").unlink()
        self.assertNotEqual(self.helper("verify", ok=False).returncode, 0)
        self.write(self.package, ".gitignore", b"# no protection\n")
        self.git(self.package, "add", ".gitignore")
        self.git(self.package, "commit", "-m", "bad rules")
        self.assertNotEqual(self.helper("verify", ok=False).returncode, 0)

    def test_seed_refuses_existing_destination(self):
        self.put_data()
        self.assertNotEqual(self.helper("seed", source=True, ok=False).returncode, 0)
        self.assert_data()

    def test_seed_uses_upstream_branch_for_build_worktree(self):
        self.git(
            self.source,
            "checkout",
            "-b",
            "temporary-build",
            "--track",
            "origin/portable-test",
        )
        self.package = self.root / "second-package"
        self.helper("seed", source=True)
        self.assertEqual(
            self.git(self.package, "branch", "--show-current").strip(), b"portable-test"
        )

    def test_seed_rejects_uncommitted_source(self):
        self.package = self.root / "second-package"
        self.write(self.source, "gui.py", b"uncommitted code")
        self.assertNotEqual(self.helper("seed", source=True, ok=False).returncode, 0)
        self.assertFalse(self.package.exists())

    def test_seed_rejects_mismatched_remote_commit(self):
        self.package = self.root / "second-package"
        self.write(self.source, "gui.py", b"unpublished code")
        self.git(self.source, "add", "gui.py")
        self.git(self.source, "commit", "-m", "unpublished")
        self.assertNotEqual(self.helper("seed", source=True, ok=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
