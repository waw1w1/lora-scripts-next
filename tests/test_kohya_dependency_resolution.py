import os
import shutil
import subprocess

import pytest

from mikazuki.download_sources import DownloadSources
from mikazuki.engines.kohya.environment import (
    TORCH_PINS,
    XFORMERS_PIN,
    install_commands,
)
from mikazuki.engines.kohya.settings import Runtime


pytestmark = pytest.mark.skipif(
    os.name != "nt" or os.environ.get("RUN_KOHYA_RESOLVER_TEST") != "1",
    reason="online Windows dependency resolution is opt-in",
)


def assert_resolves(command):
    result = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Command failed: {subprocess.list2cmdline(command)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_kohya_training_dependencies_resolve_for_windows_python_310(tmp_path):
    uv = shutil.which("uv")
    assert uv, "uv must be installed to run the Kohya dependency resolver test"

    runtime = Runtime(tmp_path / "project")
    resolver_venv = tmp_path / "resolver-venv"
    subprocess.run(
        [uv, "venv", "--python", "3.10", str(resolver_venv)],
        check=True,
        timeout=60,
    )
    resolver_python = resolver_venv / "Scripts" / "python.exe"
    commands = install_commands(runtime, DownloadSources())

    for command in commands[2:4]:
        dry_run = list(command)
        dry_run[dry_run.index("--python") + 1] = str(resolver_python)
        dry_run.extend(["--dry-run", "--no-cache"])
        assert_resolves(dry_run)

    torch_command = commands[2]
    torch_index = torch_command[torch_command.index("--index-url") + 1]
    combined = list(commands[-1])
    combined[combined.index("--python") + 1] = str(resolver_python)
    combined.extend(
        [
            *TORCH_PINS,
            XFORMERS_PIN,
            "--extra-index-url",
            torch_index,
            "--dry-run",
            "--no-cache",
        ]
    )
    assert_resolves(combined)
