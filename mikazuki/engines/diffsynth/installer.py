"""Install the complete pinned upstream checkout, leaving training code unchanged."""
import json
import shutil
import subprocess
import threading
import uuid

from mikazuki.download_sources import DownloadSources, apply_github_prefix
from mikazuki.tasks import tm, LANE_MAINTENANCE, TaskStatus
from mikazuki.train_log_hub import hub
from .environment import install_commands, install_env, process_env, audit_environment
from .extension_state import write_state
from .manifest import UPSTREAM


def assert_idle():
    if any(t.metadata.get("backend") == "diffsynth" and t.status in {TaskStatus.CREATED, TaskStatus.QUEUED, TaskStatus.RUNNING} for t in tm.tasks.values()):
        raise ValueError("DiffSynth 任务尚未结束，请先停止任务。")


def installation_plan(runtime, sources):
    return [
        ["git", "clone", "--no-checkout", apply_github_prefix(UPSTREAM["github"], sources.github_url_prefix), str(runtime.source)],
        ["git", "-C", str(runtime.source), "checkout", "--detach", UPSTREAM["commit"]],
        *install_commands(runtime, sources),
    ]


def start_install(runtime, sources=None, repair=False):
    assert_idle()
    sources = sources or DownloadSources()
    task_id = f"diffsynth-install-{uuid.uuid4()}"
    task = tm.create_task(["diffsynth-install"], process_env(), metadata={"backend": "diffsynth", "kind": "diffsynth_install"}, task_id=task_id, lane=LANE_MAINTENANCE)
    task.start_log_only()
    write_state(runtime, "installing", {"task_id": task_id})

    def work():
        try:
            if repair:
                for path in (runtime.source, runtime.root / ".venv", runtime.python_install_dir):
                    if path.exists():
                        shutil.rmtree(path)
            commands = installation_plan(runtime, sources)
            if runtime.source.exists():
                commands = [["git", "-C", str(runtime.source), "fetch", "origin", UPSTREAM["commit"]], *commands[1:]]
            for i, command in enumerate(commands):
                hub.append_event(task_id, {"type": "progress", "percent": int(i * 90 / len(commands)), "message": " ".join(command)})
                hub.append_line(task_id, "[install] " + " ".join(command))
                with subprocess.Popen(command, env=install_env(runtime), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as proc:
                    for line in proc.stdout:
                        hub.append_line(task_id, line.rstrip())
                    if proc.wait():
                        raise RuntimeError(f"安装命令失败（{proc.returncode}）: {' '.join(command)}")
            write_state(runtime, "auditing", {"task_id": task_id})
            audit = audit_environment(runtime)
            write_state(runtime, "ready" if audit["ok"] else "broken", {"task_id": task_id, "audit": audit, "source_commit": UPSTREAM["commit"]}, "; ".join(audit["errors"]))
            hub.append_line(task_id, json.dumps(audit, ensure_ascii=False))
            hub.append_event(task_id, {"type": "progress", "percent": 100, "message": "环境检查完成"})
            task.finish_log_only(0 if audit["ok"] else 1, None if audit["ok"] else "; ".join(audit["errors"]))
        except Exception as exc:
            write_state(runtime, "broken", {"task_id": task_id}, str(exc))
            hub.append_line(task_id, f"[error] {exc}")
            task.finish_log_only(1, exc)

    threading.Thread(target=work, daemon=True).start()
    return {"task_id": task_id, "log_stream": f"/api/engines/diffsynth/install/log/stream/{task_id}", "progress_stream": f"/api/engines/diffsynth/install/progress/stream/{task_id}"}


def remove_extension(runtime):
    assert_idle()
    if runtime.root.exists():
        shutil.rmtree(runtime.root)
