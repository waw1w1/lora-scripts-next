import json

from .settings import TRAIN_SCRIPT, feature_enabled


def write_state(runtime, state, facts=None, reason=""):
    runtime.root.mkdir(parents=True, exist_ok=True)
    runtime.state_file.write_text(json.dumps({"state": state, "facts": facts or {}, "reason": reason}, ensure_ascii=False, indent=2), encoding="utf-8")


def read_status(runtime):
    data = json.loads(runtime.state_file.read_text(encoding="utf-8")) if runtime.state_file.exists() else {"state": "not_installed", "facts": {}}
    if data["state"] in {"installing", "auditing"}:
        from mikazuki.tasks import tm, TaskStatus
        task = tm.tasks.get(data["facts"].get("task_id"))
        if task is None or task.status in {TaskStatus.FAILED, TaskStatus.TERMINATED, TaskStatus.FINISHED}:
            data.update(state="broken", reason="安装中断，请修复 DiffSynth 环境。")
    if data["state"] == "ready" and not (runtime.python.is_file() and (runtime.source / TRAIN_SCRIPT).is_file()):
        data.update(state="broken", reason="训练入口或独立 Python 缺失，请修复环境。")
    data["feature_enabled"] = feature_enabled()
    if not data["feature_enabled"]:
        data["state"] = "disabled"
    data["runtime"] = {"python": str(runtime.python), "environment_path": str(runtime.root), "source": str(runtime.source)}
    return data
