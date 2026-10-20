import json
import hashlib
import time

from .environment import TRAINING_DEPS, TORCH_PINS, TORCH_PINS_AARCH64, audit_python
from .settings import main_env_python_candidates


def write_state(runtime, state, facts=None, reason=""):
    runtime.root.mkdir(parents=True, exist_ok=True)
    temporary = runtime.state_file.with_suffix(".tmp")
    temporary.write_text(json.dumps({"state": state, "facts": facts or {}, "reason": reason}, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(runtime.state_file)


def fingerprint(runtime):
    # The training source ships with the repository and updates via git pull;
    # only the venv content and the pinned dependency set gate readiness.
    files = [runtime.python]
    files += sorted((runtime.root / ".venv").glob("**/*.dist-info/METADATA"))
    stats = []
    for path in files:
        if not path.is_file():
            continue
        try:
            identity = path.relative_to(runtime.root).as_posix()
        except ValueError:
            identity = path.name
        stats.append((identity, path.stat().st_size, path.stat().st_mtime_ns))
    pins = [*TRAINING_DEPS, *TORCH_PINS, *TORCH_PINS_AARCH64]
    return hashlib.sha256(json.dumps([pins, stats]).encode()).hexdigest()


# Main-env audits spawn a python subprocess importing torch (seconds), so
# results are cached in-process: successes forever, failures briefly so a
# fixed main env is picked up without requiring a GUI restart.
_MAIN_ENV_AUDIT_CACHE = {}
_MAIN_ENV_FAILURE_TTL = 120.0


def _cached_main_env_audit(python):
    key = str(python)
    hit = _MAIN_ENV_AUDIT_CACHE.get(key)
    if hit:
        ok, payload, cached_at = hit
        if ok or time.monotonic() - cached_at < _MAIN_ENV_FAILURE_TTL:
            return payload
    payload = audit_python(python)
    _MAIN_ENV_AUDIT_CACHE[key] = (bool(payload.get("ok")), payload, time.monotonic())
    return payload


def _main_env_fallback(runtime, standalone_state):
    """Legacy forward compatibility: pre-standalone releases trained in the
    main environment (portable ``python_embeded`` or the GUI interpreter). If
    that environment already passes the training audit, report ready instead
    of forcing a multi-GB reinstall into ``extensions/kohya``.

    Returns ``(ready_payload, None)`` when a candidate passes, otherwise
    ``(None, diagnostics)`` with the last rejection for support visibility.
    """

    standalone_python = None
    try:
        standalone_python = str(runtime.python.resolve())
    except OSError:
        pass
    last_failure = None
    for python in main_env_python_candidates(runtime.project_root):
        try:
            if str(python.resolve()) == standalone_python:
                continue
        except OSError:
            pass
        audit = _cached_main_env_audit(python)
        if audit.get("ok"):
            return {
                "state": "ready",
                "facts": {
                    "mode": "main_env",
                    "main_env_python": str(python),
                    "audit": audit,
                    "standalone_state": standalone_state,
                },
                "reason": "",
                "feature_enabled": True,
                "runtime": {"python": str(python), "environment_path": str(runtime.root)},
            }, None
        last_failure = {"python": str(python), "errors": audit.get("errors", [])}
    diagnostics = {"main_env_rejected": last_failure} if last_failure else None
    return None, diagnostics


def read_status(runtime):
    data = json.loads(runtime.state_file.read_text(encoding="utf-8")) if runtime.state_file.exists() else {"state": "not_installed", "facts": {}}
    if data["state"] in {"installing", "auditing"}:
        from mikazuki.tasks import tm, TaskStatus
        task = tm.tasks.get(data["facts"].get("task_id"))
        if task is None or task.status in {TaskStatus.FAILED, TaskStatus.TERMINATED, TaskStatus.FINISHED}:
            data.update(state="broken", reason="安装中断，请修复 kohya 环境。")
    if data["state"] == "ready" and not runtime.python.is_file():
        data.update(state="broken", reason="独立 Python 缺失，请修复环境。")
    if data["state"] == "ready" and data["facts"].get("fingerprint") != fingerprint(runtime):
        data.update(state="broken", reason="环境已变化，请修复后重新检查。")
    if data["state"] == "ready":
        data["facts"]["mode"] = "standalone"
        data["feature_enabled"] = True
        data["runtime"] = {"python": str(runtime.python), "environment_path": str(runtime.root)}
        return data
    if data["state"] in {"not_installed", "broken"}:
        fallback, diagnostics = _main_env_fallback(runtime, data["state"])
        if fallback is not None:
            return fallback
        if diagnostics:
            data["facts"] = {**data.get("facts", {}), **diagnostics}
    data["feature_enabled"] = True
    data["runtime"] = {"python": str(runtime.python), "environment_path": str(runtime.root)}
    return data
