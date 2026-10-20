from .extension_state import read_status
from .environment import audit_python


def check_runtime(runtime):
    status = read_status(runtime)
    if status["state"] != "ready":
        raise ValueError("kohya (sd-scripts) 环境未就绪，请到训练引擎管理中安装或修复。")
    # Explicit preflight always re-audits the selected interpreter live; the
    # status-poll audit cache may be stale by the time a run is submitted.
    audit = audit_python(status["runtime"]["python"], allow_user_site=status["facts"].get("mode") == "main_env")
    if not audit["ok"]:
        raise ValueError("kohya 环境检查失败: " + "; ".join(audit["errors"]))
