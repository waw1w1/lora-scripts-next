from .extension_state import read_status
from .environment import audit_python


def check_runtime(runtime):
    status = read_status(runtime)
    if status["state"] != "ready":
        raise ValueError("kohya (sd-scripts) 环境未就绪，请到训练引擎管理中安装或修复。")
    if status["facts"].get("mode") == "main_env":
        # read_status already audited the fallback environment.
        return
    audit = audit_python(status["runtime"]["python"])
    if not audit["ok"]:
        raise ValueError("kohya 环境检查失败: " + "; ".join(audit["errors"]))
