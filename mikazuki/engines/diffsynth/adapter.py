"""UI configuration -> official Qwen-Image-2.1 arguments and dataset metadata."""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import re

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class AdaptedConfig:
    arguments: dict
    dataset: list
    output_path: Path


def absolute(value, root):
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def positive_int(config, key, default):
    value = config.get(key, default)
    number = int(value)
    if number < 1 or float(value) != number:
        raise ValueError(f"{key} 必须为正整数")
    return number


def weight_files(directory, pattern):
    files = sorted(directory.glob(pattern))
    if not files:
        raise ValueError(f"缺少模型权重: {directory / pattern}")
    for index in directory.glob("*.safetensors.index.json"):
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        missing = [name for name in set(mapping.values()) if not (directory / name).is_file()]
        if missing:
            raise ValueError(f"模型分片缺失: {', '.join(sorted(missing))}")
    return [str(path) for path in files]


def adapt_config(config, runtime):
    for field in ("diffsynth_model_dir", "train_data_dir", "output_dir", "output_name"):
        if not str(config.get(field, "")).strip():
            raise ValueError(f"缺少必填参数: {field}")
    model_dir = absolute(config["diffsynth_model_dir"], runtime.project_root)
    models = [weight_files(model_dir / folder, pattern) for folder, pattern in (
        ("transformer", "diffusion_pytorch_model*.safetensors"),
        ("text_encoder", "model*.safetensors"),
        ("vae", "diffusion_pytorch_model*.safetensors"),
    )]
    processor = model_dir / "processor"
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.jinja", "video_preprocessor_config.json"):
        if not (processor / name).is_file():
            raise ValueError(f"缺少 processor 文件: {processor / name}")
    dataset_dir = absolute(config["train_data_dir"], runtime.project_root)
    rows = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        caption = path.with_suffix(".txt")
        if not caption.is_file():
            raise ValueError(f"缺少图片标注: {caption}")
        relative = path.relative_to(dataset_dir)
        repeat_match = re.match(r"^(\d+)_", relative.parts[0]) if len(relative.parts) > 1 else None
        repeats = int(repeat_match[1]) if repeat_match else 1
        if repeats < 1:
            raise ValueError(f"数据集目录重复次数必须大于 0: {relative.parts[0]}")
        rows.extend([{"image": relative.as_posix(), "prompt": caption.read_text(encoding="utf-8-sig").strip()}] * repeats)
    if not rows:
        raise ValueError(f"训练目录中没有图片: {dataset_dir}")
    name = str(config["output_name"])
    if name in {".", ".."} or any(c in name for c in '/\\:'):
        raise ValueError("输出名称应为文件夹名称，不能包含路径分隔符")
    output = absolute(config["output_dir"], runtime.project_root) / name
    lr = float(config.get("learning_rate", 1e-4))
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("学习率必须为正数")
    arguments = {
        "dataset_base_path": str(dataset_dir),
        "data_file_keys": "image",
        "dataset_repeat": positive_int(config, "dataset_repeat", 1),
        # The upstream collate lambda is not spawn-picklable on Windows.
        "dataset_num_workers": 0,
        "model_paths": json.dumps(models, ensure_ascii=False),
        "processor_path": str(processor),
        "max_pixels": positive_int(config, "max_pixels", 1048576),
        "learning_rate": lr,
        "num_epochs": positive_int(config, "num_epochs", 5),
        "gradient_accumulation_steps": positive_int(config, "gradient_accumulation_steps", 1),
        "output_path": str(output),
        "remove_prefix_in_ckpt": "pipe.dit.",
        "lora_base_model": "dit",
        # Empty selects upstream auto-detection, not the parser's legacy q,k,v default.
        "lora_target_modules": str(config.get("lora_target_modules", "")),
        "lora_rank": positive_int(config, "lora_rank", 32),
        "enable_tensorboard_log": True,
        "enable_csv_log": True,
        "find_unused_parameters": True,
    }
    for key, default in (("use_gradient_checkpointing", True), ("use_gradient_checkpointing_offload", False), ("initialize_model_on_cpu", True), ("enable_model_cpu_offload", False)):
        arguments[key] = bool(config.get(key, default))
    if config.get("save_steps"):
        arguments["save_steps"] = positive_int(config, "save_steps", 100)
    if config.get("lora_checkpoint"):
        checkpoint = absolute(config["lora_checkpoint"], runtime.project_root)
        if not checkpoint.is_file():
            raise ValueError(f"LoRA 权重不存在: {checkpoint}")
        arguments["lora_checkpoint"] = str(checkpoint)
    quantization = config.get("diffsynth_quantization", "none")
    if quantization not in {"none", "bitsandbytes_nf4"}:
        raise ValueError(f"不支持的量化方式: {quantization}")
    if quantization != "none":
        arguments["quant_options"] = f"{json.dumps(models[0], ensure_ascii=False)}:{quantization}"
    return AdaptedConfig(arguments, rows, output)


def dump_config(adapted, autosave_dir, run_id):
    directory = Path(autosave_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / f"{run_id}-dataset.json"
    metadata.write_text(json.dumps(adapted.dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    adapted.arguments["dataset_metadata_path"] = str(metadata.resolve())
    config_path = directory / f"{run_id}-arguments.json"
    config_path.write_text(json.dumps(adapted.arguments, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path
