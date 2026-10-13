"""ai-toolkit preflight: per-variant asset checks (DiT/TE/VAE, dataset)."""

from pathlib import Path
import json
import struct

import pytest
from PIL import Image

from mikazuki.engines.ai_toolkit.adapter import AdapterError, adapt_config
from mikazuki.engines.ai_toolkit.preflight import ProbeFacts, run_preflight
from mikazuki.engines.ai_toolkit.settings import RuntimeConfig


def _runtime(tmp_path: Path) -> RuntimeConfig:
    python = tmp_path / "toolkit" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("")
    (tmp_path / "toolkit" / "run.py").write_text("", encoding="utf-8")
    return RuntimeConfig(
        toolkit_root=tmp_path / "toolkit",
        python=python,
        lora_next_root=tmp_path,
        output_dir=tmp_path / "output",
        logging_dir=tmp_path / "logs",
        cache_dir=tmp_path / ".cache",
    )


def _ok_probe(runtime: RuntimeConfig) -> ProbeFacts:
    return ProbeFacts(
        python_version="3.11.9",
        torch_version="2.13.0+cu130",
        cuda_available=True,
        cuda_version="13.0",
        gpu_name="GB10",
        vram_total_mb=122880,
        transformers_version="5.5.3",
    )


def _weights(path: Path):
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0))


def _setup(tmp_path: Path, with_dit: bool = True, with_vae: bool = True, with_te: bool = True) -> dict:
    data = tmp_path / "train" / "klein"
    data.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(data / "img1.png")
    (data / "img1.txt").write_text("a cat", encoding="utf-8")
    dit_dir = tmp_path / "models"
    dit_dir.mkdir(exist_ok=True)
    if with_dit:
        _weights(dit_dir / "flux-2-klein-base-4b.safetensors")
    if with_vae:
        _weights(dit_dir / "ae.safetensors")
    te_dir = dit_dir / "qwen3-4b"
    if with_te:
        te_dir.mkdir(exist_ok=True)
        (te_dir / "config.json").write_text('{"hidden_size": 2560}', encoding="utf-8")
        (te_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (te_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        _weights(te_dir / "model.safetensors")
    return {
        "pretrained_model_name_or_path": str(dit_dir),
        "text_encoder": str(te_dir),
        "train_data_dir": str(data),
        "max_train_steps": 100,
    }


def test_preflight_ok(tmp_path):
    runtime = _runtime(tmp_path)
    adapted = adapt_config(_setup(tmp_path), runtime, "run-1", "klein-4b")
    result = run_preflight(adapted.config, runtime, "klein-4b", te_path=adapted.te_path, probe=_ok_probe)
    assert result.ok, result.errors
    assert result.facts["dataset_image_count"] == 1
    assert result.facts["model_assets"]["text_encoder"] == adapted.te_path
    assert result.facts["cuda_available"] is True


def test_preflight_missing_local_dit_path_is_error(tmp_path):
    """A typo'd local path is not an HF repo id; it must fail the hard gate."""
    runtime = _runtime(tmp_path)
    source = _setup(tmp_path)
    source["pretrained_model_name_or_path"] = "./sd-models/klein/typo.safetensors"
    with pytest.raises(AdapterError, match="本地路径"):
        adapt_config(source, runtime, "run-1", "klein-4b")


def test_preflight_te_variant_mismatch_is_error(tmp_path):
    runtime = _runtime(tmp_path)
    source = _setup(tmp_path)
    te_dir = Path(source["text_encoder"])
    (te_dir / "config.json").write_text('{"hidden_size": 4096}', encoding="utf-8")
    adapted = adapt_config(source, runtime, "run-1", "klein-4b")
    result = run_preflight(adapted.config, runtime, "klein-4b", te_path=adapted.te_path, probe=_ok_probe)
    assert not result.ok
    assert any("hidden_size=2560" in e for e in result.errors)

    (te_dir / "config.json").write_text('{"hidden_size": 2560}', encoding="utf-8")
    result = run_preflight(adapted.config, runtime, "klein-4b", te_path=adapted.te_path, probe=_ok_probe)
    assert result.ok, result.errors
    assert not any("hidden_size" in w for w in result.warnings)


def test_preflight_missing_dit(tmp_path):
    runtime = _runtime(tmp_path)
    adapted = adapt_config(_setup(tmp_path), runtime, "run-1", "klein-4b")
    # Assets can disappear after adaptation; preflight must recheck them.
    (tmp_path / "models" / "flux-2-klein-base-4b.safetensors").unlink()
    result = run_preflight(adapted.config, runtime, "klein-4b", te_path=adapted.te_path, probe=_ok_probe)
    assert not result.ok
    assert any("flux-2-klein-base-4b.safetensors" in e for e in result.errors)


def test_preflight_missing_vae_is_error(tmp_path):
    runtime = _runtime(tmp_path)
    adapted = adapt_config(_setup(tmp_path, with_vae=False), runtime, "run-1", "klein-4b")
    result = run_preflight(adapted.config, runtime, "klein-4b", te_path=adapted.te_path, probe=_ok_probe)
    assert not result.ok
    assert any("ae.safetensors" in e for e in result.errors)


def test_preflight_missing_te_weights_is_error(tmp_path):
    runtime = _runtime(tmp_path)
    source = _setup(tmp_path)
    te_dir = Path(source["text_encoder"])
    (te_dir / "model.safetensors").unlink()
    adapted = adapt_config(source, runtime, "run-1", "klein-4b")
    result = run_preflight(adapted.config, runtime, "klein-4b", te_path=adapted.te_path, probe=_ok_probe)
    assert not result.ok
    assert any("safetensors" in e for e in result.errors)


def test_preflight_hf_repo_rejected_before_probe(tmp_path):
    runtime = _runtime(tmp_path)
    source = _setup(tmp_path)
    source["pretrained_model_name_or_path"] = "black-forest-labs/FLUX.2-klein-base-4B"
    with pytest.raises(AdapterError, match="不会自动下载或转换"):
        adapt_config(source, runtime, "run-1", "klein-4b")


def test_preflight_missing_python(tmp_path):
    runtime = _runtime(tmp_path)
    (runtime.python).unlink()
    adapted = adapt_config(_setup(tmp_path), runtime, "run-1", "klein-4b")
    result = run_preflight(adapted.config, runtime, "klein-4b", te_path=adapted.te_path, probe=_ok_probe)
    assert not result.ok
    assert any("python 不存在" in e for e in result.errors)


def test_preflight_missing_caption_uses_empty_manifest_caption(tmp_path):
    runtime = _runtime(tmp_path)
    source = _setup(tmp_path)
    data = Path(source["train_data_dir"])
    Image.new("RGB", (8, 8)).save(data / "img2.png")  # no img2.txt
    adapted = adapt_config(source, runtime, "run-1", "klein-4b")
    result = run_preflight(adapted.config, runtime, "klein-4b", te_path=adapted.te_path, probe=_ok_probe)
    assert result.ok, result.errors
    assert result.facts["dataset_image_count"] == 2
    manifest = next(iter(adapted.dataset_manifests.values()))
    assert manifest[str(data / "img1.png")]["caption"] == "a cat"
    assert manifest[str(data / "img2.png")]["caption"] == ""


def test_preflight_probe_no_cuda(tmp_path):
    runtime = _runtime(tmp_path)

    def no_cuda(runtime: RuntimeConfig) -> ProbeFacts:
        return ProbeFacts(cuda_available=False)

    adapted = adapt_config(_setup(tmp_path), runtime, "run-1", "klein-4b")
    result = run_preflight(adapted.config, runtime, "klein-4b", te_path=adapted.te_path, probe=no_cuda)
    assert not result.ok
    assert any("CUDA" in e for e in result.errors)


@pytest.mark.parametrize("asset", ["flux-2-klein-base-4b.safetensors", "ae.safetensors",
                                  "qwen3-4b/model.safetensors"])
def test_preflight_invalid_weights_rejected_before_probe(tmp_path, asset):
    runtime = _runtime(tmp_path)
    adapted = adapt_config(_setup(tmp_path), runtime, "run-1", "klein-4b")
    (tmp_path / "models" / asset).write_bytes(b"invalid weights")

    def unexpected_probe(runtime):
        pytest.fail("Invalid assets must fail before the dependency probe")

    result = run_preflight(adapted.config, runtime, "klein-4b", probe=unexpected_probe)
    assert not result.ok
    assert any("safetensors" in error for error in result.errors)


def test_preflight_missing_te_directory_is_error(tmp_path):
    runtime = _runtime(tmp_path)
    adapted = adapt_config(_setup(tmp_path, with_te=False), runtime, "run-1", "klein-4b")
    result = run_preflight(adapted.config, runtime, "klein-4b", probe=_ok_probe)
    assert not result.ok
    assert any("config.json" in error for error in result.errors)


def test_preflight_missing_startup_script_is_error(tmp_path):
    runtime = _runtime(tmp_path)
    adapted = adapt_config(_setup(tmp_path), runtime, "run-1", "klein-4b")
    (runtime.toolkit_root / "run.py").unlink()
    result = run_preflight(adapted.config, runtime, "klein-4b", probe=_ok_probe)
    assert not result.ok
    assert any("run.py" in error for error in result.errors)


# ---------------------------------------------------------------------------
# SD 1.5 (arch sd1): Comfy/LDM single file + local diffusers configs/tokenizer
# ---------------------------------------------------------------------------

SD15_CONFIG_FILES = {
    "model_index.json": {"_class_name": "StableDiffusionPipeline"},
    "unet/config.json": {"sample_size": 64},
    "vae/config.json": {"block_out_channels": [128]},
    "text_encoder/config.json": {"hidden_size": 768},
    "feature_extractor/preprocessor_config.json": {"image_processor_type": "CLIPImageProcessor"},
    "scheduler/scheduler_config.json": {"num_train_timesteps": 1000},
    "tokenizer/tokenizer_config.json": {},
    "tokenizer/vocab.json": {},
}


def _sd15_config_dir(tmp_path: Path) -> Path:
    config = tmp_path / "sd-models" / "sd15" / "config"
    for name, payload in SD15_CONFIG_FILES.items():
        path = config / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return config


def _sd15_source(tmp_path: Path, config_dir: Path | None = None) -> dict:
    data = tmp_path / "train" / "sd15"
    data.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(data / "img1.png")
    checkpoint = tmp_path / "models" / "v1-5-pruned-emaonly.safetensors"
    checkpoint.parent.mkdir(exist_ok=True)
    _weights(checkpoint)
    source = {
        "model_input_mode": "single_file",
        "dit_path": str(checkpoint),
        "train_data_dir": str(data),
        "max_train_steps": 100,
    }
    if config_dir is not None:
        source["model_config_dir"] = str(config_dir)
    return source


def test_sd15_single_file_binds_local_config_dir(tmp_path):
    runtime = _runtime(tmp_path)
    config = _sd15_config_dir(tmp_path)
    adapted = adapt_config(_sd15_source(tmp_path, config), runtime, "run-1", "sd15")
    model = adapted.config["config"]["process"][0]["model"]
    assert model["arch"] == "sd1"
    assert Path(model["name_or_path"]) == tmp_path / "models" / "v1-5-pruned-emaonly.safetensors"
    assert Path(model["model_kwargs"]["next_trainer_sd15_config"]) == config
    assert "next_trainer_sdxl_config" not in model["model_kwargs"]
    assert Path(adapted.config["meta"]["next_trainer"]["assets"]["config_dir"]) == config


def test_sd15_single_file_defaults_config_dir_under_project(tmp_path):
    runtime = _runtime(tmp_path)
    adapted = adapt_config(_sd15_source(tmp_path), runtime, "run-1", "sd15")
    model = adapted.config["config"]["process"][0]["model"]
    assert Path(model["model_kwargs"]["next_trainer_sd15_config"]) == tmp_path / "sd-models" / "sd15" / "config"


def test_preflight_sd15_ok(tmp_path):
    runtime = _runtime(tmp_path)
    adapted = adapt_config(_sd15_source(tmp_path, _sd15_config_dir(tmp_path)), runtime, "run-1", "sd15")
    result = run_preflight(adapted.config, runtime, "sd15", te_path=adapted.te_path, probe=_ok_probe)
    assert result.ok, result.errors
    assert result.facts["arch"] == "sd1"
    assert result.facts["model_assets"]["mode"] == "single_file"


def test_preflight_sd15_missing_config_dir_is_error(tmp_path):
    runtime = _runtime(tmp_path)
    adapted = adapt_config(_sd15_source(tmp_path), runtime, "run-1", "sd15")
    result = run_preflight(adapted.config, runtime, "sd15", te_path=adapted.te_path, probe=_ok_probe)
    assert not result.ok
    assert any("model_index.json" in error for error in result.errors)


# ---------------------------------------------------------------------------
# Ideogram 4 (arch ideogram4): Comfy-Org single files, loaded in place
# ---------------------------------------------------------------------------

def _keys_weights(path: Path, keys: list[str]) -> None:
    entries, offset = {}, 0
    for key in keys:
        entries[key] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        offset += 4
    header = json.dumps(entries).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * offset)


def _ideogram4_extension(runtime: RuntimeConfig) -> None:
    extension = runtime.toolkit_root / "extensions_built_in" / "diffusion_models" / "ideogram4" / "ideogram4.py"
    extension.parent.mkdir(parents=True, exist_ok=True)
    extension.write_text("", encoding="utf-8")


def _ideogram4_te(tmp_path: Path) -> Path:
    te_dir = tmp_path / "models" / "qwen3-vl-8b-instruct"
    te_dir.mkdir(parents=True, exist_ok=True)
    (te_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl", "text_config": {"hidden_size": 4096}}), encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (te_dir / name).write_text("{}", encoding="utf-8")
    _weights(te_dir / "model.safetensors")
    return te_dir


def _ideogram4_source(tmp_path: Path) -> dict:
    data = tmp_path / "train" / "ideogram4"
    data.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(data / "img1.png")
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    dit = models / "ideogram4_fp8_scaled.safetensors"
    _keys_weights(dit, ["input_proj.weight", "layers.0.attention.qkv.weight",
                        "layers.0.attention.qkv.weight_scale", "layers.0.attention.qkv.comfy_quant"])
    vae = models / "flux2-vae.safetensors"
    _keys_weights(vae, ["decoder.conv_in.weight", "decoder.mid.attn_1.k.weight"])
    return {
        "model_input_mode": "comfyui_files",
        "dit_path": str(dit),
        "vae_path": str(vae),
        "text_encoder_path": str(_ideogram4_te(tmp_path)),
        "train_data_dir": str(data),
        "max_train_steps": 100,
    }


def test_preflight_ideogram4_comfyui_files_ok(tmp_path):
    runtime = _runtime(tmp_path)
    _ideogram4_extension(runtime)
    adapted = adapt_config(_ideogram4_source(tmp_path), runtime, "run-1", "ideogram4")
    result = run_preflight(adapted.config, runtime, "ideogram4", te_path=adapted.te_path, probe=_ok_probe)
    assert result.ok, result.errors
    assert result.facts["arch"] == "ideogram4"
    comfy = result.facts["comfy_in_place"]
    assert comfy["conversion_cache"] is None
    assert comfy["transformer"].endswith("ideogram4_fp8_scaled.safetensors")
    assert any("不写入 diffusers 目录" in warning for warning in result.warnings)


def test_preflight_ideogram4_model_directory_ok(tmp_path):
    runtime = _runtime(tmp_path)
    _ideogram4_extension(runtime)
    model_dir = tmp_path / "models" / "ideogram4-diffusers"
    for folder in ("transformer", "vae"):
        (model_dir / folder).mkdir(parents=True, exist_ok=True)
        _weights(model_dir / folder / "diffusion_pytorch_model.safetensors")
    source = _ideogram4_source(tmp_path)
    source["model_input_mode"] = "model_directory"
    source["model_path"] = str(model_dir)
    source.pop("dit_path")
    adapted = adapt_config(source, runtime, "run-1", "ideogram4")
    result = run_preflight(adapted.config, runtime, "ideogram4", te_path=adapted.te_path, probe=_ok_probe)
    assert result.ok, result.errors
    assert "comfy_in_place" not in result.facts


def test_preflight_ideogram4_wrong_dit_rejected(tmp_path):
    runtime = _runtime(tmp_path)
    _ideogram4_extension(runtime)
    source = _ideogram4_source(tmp_path)
    _keys_weights(Path(source["dit_path"]), ["decoder.conv_in.weight"])
    adapted = adapt_config(source, runtime, "run-1", "ideogram4")
    result = run_preflight(adapted.config, runtime, "ideogram4", te_path=adapted.te_path, probe=_ok_probe)
    assert not result.ok
    assert any("条件 DiT" in error for error in result.errors)


def test_preflight_ideogram4_te_size_mismatch_rejected(tmp_path):
    runtime = _runtime(tmp_path)
    _ideogram4_extension(runtime)
    source = _ideogram4_source(tmp_path)
    te_config = Path(source["text_encoder_path"]) / "config.json"
    te_config.write_text(json.dumps({"model_type": "qwen3_vl", "text_config": {"hidden_size": 2560}}), encoding="utf-8")
    adapted = adapt_config(source, runtime, "run-1", "ideogram4")
    result = run_preflight(adapted.config, runtime, "ideogram4", te_path=adapted.te_path, probe=_ok_probe)
    assert not result.ok
    assert any("hidden_size=4096" in error for error in result.errors)


def test_preflight_ideogram4_missing_extension_is_error(tmp_path):
    runtime = _runtime(tmp_path)
    adapted = adapt_config(_ideogram4_source(tmp_path), runtime, "run-1", "ideogram4")
    result = run_preflight(adapted.config, runtime, "ideogram4", te_path=adapted.te_path, probe=_ok_probe)
    assert not result.ok
    assert any("不含 Ideogram 4" in error for error in result.errors)
