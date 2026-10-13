"""Pack-side Comfy loader overrides: Ideogram 4 normalization + driver wiring.

Covers the pure normalization helpers, the module hooks the driver installs for
Ideogram 4 (no diffusers directory, nothing written to disk) and the SD 1.5
single-file diffusers config binding.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from mikazuki.engines.ai_toolkit import driver
from mikazuki.engines.ai_toolkit.ideogram4_comfy import (
    comfy_component_state_dict,
    is_comfy_marker,
    is_original_vae_state_dict,
    normalize_dit_state_dict,
    normalize_scale_tensor,
    scale_target_shape,
)


def test_is_comfy_marker():
    assert is_comfy_marker("layers.0.attention.qkv.comfy_quant")
    assert not is_comfy_marker("layers.0.attention.qkv.weight")
    assert not is_comfy_marker("layers.0.attention.qkv.weight_scale")


def test_scale_target_shape_widens_scalars_and_squeezes_singletons():
    assert scale_target_shape((), (13824, 4608)) == (13824,)
    assert scale_target_shape((4608,), (13824, 4608)) == (4608,)
    assert scale_target_shape((4608, 1), (13824, 4608)) == (4608,)
    assert scale_target_shape((1, 4608), (13824, 4608)) == (4608,)
    with pytest.raises(ValueError):
        scale_target_shape((2, 4608), (13824, 4608))
    with pytest.raises(ValueError):
        scale_target_shape((), None)


def test_scale_tensor_expands_without_copying_values():
    torch = pytest.importorskip("torch")
    scale = torch.tensor(0.25)
    out = normalize_scale_tensor(scale, (3, 2))
    assert tuple(out.shape) == (3,)
    assert out.tolist() == pytest.approx([0.25, 0.25, 0.25])
    kept = normalize_scale_tensor(torch.tensor([1.0, 2.0, 3.0]), (3, 2))
    assert kept.tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_normalize_dit_state_dict_drops_markers_and_fixes_scales():
    torch = pytest.importorskip("torch")
    weight = torch.zeros(4, 2, dtype=torch.float8_e4m3fn)
    state_dict = {
        "layers.0.attention.qkv.weight": weight,
        "layers.0.attention.qkv.weight_scale": torch.tensor(0.5),
        "layers.0.attention.qkv.comfy_quant": torch.zeros(27, dtype=torch.uint8),
        "layers.0.attention_norm1.weight": torch.ones(2),
        "final_layer.linear.bias": torch.zeros(4),
    }
    out = normalize_dit_state_dict(state_dict)
    assert set(out) == {
        "layers.0.attention.qkv.weight",
        "layers.0.attention.qkv.weight_scale",
        "layers.0.attention_norm1.weight",
        "final_layer.linear.bias",
    }
    assert tuple(out["layers.0.attention.qkv.weight_scale"].shape) == (4,)
    assert out["layers.0.attention.qkv.weight"] is weight
    assert out["layers.0.attention_norm1.weight"] is state_dict["layers.0.attention_norm1.weight"]
    # The extension's dequantizer folds weight * scale.unsqueeze(1).
    folded = out["layers.0.attention.qkv.weight"].to(torch.float32) * out["layers.0.attention.qkv.weight_scale"].unsqueeze(1)
    assert tuple(folded.shape) == (4, 2)


def test_normalize_dit_state_dict_scalar_scale_keeps_rank_for_dequant():
    torch = pytest.importorskip("torch")
    state_dict = {
        "input_proj.weight": torch.zeros(2, 3, dtype=torch.float8_e4m3fn),
        "input_proj.weight_scale": torch.tensor(2.0),
    }
    out = normalize_dit_state_dict(state_dict)
    # unsqueeze(1) on the normalized [out] scale must not raise (0-dim would).
    folded = out["input_proj.weight"].to(torch.float32) * out["input_proj.weight_scale"].unsqueeze(1)
    assert tuple(folded.shape) == (2, 3)
    assert folded.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_original_vae_state_dict_detection():
    assert is_original_vae_state_dict({"decoder.mid.attn_1.k.weight": 1})
    assert not is_original_vae_state_dict({"decoder.mid_block.attentions.0.to_k.weight": 1})


def test_comfy_component_state_dict_reads_comfy_layout(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    source = tmp_path / "ideogram4_fp8_scaled.safetensors"
    save_file({
        "input_proj.weight": torch.zeros(4, 2, dtype=torch.float8_e4m3fn),
        "input_proj.weight_scale": torch.tensor(0.5),
        "input_proj.comfy_quant": torch.zeros(4, dtype=torch.uint8),
    }, str(source), metadata={"model_type": "ideogram4_cond"})
    out = comfy_component_state_dict(source, "transformer")
    assert set(out) == {"input_proj.weight", "input_proj.weight_scale"}
    assert out["input_proj.weight"].dtype == torch.float8_e4m3fn
    assert out["input_proj.weight_scale"].shape == (4,)


# ---------------------------------------------------------------------------
# driver wiring
# ---------------------------------------------------------------------------

def _fake_extension(monkeypatch, calls: dict):
    package = types.ModuleType("extensions_built_in.diffusion_models.ideogram4")
    module = types.ModuleType("extensions_built_in.diffusion_models.ideogram4.ideogram4")

    def original_loader(base, subfolder, basename):
        calls["loader"] = (base, subfolder, basename)
        return {"source": "diffusers-dir"}

    def original_convert(state_dict):
        return {**state_dict, "converted": True}

    def original_text_encoder(self, base):
        calls["text_encoder"] = base
        return "original-te"

    module._load_component_state_dict = original_loader
    module.convert_diffusers_state_dict = original_convert
    module._load_text_encoder = original_text_encoder
    package.ideogram4 = module
    monkeypatch.setitem(sys.modules, "extensions_built_in", types.ModuleType("extensions_built_in"))
    monkeypatch.setitem(sys.modules, "extensions_built_in.diffusion_models", types.ModuleType("extensions_built_in.diffusion_models"))
    monkeypatch.setitem(sys.modules, "extensions_built_in.diffusion_models.ideogram4", package)
    monkeypatch.setitem(sys.modules, "extensions_built_in.diffusion_models.ideogram4.ideogram4", module)
    return module, original_loader, original_convert


def test_driver_ideogram4_comfy_hooks(monkeypatch):
    calls: dict = {}
    module, original_loader, original_convert = _fake_extension(monkeypatch, calls)
    driver.apply_ideogram4_comfy_inputs({})

    assert module._load_component_state_dict is not original_loader
    # Unknown subfolders keep the diffusers-directory path untouched.
    assert module._load_component_state_dict("/some/dir", "transformer", "diffusion_pytorch_model") == {"source": "diffusers-dir"}
    assert calls["loader"] == ("/some/dir", "transformer", "diffusion_pytorch_model")
    # Original (LDM) VAE layouts skip the diffusers converter; diffusers layouts use it.
    assert module.convert_diffusers_state_dict({"decoder.mid.attn_1.k.weight": 1}) == {"decoder.mid.attn_1.k.weight": 1}
    assert module.convert_diffusers_state_dict({"decoder.conv_in.weight": 1})["converted"] is True


def test_driver_ideogram4_comfy_text_encoder_hook(monkeypatch):
    """A Comfy FP8 single-file TE must split weights from tokenizer/config (#446 follow-up)."""
    import sys
    import types

    calls: dict = {}
    module, _, _ = _fake_extension(monkeypatch, calls)
    original_te = module._load_text_encoder
    driver.apply_ideogram4_comfy_inputs({})
    wrapped = module._load_text_encoder
    assert wrapped is not original_te

    # Fake upstream toolkit modules the wrapper imports at call time.
    toolkit = types.ModuleType("toolkit")
    basic = types.ModuleType("toolkit.basic")
    basic.flush = lambda: None
    encoders = types.ModuleType("toolkit.models.v2.text_encoders")
    qwen3_vl = types.ModuleType("toolkit.models.v2.text_encoders.qwen3_vl")
    loaded: dict = {}

    class _Encoder:
        @staticmethod
        def load_model(path, **kwargs):
            loaded["path"] = path
            loaded["kwargs"] = kwargs
            return types.SimpleNamespace(eval=lambda: None, requires_grad_=lambda _flag: None)

    qwen3_vl.Qwen3VLModelEncoder = _Encoder
    monkeypatch.setitem(sys.modules, "toolkit", toolkit)
    monkeypatch.setitem(sys.modules, "toolkit.basic", basic)
    monkeypatch.setitem(sys.modules, "toolkit.models", types.ModuleType("toolkit.models"))
    monkeypatch.setitem(sys.modules, "toolkit.models.v2", types.ModuleType("toolkit.models.v2"))
    monkeypatch.setitem(sys.modules, "toolkit.models.v2.text_encoders", encoders)
    monkeypatch.setitem(sys.modules, "toolkit.models.v2.text_encoders.qwen3_vl", qwen3_vl)

    # The tokenizer would talk to the Hub for a non-existent local path; stub it.
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda *args, **kwargs: "tokenizer"),
        raising=True,
    )

    class _Self:
        torch_dtype = "bf16"
        model_config = types.SimpleNamespace(model_kwargs={
            "text_encoder_path": "/models/qwen3vl_8b_fp8_scaled.safetensors",
            "text_encoder_tokenizer_path": "/models/qwen3-vl-8b-instruct",
        })

        def print_and_status_update(self, _message):
            pass

    wrapped(_Self(), "/base")
    assert "text_encoder" not in calls  # the original loader must not run
    assert loaded["path"] == "/models/qwen3vl_8b_fp8_scaled.safetensors"
    assert loaded["kwargs"]["config_path"] == "/models/qwen3-vl-8b-instruct"
    assert loaded["kwargs"]["use_comfy_weights"] is False

    # A directory TE keeps upstream's original behaviour.
    class _DirSelf(_Self):
        model_config = types.SimpleNamespace(model_kwargs={"text_encoder_path": "/models/qwen3-vl-8b-instruct"})

    assert wrapped(_DirSelf(), "/base") == "original-te"
    assert calls["text_encoder"] == "/base"


def test_driver_ideogram4_ignores_other_architectures():
    # apply_local_inputs must not import/patch anything for the legacy paths.
    driver.apply_local_inputs({"arch": "klein-4b", "model_kwargs": {}})


def test_driver_sd15_binds_local_diffusers_config():
    diffusers = pytest.importorskip("diffusers")
    pipeline = diffusers.StableDiffusionPipeline
    original = pipeline.from_single_file.__func__
    try:
        driver.apply_local_inputs({
            "arch": "sd1",
            "model_kwargs": {"next_trainer_sd15_config": "/cfg/sd15"},
        })
        assert pipeline.from_single_file.__func__ is not original
    finally:
        pipeline.from_single_file = classmethod(original)


def test_driver_sd15_patch_forces_local_config_and_drops_safety_fetch():
    diffusers = pytest.importorskip("diffusers")
    pipeline = diffusers.StableDiffusionPipeline
    original = pipeline.from_single_file.__func__
    calls: dict = {}

    def recorder(cls, path, **kwargs):
        calls.update(path=path, **kwargs)
        return "pipe"

    pipeline.from_single_file = classmethod(recorder)
    try:
        driver.apply_local_inputs({
            "arch": "sd1",
            "model_kwargs": {"next_trainer_sd15_config": "/cfg/sd15"},
        })
        # Mirrors the upstream ai-toolkit SD 1.5 call shape.
        result = pipeline.from_single_file(
            "ckpt.safetensors", dtype="bf16", load_safety_checker=False,
            requires_safety_checker=False, torch_dtype=None, safety_checker=None,
            trust_remote_code=True,
        )
    finally:
        pipeline.from_single_file = classmethod(original)
    assert result == "pipe"
    assert calls["path"] == "ckpt.safetensors"
    assert calls["config"] == "/cfg/sd15"
    assert calls["local_files_only"] is True
    assert "load_safety_checker" not in calls
    assert calls["requires_safety_checker"] is False


def test_driver_sd15_without_config_stays_upstream():
    diffusers = pytest.importorskip("diffusers")
    pipeline = diffusers.StableDiffusionPipeline
    original = pipeline.from_single_file.__func__
    # No pack kwarg (model_directory mode or a hand-written job): no patch.
    driver.apply_local_inputs({"arch": "sd1", "model_kwargs": {}})
    assert pipeline.from_single_file.__func__ is original
