from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mikazuki.model_assets import (
    AssetDef,
    check_assets,
    download_assets,
    krea2_tokenizer_dir,
    manifest_for,
    normalize_tokenizer_config,
    patch_krea2_tokenizer_path,
    resolve_train_type,
)


class ModelAssetsManifestTests(unittest.TestCase):
    def test_krea2_manifest_has_required_assets(self):
        keys = [asset.key for asset in manifest_for("krea2-lora")]
        self.assertEqual(keys, ["dit", "vae", "text_encoder", "turbo_dit", "tokenizer"])

    def test_krea2_manifest_has_configured_sources(self):
        for asset in manifest_for("krea2-lora"):
            self.assertTrue(asset.hf_repo, asset.key)
            self.assertTrue(asset.ms_repo, asset.key)

    def test_unknown_train_type_has_empty_manifest(self):
        self.assertEqual(manifest_for("sd15-lora"), [])

    def test_klein_manifests_have_dit_and_vae(self):
        for train_type, dit_file in (
            ("klein-4b-lora", "flux-2-klein-base-4b.safetensors"),
            ("klein-9b-lora", "flux-2-klein-base-9b.safetensors"),
        ):
            assets = {asset.key: asset for asset in manifest_for(train_type)}
            self.assertEqual(set(assets), {"dit", "vae", "text_encoder"})
            self.assertEqual(assets["dit"].hf_file, dit_file)
            self.assertTrue(assets["dit"].ms_repo, "ModelScope mirror for DiT")
            self.assertFalse(assets["dit"].optional)
            # VAE: HF + ModelScope (KanKanKan/flux2-vae mirror), required
            self.assertEqual(assets["vae"].hf_repo, "ai-toolkit/flux2_vae")
            self.assertEqual(assets["vae"].ms_repo, "KanKanKan/flux2-vae")
            self.assertEqual(assets["vae"].ms_file, "flux2-vae.safetensors")
            self.assertFalse(assets["vae"].optional)
            # TE: full model dir, HF + ModelScope same repo id, needs weights
            te = assets["text_encoder"]
            self.assertEqual(te.kind, "dir")
            self.assertTrue(te.hf_repo and te.ms_repo)
            self.assertTrue(te.dir_needs_weights)
            self.assertIn("*.safetensors", te.dir_patterns)
        self.assertEqual(
            {a.key: a.hf_repo for a in manifest_for("klein-4b-lora")}["text_encoder"], "Qwen/Qwen3-4B"
        )
        self.assertEqual(
            {a.key: a.hf_repo for a in manifest_for("klein-9b-lora")}["text_encoder"], "Qwen/Qwen3-8B"
        )

    def test_klein_vae_lands_next_to_dit(self):
        # ai-toolkit auto-loads <name_or_path>/ae.safetensors: dit and vae defaults
        # must share the parent dir.
        assets = {asset.key: asset for asset in manifest_for("klein-4b-lora")}
        self.assertEqual(str(Path(assets["dit"].default_path).parent), str(Path(assets["vae"].default_path).parent))

    def test_resolve_train_type_prefers_config_value(self):
        self.assertEqual(resolve_train_type("lora-master", {"model_train_type": "krea2-lora"}), "krea2-lora")
        self.assertEqual(resolve_train_type("krea2-lora", {}), "krea2-lora")

    def test_ai_toolkit_sd15_manifest_keeps_comfy_single_file(self):
        assets = {asset.key: asset for asset in manifest_for("ai-toolkit-sd15-lora")}
        self.assertEqual(set(assets), {"dit", "model_config_dir"})
        dit = assets["dit"]
        self.assertEqual(dit.hf_file, "v1-5-pruned-emaonly.safetensors")
        self.assertEqual(dit.ms_repo, "AI-ModelScope/stable-diffusion-v1-5")
        self.assertEqual(dit.default_path, "sd-models/sd15/v1-5-pruned-emaonly.safetensors")
        config = assets["model_config_dir"]
        self.assertEqual(config.kind, "dir")
        self.assertFalse(config.dir_needs_weights)
        self.assertIn("model_index.json", config.dir_required)
        self.assertIn("feature_extractor/preprocessor_config.json", config.dir_required)
        self.assertIn("tokenizer/vocab.json", config.dir_required)
        # The form default in model_inputs/shared.ts must match the asset path.
        self.assertEqual(config.default_path, "sd-models/sd15/config")

    def test_ai_toolkit_ideogram4_manifest_reuses_comfy_org_sources(self):
        assets = {asset.key: asset for asset in manifest_for("ai-toolkit-ideogram4-lora")}
        self.assertEqual(set(assets), {"dit", "vae", "text_encoder", "text_encoder_tokenizer", "unconditional_lora"})
        self.assertEqual(assets["dit"].hf_repo, "Comfy-Org/Ideogram-4")
        self.assertEqual(assets["dit"].hf_file, "diffusion_models/ideogram4_fp8_scaled.safetensors")
        self.assertIn("非商用", assets["dit"].label)
        self.assertEqual(assets["vae"].hf_file, "vae/flux2-vae.safetensors")
        te = assets["text_encoder"]
        self.assertEqual(te.kind, "file")
        self.assertEqual(te.hf_repo, "Comfy-Org/Ideogram-4")
        self.assertEqual(te.hf_file, "text_encoders/qwen3vl_8b_fp8_scaled.safetensors")
        lora = assets["unconditional_lora"]
        self.assertTrue(lora.optional)
        self.assertEqual(lora.hf_repo, "ostris/ideogram_4_unconditional_lora")
        self.assertEqual(lora.hf_file, "ideogram_4_unconditional_lora_r16.safetensors")
        tokenizer = assets["text_encoder_tokenizer"]
        self.assertEqual(tokenizer.kind, "dir")
        self.assertEqual(tokenizer.hf_repo, "Qwen/Qwen3-VL-8B-Instruct")
        self.assertFalse(tokenizer.dir_needs_weights)
        self.assertIn("config.json", tokenizer.dir_required)

    def test_ai_toolkit_ideogram4_shares_comfy_defaults_with_musubi(self):
        musubi = {asset.key: asset for asset in manifest_for("ideogram4-lora")}
        ai_toolkit = {asset.key: asset for asset in manifest_for("ai-toolkit-ideogram4-lora")}
        self.assertEqual(ai_toolkit["dit"].default_path, musubi["dit"].default_path)
        self.assertEqual(ai_toolkit["vae"].default_path, musubi["vae"].default_path)
        self.assertEqual(ai_toolkit["dit"].hf_repo, musubi["dit"].hf_repo)
        self.assertEqual(ai_toolkit["vae"].hf_repo, musubi["vae"].hf_repo)


class ModelAssetsToolkitTrainTypeCheckTests(unittest.TestCase):
    def test_sd15_config_dir_completeness_and_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "sd-models" / "sd15" / "config"
            for name in ("model_index.json", "unet/config.json", "vae/config.json",
                         "text_encoder/config.json", "feature_extractor/preprocessor_config.json",
                         "scheduler/scheduler_config.json",
                         "tokenizer/vocab.json", "tokenizer/tokenizer_config.json"):
                path = config / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            items = {item["key"]: item for item in check_assets("ai-toolkit-sd15-lora", {}, root)}
            self.assertTrue(items["model_config_dir"]["exists"])
            self.assertEqual(items["model_config_dir"]["path"], str(config.resolve()))
            self.assertFalse(items["dit"]["exists"])
            self.assertTrue(items["dit"]["sources"]["huggingface"])
            self.assertTrue(items["dit"]["sources"]["modelscope"])

    def test_ideogram4_text_encoder_is_a_comfy_single_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            te = root / "sd-models" / "ideogram4" / "qwen3vl_8b_fp8_scaled.safetensors"
            te.parent.mkdir(parents=True)
            before = {item["key"]: item for item in check_assets("ai-toolkit-ideogram4-lora", {}, root)}
            self.assertFalse(before["text_encoder"]["exists"])
            te.write_text("x", encoding="utf-8")
            after = {item["key"]: item for item in check_assets("ai-toolkit-ideogram4-lora", {}, root)}
            self.assertTrue(after["text_encoder"]["exists"])

    def test_ideogram4_tokenizer_dir_no_longer_needs_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tokenizer = root / "sd-models" / "ideogram4" / "qwen3-vl-8b-instruct"
            tokenizer.mkdir(parents=True)
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
                (tokenizer / name).write_text("{}", encoding="utf-8")
            items = {item["key"]: item for item in check_assets("ai-toolkit-ideogram4-lora", {}, root)}
            self.assertTrue(items["text_encoder_tokenizer"]["exists"])


class ModelAssetsCheckTests(unittest.TestCase):
    def test_check_reports_missing_and_existing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            existing = root / "models" / "dit.safetensors"
            existing.parent.mkdir(parents=True)
            existing.write_text("x", encoding="utf-8")
            items = check_assets("krea2-lora", {"dit": "models/dit.safetensors"}, root)
            by_key = {item["key"]: item for item in items}
            self.assertTrue(by_key["dit"]["exists"])
            self.assertEqual(by_key["dit"]["path"], str(existing.resolve()))
            self.assertFalse(by_key["vae"]["exists"])
            self.assertFalse(by_key["text_encoder"]["exists"])
            self.assertFalse(by_key["tokenizer"]["exists"])
            self.assertTrue(by_key["turbo_dit"]["optional"])

    def test_check_marks_sources_configured_for_krea2(self):
        with tempfile.TemporaryDirectory() as td:
            items = check_assets("krea2-lora", {}, Path(td))
            for item in items:
                self.assertTrue(item["sources"]["huggingface"], item["key"])
                self.assertTrue(item["sources"]["modelscope"], item["key"])

    def test_tokenizer_dir_complete_when_required_files_present(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tokenizer_dir = root / "sd-models/krea2/qwen3-vl-tokenizer"
            tokenizer_dir.mkdir(parents=True)
            (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            items = check_assets("krea2-lora", {}, root)
            tokenizer = next(item for item in items if item["key"] == "tokenizer")
            self.assertTrue(tokenizer["exists"])
            self.assertEqual(tokenizer["path"], str(tokenizer_dir.resolve()))


class ModelAssetsDownloadTests(unittest.TestCase):
    def test_download_via_huggingface(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            def fake_hf_download(repo_id, filename, local_dir):
                target = Path(local_dir) / filename
                target.write_text("weights", encoding="utf-8")
                return str(target)

            with mock.patch("mikazuki.model_assets.manifest_for") as manifest, \
                 mock.patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_download):
                manifest.return_value = [
                    AssetDef("dit", "DiT", "sd-models/krea2/krea2.safetensors", hf_repo="org/repo", hf_file="krea2.safetensors")
                ]
                logs = []
                download_assets("krea2-lora", [{"key": "dit"}], "huggingface", root, logs.append)

            self.assertEqual((root / "sd-models/krea2/krea2.safetensors").read_text(encoding="utf-8"), "weights")
            self.assertTrue(any("huggingface org/repo" in line for line in logs))

    def test_download_rejects_unconfigured_source(self):
        with tempfile.TemporaryDirectory() as td, mock.patch("mikazuki.model_assets.manifest_for") as manifest:
            manifest.return_value = [AssetDef("dit", "DiT", "x.safetensors", hf_repo="org/repo", hf_file="f.safetensors")]
            with self.assertRaises(ValueError):
                download_assets("krea2-lora", [{"key": "dit"}], "modelscope", Path(td), lambda line: None)

    def test_download_tokenizer_dir_via_huggingface(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            def fake_snapshot_download(repo_id, allow_patterns, local_dir):
                for name in allow_patterns:
                    (Path(local_dir) / name).write_text("{}", encoding="utf-8")
                return str(local_dir)

            with mock.patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download), \
                 mock.patch("mikazuki.model_assets.patch_krea2_tokenizer_everywhere") as patch:
                logs = []
                download_assets("krea2-lora", [{"key": "tokenizer"}], "huggingface", root, logs.append)

            tokenizer_dir = krea2_tokenizer_dir(root)
            self.assertTrue((tokenizer_dir / "tokenizer.json").is_file())
            self.assertTrue((tokenizer_dir / "merges.txt").is_file())
            patch.assert_called_once()


class Krea2TokenizerPatchTests(unittest.TestCase):
    ENCODER = (
        '"""encoder"""\n'
        "def load_qwen3_vl_conditioner(model_path, tokenizer_repo, max_length):\n"
        "    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, max_length=max_length)\n"
        "    processor = Qwen2TokenizerFast.from_pretrained(tokenizer_repo, max_length=max_length)\n"
    )

    def _layout(self, root: Path, with_tokenizer: bool = True) -> tuple[Path, Path]:
        tokenizer_dir = root / "tokenizer"
        if with_tokenizer:
            tokenizer_dir.mkdir(parents=True)
            (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        source_root = root / "musubi"
        encoder_dir = source_root / "src" / "musubi_tuner" / "krea2"
        encoder_dir.mkdir(parents=True, exist_ok=True)
        (encoder_dir / "krea2_encoder.py").write_text(self.ENCODER, encoding="utf-8")
        return source_root, tokenizer_dir

    def test_patch_injects_local_tokenizer_dir_once(self):
        with tempfile.TemporaryDirectory() as td:
            source_root, tokenizer_dir = self._layout(Path(td))
            logs = []
            self.assertTrue(patch_krea2_tokenizer_path(source_root, tokenizer_dir, logs.append))
            encoder = source_root / "src" / "musubi_tuner" / "krea2" / "krea2_encoder.py"
            text = encoder.read_text(encoding="utf-8")
            self.assertIn(f'from_pretrained(r"{tokenizer_dir.as_posix()}"', text)
            self.assertNotIn("from_pretrained(tokenizer_repo", text)
            compile(text, str(encoder), "exec")
            self.assertTrue(patch_krea2_tokenizer_path(source_root, tokenizer_dir, logs.append))
            self.assertEqual(encoder.read_text(encoding="utf-8").count("mikazuki: patched tokenizer path"), 2)

    def test_patch_noop_without_tokenizer_files(self):
        with tempfile.TemporaryDirectory() as td:
            source_root, tokenizer_dir = self._layout(Path(td), with_tokenizer=False)
            self.assertFalse(patch_krea2_tokenizer_path(source_root, tokenizer_dir, lambda line: None))
            encoder = source_root / "src" / "musubi_tuner" / "krea2" / "krea2_encoder.py"
            self.assertEqual(encoder.read_text(encoding="utf-8"), self.ENCODER)


class TokenizerConfigNormalizeTests(unittest.TestCase):
    """#416: legacy list form of extra_special_tokens breaks tokenizer loading."""

    def _write_config(self, root: Path, payload) -> Path:
        config_path = root / "tokenizer_config.json"
        if isinstance(payload, str):
            config_path.write_text(payload, encoding="utf-8")
        else:
            config_path.write_text(json.dumps(payload), encoding="utf-8")
        return config_path

    def test_list_form_is_converted_to_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = self._write_config(
                Path(td),
                {"tokenizer_class": "Qwen2TokenizerFast", "extra_special_tokens": ["<|a|>", "<|b|>"]},
            )
            logs = []
            self.assertTrue(normalize_tokenizer_config(Path(td), logs.append))
            payload = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["extra_special_tokens"], {"additional_special_tokens": ["<|a|>", "<|b|>"]})
        self.assertEqual(payload["tokenizer_class"], "Qwen2TokenizerFast")
        self.assertTrue(any("extra_special_tokens" in line for line in logs))

    def test_mapping_form_is_left_untouched(self):
        original = {"extra_special_tokens": {"additional_special_tokens": ["<|a|>"]}}
        with tempfile.TemporaryDirectory() as td:
            config_path = self._write_config(Path(td), original)
            before = config_path.read_text(encoding="utf-8")

            self.assertFalse(normalize_tokenizer_config(Path(td), lambda line: None))
            after = config_path.read_text(encoding="utf-8")

        self.assertEqual(after, before)

    def test_missing_or_malformed_config_is_safe(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertFalse(normalize_tokenizer_config(root, lambda line: None))
            config_path = self._write_config(root, "{not json")
            self.assertFalse(normalize_tokenizer_config(root, lambda line: None))
            self.assertEqual(config_path.read_text(encoding="utf-8"), "{not json")

    def test_krea2_patch_normalizes_before_pointing_at_the_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_root = root / "musubi"
            encoder_dir = source_root / "src" / "musubi_tuner" / "krea2"
            encoder_dir.mkdir(parents=True)
            (encoder_dir / "krea2_encoder.py").write_text(Krea2TokenizerPatchTests.ENCODER, encoding="utf-8")
            tokenizer_dir = root / "tokenizer"
            tokenizer_dir.mkdir()
            (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (tokenizer_dir / "tokenizer_config.json").write_text(
                json.dumps({"extra_special_tokens": ["<|a|>"]}), encoding="utf-8"
            )

            self.assertTrue(patch_krea2_tokenizer_path(source_root, tokenizer_dir, lambda line: None))
            payload = json.loads((tokenizer_dir / "tokenizer_config.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["extra_special_tokens"], {"additional_special_tokens": ["<|a|>"]})


if __name__ == "__main__":
    unittest.main()
