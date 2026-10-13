"""#417 / #418: musubi parameter guards (fp8 base, steps vs epochs)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mikazuki.engines.musubi.adapter import adapt_config
from mikazuki.engines.musubi.settings import RuntimeConfig


def make_runtime(root: Path) -> RuntimeConfig:
    return RuntimeConfig(
        musubi_root=(root / "vendor" / "musubi-tuner").resolve(),
        python=(root / "vendor" / "musubi-tuner" / ".venv" / "Scripts" / "python.exe").resolve(),
        lora_next_root=root.resolve(),
        output_dir=(root / "output" / "musubi").resolve(),
        logging_dir=(root / "logs" / "musubi").resolve(),
        cache_dir=(root / ".cache" / "musubi").resolve(),
    )


def write_safetensors_header(path: Path, dtype: str) -> None:
    header = {"weight": {"dtype": dtype, "shape": [1], "data_offsets": [0, 2]}}
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(len(blob).to_bytes(8, "little"))
        handle.write(blob)


def krea2_config(root: Path, dit: Path | None = None) -> dict:
    (root / "train").mkdir(exist_ok=True)
    return {
        "model_train_type": "krea2-lora",
        "dit": str(dit or (root / "krea2.safetensors")),
        "vae": "./sd-models/krea2/qwen_image_vae.safetensors",
        "text_encoder": "./sd-models/krea2/qwen3_vl_4b.safetensors",
        "train_data_dir": str(root / "train"),
    }


class StepsVersusEpochsTests(unittest.TestCase):
    """#418: steps must win, and the contract must be stated in a warning."""

    def test_steps_win_when_both_are_set(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            adapted = adapt_config(
                {**krea2_config(root), "max_train_steps": 100, "max_train_epochs": 100},
                make_runtime(root),
                "run-id",
            )

        self.assertEqual(adapted.values["max_train_steps"], 100)
        self.assertNotIn("max_train_epochs", adapted.values)
        warnings = "\n".join(adapted.warnings)
        self.assertIn("max_train_steps", warnings)
        self.assertIn("max_train_epochs", warnings)
        self.assertIn("步数优先", warnings)

    def test_single_field_is_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            epochs_only = adapt_config({**krea2_config(root), "max_train_epochs": 16}, make_runtime(root), "run-id")
            steps_only = adapt_config({**krea2_config(root), "max_train_steps": 500}, make_runtime(root), "run-id")

        self.assertEqual(epochs_only.values["max_train_epochs"], 16)
        self.assertNotIn("max_train_steps", epochs_only.values)
        self.assertEqual(steps_only.values["max_train_steps"], 500)
        self.assertNotIn("max_train_epochs", steps_only.values)
        self.assertNotIn("步数优先", "\n".join(epochs_only.warnings))


class Fp8BaseTests(unittest.TestCase):
    """#417: an already-FP8 DiT must not keep fp8_base/fp8_scaled enabled."""

    def test_fp8_base_disables_the_pair_with_a_warning(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dit = root / "krea2-fp8.safetensors"
            write_safetensors_header(dit, "F8_E4M3")

            adapted = adapt_config(
                {**krea2_config(root, dit), "fp8_base": True, "fp8_scaled": True},
                make_runtime(root),
                "run-id",
            )

        self.assertNotIn("fp8_base", adapted.values)
        self.assertNotIn("fp8_scaled", adapted.values)
        warnings = "\n".join(adapted.warnings)
        self.assertIn("已是 FP8", warnings)
        self.assertIn("fp8_base", warnings)

    def test_fp8_base_with_both_switches_off_stays_quiet(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dit = root / "krea2-fp8.safetensors"
            write_safetensors_header(dit, "F8_E5M2")

            adapted = adapt_config(krea2_config(root, dit), make_runtime(root), "run-id")

        self.assertNotIn("fp8_base", adapted.values)
        self.assertNotIn("已是 FP8", "\n".join(adapted.warnings))

    def test_non_fp8_base_keeps_the_pairing_behaviour(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dit = root / "krea2-bf16.safetensors"
            write_safetensors_header(dit, "BF16")

            adapted = adapt_config({**krea2_config(root, dit), "fp8_base": True}, make_runtime(root), "run-id")

        self.assertTrue(adapted.values["fp8_base"])
        self.assertTrue(adapted.values["fp8_scaled"])
        self.assertIn("成对开启", "\n".join(adapted.warnings))

    def test_unreadable_dit_does_not_break_adaptation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dit = root / "broken.safetensors"
            dit.write_bytes(b"\x00\x01\x02")

            adapted = adapt_config({**krea2_config(root, dit), "fp8_scaled": True}, make_runtime(root), "run-id")

        # Detection fails open: the existing pair rule still applies.
        self.assertTrue(adapted.values["fp8_base"])
        self.assertTrue(adapted.values["fp8_scaled"])


if __name__ == "__main__":
    unittest.main()
