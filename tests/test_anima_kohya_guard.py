"""#408: Kohya must reject Anima 2.9B checkpoints before the trainer starts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mikazuki.utils.train_utils import (
    KOHYA_ANIMA_MAX_BLOCKS,
    anima_checkpoint_block_count,
    validate_model,
)


def write_safetensors_header(path: Path, tensor_names: list[str]) -> None:
    """A safetensors file with a real header and no tensor data is enough for the
    header-only guard, which never reads weights."""
    payload = {
        name: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
        for name in tensor_names
    }
    blob = json.dumps(payload).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(len(blob).to_bytes(8, "little"))
        handle.write(blob)


def anima_like_names(blocks: int) -> list[str]:
    names = ["input_embed.weight", "final_layer.linear.weight"]
    for index in range(blocks):
        names.append(f"blocks.{index}.adaln_modulation_caption.1.weight")
        names.append(f"blocks.{index}.attention.qkv.weight")
    return names


class AnimaBlockCountTests(unittest.TestCase):
    def test_counts_blocks_from_the_header(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "anima2b.safetensors"
            write_safetensors_header(model, anima_like_names(28))

            self.assertEqual(anima_checkpoint_block_count(model), 28)

    def test_unreadable_or_foreign_files_return_none(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertIsNone(anima_checkpoint_block_count(root / "missing.safetensors"))
            text = root / "not_a_checkpoint.safetensors"
            text.write_text("SDXL base model", encoding="utf-8")
            self.assertIsNone(anima_checkpoint_block_count(text))


class KohyaAnimaGuardTests(unittest.TestCase):
    def test_kohya_rejects_2_9b_with_actionable_message(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "whatever-name.safetensors"
            write_safetensors_header(model, anima_like_names(KOHYA_ANIMA_MAX_BLOCKS + 1))

            ok, message = validate_model(str(model), "anima-lora")

        self.assertFalse(ok)
        self.assertIn("2.9B", message)
        self.assertIn("Anima Fast", message)

    def test_kohya_accepts_2b(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "anima2b.safetensors"
            write_safetensors_header(model, anima_like_names(KOHYA_ANIMA_MAX_BLOCKS))

            self.assertEqual(validate_model(str(model), "anima-lora"), (True, "ok"))

    def test_anima_fast_is_not_blocked_by_this_check(self):
        """The guard lives in the Kohya branch only: the Fast engine runs through its
        own adapter and never calls validate_model, so a 2.9B checkpoint must not be
        rejected with the Kohya message here."""
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "anima29b.safetensors"
            write_safetensors_header(model, anima_like_names(KOHYA_ANIMA_MAX_BLOCKS + 1))

            _ok, message = validate_model(str(model), "anima-lora-fast")

        self.assertNotIn("Kohya 引擎不支持", message)

    def test_broken_or_unrecognised_weights_are_not_reported_as_2_9b(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "broken.safetensors"
            model.write_bytes(b"\x00\x01\x02\x03")

            self.assertEqual(validate_model(str(model), "anima-lora"), (True, "ok"))


if __name__ == "__main__":
    unittest.main()
