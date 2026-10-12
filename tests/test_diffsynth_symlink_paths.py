"""#390: AutoDL-style mounts name a *.safetensors link over an extension-less file."""

from __future__ import annotations

import pytest

from mikazuki.engines.diffsynth.inputs import InputError, absolute, weight_files


def _symlink_or_skip(link, target):
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")


def test_symlinked_safetensors_name_survives_absolute(tmp_path):
    target = tmp_path / "autodl-hash-file"  # real file without an extension
    target.write_bytes(b"\x00" * 32)
    link = tmp_path / "qwen_image_2.1_bf16.safetensors"
    _symlink_or_skip(link, target)

    path = absolute(str(link), tmp_path)

    assert path.name == "qwen_image_2.1_bf16.safetensors"
    assert weight_files(path, "dit_path") == [path]


def test_relative_path_with_parent_segments_is_still_normalised(tmp_path):
    model = tmp_path / "sd-models" / "qwen" / "dit.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"\x00" * 8)

    path = absolute("sd-models/qwen/../qwen/dit.safetensors", tmp_path)

    assert path == model
    assert path.is_absolute()


def test_extension_less_target_without_a_link_still_fails(tmp_path):
    """The suffix check must stay strict for genuinely wrong paths."""
    plain = tmp_path / "autodl-hash-file"
    plain.write_bytes(b"\x00" * 8)

    with pytest.raises(InputError):
        weight_files(absolute(str(plain), tmp_path), "dit_path")
