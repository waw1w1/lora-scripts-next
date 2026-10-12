"""#338: warn small datasets whose batch size will make bucket sampling drop images."""

from __future__ import annotations

from mikazuki.engines.anima_fast.preflight import (
    BUCKET_DROP_HINT_MAX_IMAGES,
    bucket_drop_warning,
)


def test_small_dataset_with_batch_size_above_one_is_flagged():
    warning = bucket_drop_warning(100, 4, 16)

    assert warning is not None
    assert "100 张图" in warning
    assert "train_batch_size=1" in warning
    assert "桶" in warning


def test_batch_size_one_is_never_flagged():
    assert bucket_drop_warning(100, 1, 16) is None


def test_large_datasets_are_not_flagged():
    assert bucket_drop_warning(BUCKET_DROP_HINT_MAX_IMAGES, 4, 16) is None
    assert bucket_drop_warning(BUCKET_DROP_HINT_MAX_IMAGES + 2000, 4, 16) is None


def test_degenerate_inputs_are_not_flagged():
    assert bucket_drop_warning(0, 4, 0) is None
    assert bucket_drop_warning(10, 4, 10) is None  # validation split ate the dataset
