"""Invalid UI/API values must fail before creating a task or reading models."""
import json
from types import SimpleNamespace

import pytest

from mikazuki.engines.diffsynth import adapter
from mikazuki.engines.diffsynth.samples import sample_config


@pytest.fixture
def light_config(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "model_inputs", lambda *_: ([], tmp_path))
    monkeypatch.setattr(adapter, "dataset_inputs", lambda *_: (tmp_path, None, [{}]))
    return SimpleNamespace(project_root=tmp_path, root=tmp_path), {
        "output_dir": str(tmp_path), "output_name": "test",
    }


@pytest.mark.parametrize("patch,field", [
    ({"training_task": "image-edit"}, "training_task"),
    ({"training_task": "unknown"}, "training_task"),
    ({"control_data_dirs": ["reference"]}, "control_data_dirs"),
    ({"control_data_dirs": "reference"}, "control_data_dirs"),
    ({"num_epochs": None}, "num_epochs"),
    ({"num_epochs": True}, "num_epochs"),
    ({"lora_rank": float("inf")}, "lora_rank"),
    ({"learning_rate": []}, "learning_rate"),
    ({"learning_rate": True}, "learning_rate"),
    ({"save_steps": 0}, "save_steps"),
])
def test_invalid_training_values(light_config, patch, field):
    rt, config = light_config
    with pytest.raises(ValueError, match=field):
        adapter.adapt_config({**config, **patch}, rt)


@pytest.mark.parametrize("patch,field", [
    ({"sample_enabled": "false"}, "sample_enabled"),
    ({"sample_every_n_steps": None}, "sample_every_n_steps"),
    ({"sample_every_n_steps": float("inf")}, "sample_every_n_steps"),
    ({"preview_samples": None}, "preview_samples"),
    ({"preview_samples": "{}"}, "preview_samples"),
    ({"preview_samples": [{}]}, "preview_samples"),
    ({"preview_samples": ["broken"]}, "preview_samples"),
    ({"preview_samples": [json.dumps({"guidance_scale": None})]}, "guidance_scale"),
    ({"preview_samples": [json.dumps({"guidance_scale": True})]}, "guidance_scale"),
])
def test_invalid_preview_values(patch, field):
    with pytest.raises(ValueError, match=field):
        sample_config({"sample_enabled": True, **patch})


def test_empty_reference_list_and_legacy_t2i_still_work(light_config):
    rt, config = light_config
    result = adapter.adapt_config({**config, "training_task": "text-to-image", "control_data_dirs": []}, rt)
    assert result.engine["samples"]["enabled"] is False


@pytest.mark.parametrize("patch,field", [
    ({"training_task": "image-edit"}, "training_task"),
    ({"num_epochs": None}, "num_epochs"),
    ({"sample_enabled": True, "preview_samples": [{}]}, "preview_samples"),
])
def test_entry_points_return_field_error_without_creating_task(light_config, monkeypatch, patch, field):
    import asyncio
    from unittest.mock import Mock
    from mikazuki.engines.diffsynth import run, routes
    rt, config = light_config
    for module in (run, routes):
        monkeypatch.setattr(module, "runtime", lambda: rt)
        monkeypatch.setattr(module, "check_runtime", lambda _: None)
    monkeypatch.setattr(run, "assert_idle", lambda **_: None)
    create = Mock(side_effect=AssertionError("Must not create a task"))
    monkeypatch.setattr(run.tm, "create_task", create)
    config = {**config, **patch}
    responses = [
        run.handle_run(config, SimpleNamespace()),
        asyncio.run(routes.preflight(config)),
        asyncio.run(routes.dry_run(config)),
    ]
    for response in responses:
        assert response.status == "fail"
        assert field in response.message
    create.assert_not_called()
