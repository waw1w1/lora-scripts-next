"""Edit data contracts and conditioned caches; no real model weights required."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from test_diffsynth_engine import configured  # noqa: F401
from mikazuki.engines.diffsynth.adapter import adapt_config, dump_config
from mikazuki.engines.diffsynth.inputs import dataset_inputs
from mikazuki.engines.diffsynth.samples import sample_config


def edit_config(rt, config):
    controls = rt.project_root / '参考图' / '3_character'
    controls.mkdir(parents=True)
    Image.new('RGBA', (64, 96), 'blue').save(controls / 'sample.webp')
    return {**config, 'training_task': 'image-edit', 'control_data_dirs': [str(controls.parent)]}


def test_edit_folder_arguments_and_config_roundtrip(configured):
    from mikazuki.utils.config_import import validate_config_import
    from mikazuki.utils.config_export import normalize_config_for_export
    rt, config = configured
    config = edit_config(rt, config)
    exported, _ = normalize_config_for_export(config, page_train_type='qwen-image-21-lora')
    restored = validate_config_import('qwen-image-21-lora', exported)['config']
    assert restored['training_task'] == 'image-edit'
    assert restored['control_data_dirs'] == config['control_data_dirs']
    adapted = adapt_config(restored, rt)
    assert adapted.arguments['data_file_keys'] == 'image,edit_image'
    assert adapted.arguments['extra_inputs'] == 'edit_image'
    assert len(adapted.dataset) == 3
    assert len(adapted.dataset[0]['edit_image']) == 1
    assert adapted.engine['batches_per_epoch'] == 3
    dump_config(adapted, rt.project_root / 'autosave', 'edit')
    assert json.loads(Path(adapted.arguments['dataset_metadata_path']).read_text()) == adapted.dataset
    with pytest.raises(ValueError, match='train_batch_size=1'):
        adapt_config({**config, 'train_batch_size': 2, 'cache_embeddings': True}, rt)
    with pytest.raises(ValueError, match='找到 0'):
        (Path(config['control_data_dirs'][0]) / '3_character/sample.webp').unlink()
        dataset_inputs(config, rt.project_root)


@pytest.mark.parametrize('suffix', ['json', 'jsonl', 'csv'])
def test_metadata_normalizes_multi_reference_paths(tmp_path, suffix):
    for name in ['target.png', 'a.png', 'b.png']:
        Image.new('RGB', (64, 64)).save(tmp_path / name)
    row = {'image': 'target.png', 'prompt': '', 'edit_image': ['a.png', 'b.png']}
    metadata = tmp_path / ('dataset.' + suffix)
    if suffix == 'csv':
        with metadata.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=row)
            writer.writeheader()
            writer.writerow({**row, 'edit_image': json.dumps(row['edit_image'])})
    else:
        metadata.write_text(json.dumps([row] if suffix == 'json' else row))
    before = metadata.read_bytes()
    config = dict(training_task='image-edit', dataset_format='metadata', dataset_base_path=str(tmp_path), dataset_metadata_path=str(metadata))
    base, generated, rows = dataset_inputs(config, tmp_path)
    assert generated is None
    assert rows[0]['edit_image'] == [str(tmp_path / 'a.png'), str(tmp_path / 'b.png')]
    assert metadata.read_bytes() == before
    (tmp_path / 'a.png').unlink()
    with pytest.raises(ValueError, match='参考图不存在'):
        dataset_inputs(config, tmp_path)


def test_edit_samples_require_real_references(tmp_path):
    image = tmp_path / 'ref.png'
    Image.new('RGB', (64, 64)).save(image)
    config = dict(training_task='image-edit', sample_enabled=True, preview_samples=[json.dumps({'prompt': '改成蓝色', 'controlImages': ['ref.png']})])
    sample = sample_config(config, tmp_path)['samples'][0]
    assert sample['controlImages'] == [str(image)]
    with pytest.raises(ValueError, match='文生图不支持'):
        sample_config({**config, 'training_task': 'text-to-image'}, tmp_path)
    with pytest.raises(ValueError, match='至少需要'):
        sample_config({**config, 'preview_samples': ['{"prompt":"edit"}']}, tmp_path)


def test_edit_cache_conditions_te_and_vae_and_invalidates_reference_changes(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    upstream = pytest.importorskip('diffsynth.pipelines.qwen_image_21')
    from mikazuki.engines.diffsynth.encoding_cache import prepare_cache, read_tensor, preview_key
    from mikazuki.engines.diffsynth.edit_images import load_references
    calls = {'text': [], 'vae': []}
    class FakeModule:
        def to(self, *_): return self
        def eval(self): return self
        def encode(self, image):
            calls['vae'].append(image.clone())
            return image
    def make_pipe(**kwargs):
        norm = SimpleNamespace(_forward_hooks={})
        te = FakeModule()
        te.model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(norm=norm)))
        pipe = SimpleNamespace(text_encoder=te, vae=FakeModule(), processor=SimpleNamespace(image_processor=SimpleNamespace(size={'shortest_edge': 4096})))
        pipe.preprocess_image = lambda image: torch.full((1, 4, image.height // 16, image.width // 16), float(image.getpixel((0, 0))[0]))
        return pipe
    class Prompt:
        def process(self, pipe, caption, images):
            assert images and images[0].mode == 'RGBA'
            calls['text'].append((caption, [im.size for im in images], images[0].getpixel((0, 0))))
            return {'prompt_embeds': torch.full((1, 3, 4), float(images[0].getpixel((0, 0))[0])),
                    'prompt_embeds_mask': None, 'edit_image_pad_mask': torch.tensor([[True, False, False]])}
    monkeypatch.setattr(upstream.QwenImage21Pipeline, 'from_pretrained', make_pipe)
    monkeypatch.setattr(upstream, 'QwenImage21Unit_PromptEmbedder', Prompt)
    for name, color in [('target.png', 'green'), ('a.png', 'red'), ('b.png', 'blue')]:
        Image.new('RGBA', (64, 64), color).save(tmp_path / name)
    processor = tmp_path / 'processor'
    processor.mkdir()
    (processor / 'config.json').write_text('{}')
    weights = tmp_path / 'weights'
    weights.write_text('test')
    class Dataset:
        repeat = 1
        data = [{'image': 'target.png', 'prompt': 'same', 'edit_image': [str(tmp_path / ref)]} for ref in ('a.png', 'b.png')]
        def __getitem__(self, index):
            row = self.data[index]
            return {**row, 'image': load_references([tmp_path / row['image']])[0], 'edit_image': load_references(row['edit_image'])}
    args = SimpleNamespace(processor_path=str(processor), dataset_base_path=str(tmp_path), max_pixels=4096, height=None, width=None,
                           output_path=str(tmp_path), use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False)
    samples = [{'prompt': 'same', 'width': 64, 'height': 64, 'controlImages': [str(tmp_path / ref)]} for ref in ('a.png', 'b.png')]
    config = {'training_task': 'image-edit', 'cache_dir': str(tmp_path / 'cache/models'), 'samples': {'enabled': True, 'samples': samples}}
    paths = [[str(weights)]] * 3
    encoded, previews = prepare_cache(Dataset(), paths, args, config, 'cpu')
    assert len(encoded) == 2
    assert len(previews) == 4  # positive AND negative are conditioned on each sample.
    assert previews[preview_key(samples[0], 'same')] != previews[preview_key(samples[1], 'same')]
    first, second = encoded[0], encoded[1]
    assert first[0]['edit_latents'][0].shape == (1, 4, 4, 4)
    assert not torch.equal(first[0]['edit_latents'][0], second[0]['edit_latents'][0])
    assert not torch.equal(first[1]['prompt_embeds'], second[1]['prompt_embeds'])
    before = (len(calls['text']), len(calls['vae']))
    prepare_cache(Dataset(), paths, args, config, 'cpu')
    assert before == (len(calls['text']), len(calls['vae']))
    Image.new('RGBA', (96, 64), 'white').save(tmp_path / 'a.png')
    changed, _ = prepare_cache(Dataset(), paths, args, config, 'cpu')
    assert changed.items[0] != encoded.items[0]
    assert changed.items[1] == encoded.items[1]
    assert len(calls['text']) > before[0]
    # A readable but incomplete cached payload must be regenerated.
    torch.save({'input_latents': torch.ones(1, 4, 4, 4)}, changed.items[0][1])
    prepare_cache(Dataset(), paths, args, config, 'cpu')
    assert 'edit_latents' in read_tensor(changed.items[0][1])


def test_actual_upstream_edit_input_and_model_contract(configured):
    import importlib.util
    torch = pytest.importorskip('torch')
    pipeline = pytest.importorskip('diffsynth.pipelines.qwen_image_21')
    from diffsynth.core import UnifiedDataset
    from mikazuki.engines.diffsynth.buckets import BucketImageLoader
    from mikazuki.engines.diffsynth.settings import TRAIN_SCRIPT
    source = Path(pipeline.__file__).parents[2] / TRAIN_SCRIPT
    spec = importlib.util.spec_from_file_location('edit_upstream_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rt, config = configured
    config = edit_config(rt, config)
    adapted = adapt_config(config, rt)
    dump_config(adapted, rt.project_root / 'autosave', 'edit-contract')
    argv = []
    for key, value in adapted.arguments.items():
        if isinstance(value, bool):
            if value: argv.append('--' + key)
        else:
            argv.extend(['--' + key, str(value)])
    args = module.qwen_image_21_parser().parse_args(argv)
    reference_operator = UnifiedDataset.default_image_operator(base_path=args.dataset_base_path, max_pixels=args.max_pixels,
        height_division_factor=32, width_division_factor=32, convert_RGB=False, convert_RGBA=True)
    dataset = UnifiedDataset(base_path=args.dataset_base_path, metadata_path=args.dataset_metadata_path,
        data_file_keys=args.data_file_keys.split(','), main_data_operator=BucketImageLoader(args.dataset_base_path, adapted.engine['bucket_settings']),
        special_operator_map={'edit_image': reference_operator})
    data = dataset[0]
    model = module.QwenImage21TrainingModule.__new__(module.QwenImage21TrainingModule)
    torch.nn.Module.__init__(model)
    model.pipe = SimpleNamespace(device='cpu')
    model.extra_inputs = args.extra_inputs.split(',')
    model.use_gradient_checkpointing = True
    model.use_gradient_checkpointing_offload = False
    shared, positive, _ = model.get_pipeline_inputs(data)
    assert shared['edit_image'] == data['edit_image']
    assert positive['prompt'] == '中文提示词'
    assert shared['input_image'].size == tuple(adapted.engine['bucket_sizes'][0])
    captured = {}
    def dit(**kwargs):
        captured.update(kwargs)
        return kwargs['hidden_states']
    latent = torch.randn(1, 64, 4, 4)
    references = [torch.randn(1, 64, 4, 6), torch.randn(1, 64, 6, 4)]
    output = pipeline.model_fn_qwen_image_21(dit, latent, torch.tensor([500.]), torch.randn(1, 8, 4), None,
        torch.tensor([[True, True, False, False, False, False, False, False]]), edit_latents=references)
    assert captured['hidden_states'].shape[1] == 64  # 24 + 24 reference tokens + 16 target tokens.
    assert captured['img_shapes'] == [[(1, 4, 6), (1, 6, 4), (1, 4, 4)]]
    assert torch.equal(output, latent)  # Reference tokens never enter the target loss.


def test_existing_dry_run_api_emits_edit_contract(configured, monkeypatch):
    from fastapi.testclient import TestClient
    from mikazuki.app.application import app
    rt, config = configured
    config = edit_config(rt, config)
    monkeypatch.chdir(rt.project_root)
    client = TestClient(app)
    response = client.post('/api/engines/diffsynth/dry-run', json=config).json()
    assert response['status'] == 'success', response
    stored = json.loads(Path(response['data']['engine_config_path']).read_text())
    assert stored['arguments']['extra_inputs'] == 'edit_image'
    assert stored['training_task'] == 'image-edit'
    rows = json.loads(Path(stored['arguments']['dataset_metadata_path']).read_text())
    assert len(rows[0]['edit_image']) == 1
    response = client.post('/api/engines/diffsynth/dry-run', json={**config, 'control_data_dirs': []}).json()
    assert response['status'] != 'success'
    assert '参考图目录' in response['message']


@pytest.mark.parametrize('count', [128, 512])
def test_reference_index_scans_each_parent_once(tmp_path, monkeypatch, count):
    targets = tmp_path / 'targets'
    controls = [tmp_path / 'refs-b', tmp_path / 'refs-a']
    directories = ['', '2_nested']
    for base in [targets, *controls]:
        for directory in directories:
            (base / directory).mkdir(parents=True, exist_ok=True)
    for i in range(count):
        relative = Path(directories[i % 2]) / f'{i:04d}'
        Image.new('RGB', (32, 32)).save((targets / relative).with_suffix('.png'))
        (targets / relative).with_suffix('.txt').write_text('edit')
        for base in controls:
            Image.new('RGB', (32, 32)).save((base / relative).with_suffix('.webp'))
    original = Path.glob
    scans, candidates = [], []
    def counted(path, pattern, *args, **kwargs):
        if pattern == '*' and any(path == base or path.is_relative_to(base) for base in controls):
            scans.append(path)
            entries = list(original(path, pattern, *args, **kwargs))
            candidates.extend(entries)
            return iter(entries)
        return original(path, pattern, *args, **kwargs)
    monkeypatch.setattr(Path, 'glob', counted)
    config = dict(training_task='image-edit', train_data_dir=str(targets), control_data_dirs=list(map(str, controls)))
    _, _, rows = dataset_inputs(config, tmp_path)
    assert len(scans) == 4  # 2 reference groups × 2 parent directories, independent of N.
    assert len(candidates) == count * 2 + 2  # Files plus one subdirectory per group.
    assert len(rows) == count * 3 // 2  # Nested files repeated twice.
    first = rows[0]
    assert first['edit_image'] == [str((base / first['image']).with_suffix('.webp')) for base in controls]
    print(f'pairing: targets={count}, scans={len(scans)}, candidates={len(candidates)}')
    duplicate = (controls[0] / first['image']).with_suffix('.jpg')
    Image.new('RGB', (32, 32)).save(duplicate)
    with pytest.raises(ValueError, match='找到 2'):
        dataset_inputs(config, tmp_path)
    with pytest.raises(ValueError, match='互不包含'):
        dataset_inputs({**config, 'control_data_dirs': [str(targets)]}, tmp_path)


def test_pinned_encoder_temporary_hooks_are_cleaned_on_success_and_failure():
    torch = pytest.importorskip('torch')
    upstream = pytest.importorskip('diffsynth.models.qwen_image_21_text_encoder')
    from mikazuki.engines.diffsynth.text_encoder_hooks import cleanup_text_encoder_hooks, install_text_encoder_hook_cleanup
    class TinyHF(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.norm = torch.nn.Identity()
            self.fail = False
            self.debug = None
        def forward(self, input_ids=None, **kwargs):
            norm = self.model.language_model.norm
            if self.debug is None:
                self.debug = norm.register_forward_hook(lambda *_: None)
            if self.fail:
                raise RuntimeError('encoding failed')
            return norm(input_ids)
    encoder = upstream.QwenImage21TextEncoder.__new__(upstream.QwenImage21TextEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.model = TinyHF()
    norm = encoder.model.model.language_model.norm
    baseline = norm.register_forward_hook(lambda *_: None)
    value = torch.randn(1, 3, 4)
    # Same cleanup used by prepare_cache, with the real pinned forward method.
    with cleanup_text_encoder_hooks(encoder):
        assert torch.equal(encoder(input_ids=value), value)
    assert set(norm._forward_hooks) == {baseline.id, encoder.model.debug.id}
    install_text_encoder_hook_cleanup(encoder)
    installed = encoder.forward
    install_text_encoder_hook_cleanup(encoder)
    assert encoder.forward is installed
    for _ in range(32):
        assert torch.equal(encoder(input_ids=value), value)
        assert set(norm._forward_hooks) == {baseline.id, encoder.model.debug.id}
    encoder.model.fail = True
    for _ in range(3):
        with pytest.raises(RuntimeError, match='encoding failed'):
            encoder(input_ids=value)
        assert set(norm._forward_hooks) == {baseline.id, encoder.model.debug.id}
    assert 'register_forward_hook' not in norm.__dict__


def test_restored_edit_api_task_preserves_engine_arguments(configured, monkeypatch):
    from fastapi.testclient import TestClient
    from mikazuki.app.application import app
    from mikazuki.engines.diffsynth import run
    from mikazuki.utils.config_import import validate_config_import
    rt, config = configured
    config = edit_config(rt, config)
    monkeypatch.chdir(rt.project_root)
    monkeypatch.setattr(run, 'check_runtime', lambda _: None)
    submitted = []
    monkeypatch.setattr(run.tm, 'submit', submitted.append)
    config = validate_config_import('qwen-image-21-lora', config)['config']
    result = TestClient(app).post('/api/run', json=config).json()
    assert result['status'] == 'success', result
    task = submitted[0]
    try:
        payload = json.loads(Path(task.metadata['engine_config_path']).read_text())
        assert payload['arguments']['data_file_keys'] == 'image,edit_image'
        assert payload['arguments']['extra_inputs'] == 'edit_image'
        assert payload['training_task'] == 'image-edit'
    finally:
        run.tm.tasks.pop(task.task_id)
