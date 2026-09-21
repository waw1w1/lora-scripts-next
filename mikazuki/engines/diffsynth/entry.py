"""Thin standalone entry, using the complete pinned DiffSynth checkout."""
import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mikazuki.engines.diffsynth.resource import environment_lock
from mikazuki.engines.diffsynth.settings import Runtime, TRAIN_SCRIPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--check-only', action='store_true', help='Validate config and upstream parser; never load model tensors or train.')
    options = parser.parse_args()
    rt = Runtime(Path(options.project_root))
    config = json.loads(Path(options.config).read_text(encoding='utf-8'))
    sys.path.insert(0, str(rt.source))
    spec = importlib.util.spec_from_file_location('qwen21_upstream', rt.source / TRAIN_SCRIPT)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    argv = []
    for key, value in config['arguments'].items():
        if isinstance(value, bool):
            if value:
                argv.append('--' + key)
        else:
            argv.extend(['--' + key, str(value)])
    args = upstream.qwen_image_21_parser().parse_args(argv)
    if options.check_only:
        print(json.dumps({'arguments': vars(args), 'samples': config['samples'], 'model_components': [m['component'] for m in config['models']]}, ensure_ascii=False))
        return
    from accelerate import Accelerator
    from diffsynth.core import UnifiedDataset
    from diffsynth.diffusion import launch_training_task
    from mikazuki.engines.diffsynth.model_cache import materialize_models
    from mikazuki.engines.diffsynth.training import QwenLogger
    with environment_lock(rt.root):
        args.model_paths = json.dumps(materialize_models(config['models'], config['cache_dir']))
        output = Path(args.output_path)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'engine_config.json').write_text(json.dumps({**config, 'arguments': vars(args)}, ensure_ascii=False, indent=2), encoding='utf-8')
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision='bf16')
        dataset = UnifiedDataset(base_path=args.dataset_base_path, metadata_path=args.dataset_metadata_path,
                                 repeat=args.dataset_repeat, data_file_keys=['image'],
                                 main_data_operator=UnifiedDataset.default_image_operator(base_path=args.dataset_base_path,
                                     max_pixels=args.max_pixels, height=args.height, width=args.width,
                                     height_division_factor=32, width_division_factor=32, convert_RGB=False, convert_RGBA=True))
        # pandas represents an explicitly empty CSV caption as NaN. The UI preflight
        # already distinguishes it from a missing prompt column; preserve empty TXT semantics.
        for row in dataset.data:
            if isinstance(row['prompt'], float) and row['prompt'] != row['prompt']:
                row['prompt'] = ''
        parameters = inspect.signature(upstream.QwenImage21TrainingModule).parameters
        model_args = {key: value for key, value in vars(args).items() if key in parameters}
        model_args['device'] = 'cpu' if args.initialize_model_on_cpu or args.enable_model_cpu_offload else accelerator.device
        model = upstream.QwenImage21TrainingModule(**model_args)
        logger = QwenLogger(args.output_path, config['samples'], config['output_name'])
        launch_training_task(accelerator, dataset, model, logger, args=args)


if __name__ == '__main__':
    main()
