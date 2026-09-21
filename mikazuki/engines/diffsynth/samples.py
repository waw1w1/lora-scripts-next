"""Structured sample settings shared by preflight and the training callback."""
import json
import math

DEFAULT_SAMPLE = {'prompt': '', 'width': 1024, 'height': 1024, 'seed': 42, 'guidance_scale': 4, 'sample_steps': 20}


def sample_config(config):
    enabled = config.get('sample_enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('sample_enabled 必须是布尔值')
    if not enabled:
        return {'enabled': False, 'every_steps': 100, 'samples': []}
    interval = config.get('sample_every_n_steps', 100)
    try:
        valid = not isinstance(interval, bool) and int(interval) == float(interval) and int(interval) >= 1
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError('sample_every_n_steps 必须为正整数（优化器更新次数）')
    samples = config.get('preview_samples', [json.dumps(DEFAULT_SAMPLE)])
    if not isinstance(samples, list) or not samples:
        raise ValueError('preview_samples 必须是非空的 JSON 字符串列表')
    result = []
    for i, value in enumerate(samples):
        try:
            if not isinstance(value, str):
                raise ValueError()
            sample = json.loads(value)
        except (TypeError, ValueError):
            raise ValueError(f'preview_samples 第 {i + 1} 项必须是有效 JSON 字符串') from None
        if not isinstance(sample, dict) or set(sample) - (set(DEFAULT_SAMPLE) | {'controlImages'}):
            raise ValueError(f'预览样例 {i + 1} 包含不支持的参数')
        if sample.pop('controlImages', []) != []:
            raise ValueError(f'预览样例 {i + 1}: 文生图不支持参考图，controlImages 必须为空数组')
        sample = {**DEFAULT_SAMPLE, **sample}
        if not isinstance(sample['prompt'], str):
            raise ValueError(f'预览样例 {i + 1} 的 prompt 必须是字符串')
        for key in ('width', 'height', 'seed', 'sample_steps'):
            number = sample[key]
            minimum = 0 if key == 'seed' else 1
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or int(number) != number or number < minimum:
                raise ValueError(f'预览样例 {i + 1}: {key} 必须是 >= {minimum} 的整数')
            sample[key] = int(number)
        if sample['width'] % 32 or sample['height'] % 32:
            raise ValueError('预览宽高必须是 32 的倍数')
        try:
            cfg = float(sample['guidance_scale'])
            valid = not isinstance(sample['guidance_scale'], bool) and math.isfinite(cfg) and cfg >= 1
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f'preview_samples 第 {i + 1} 项 guidance_scale 必须是 >= 1 的有限数值') from None
        sample['guidance_scale'] = cfg
        result.append(sample)
    return {'enabled': True, 'every_steps': int(interval), 'samples': result}
