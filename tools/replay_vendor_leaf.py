"""Replay a captured Torch leaf forward/backward without importing UP/platform.

Only trusted captures produced by observe_vendor_prefix.py are accepted. This
diagnostic uses a common synthetic upstream gradient, not a training change.
Each repetition restores module state and its own vendor's captured RNG state.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import torch

from observe_vendor_prefix import saved_mask, snapshot, tensor_info


def digest(value):
    if isinstance(value, torch.Tensor):
        return tensor_info(value)['sha256']
    if isinstance(value, dict):
        return {key: digest(child) for key, child in value.items()}
    return value


def restore(tensor, stride=None):
    value = tensor.to('cuda:0')
    if stride is None or tuple(value.stride()) == tuple(stride):
        return value
    result = torch.empty_strided(tensor.shape, stride, dtype=tensor.dtype, device='cuda:0')
    # Expanded views cannot be populated by copy_ over overlapping addresses.
    target, source = result, tensor
    for i, (size, step) in enumerate(zip(tensor.shape, stride)):
        if size > 1 and step == 0:
            target, source = target.narrow(i, 0, 1), source.narrow(i, 0, 1)
    target.copy_(source)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True, help='Common NVIDIA logical operands/state')
    p.add_argument('--own-rng', type=Path, required=True, help='This vendor own capture of the same leaf')
    p.add_argument('--layout', choices=('native', 'reference'), required=True)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    payload = torch.load(a.capture, map_location='cpu', weights_only=False)
    native = torch.load(a.own_rng, map_location='cpu', weights_only=False)
    expression = ast.parse(payload['repr'], mode='eval').body
    allowed = ('Conv2d', 'BatchNorm2d', 'LayerNorm', 'Dropout', 'SiLU', 'ReLU', 'MaxPool2d')
    if not isinstance(expression, ast.Call) or expression.func.id not in allowed:
        raise ValueError('Unsupported leaf repr ' + payload['repr'])
    constructor = getattr(torch.nn, expression.func.id)
    positional = [ast.literal_eval(node) for node in expression.args]
    keywords = {node.arg: ast.literal_eval(node.value) for node in expression.keywords}
    records = []
    first = None
    for _ in range(a.repeats):
        module = constructor(*positional, **keywords).to('cuda:0').train()
        module.load_state_dict(payload['state_before'], strict=True)
        if a.layout == 'reference':
            for key, info in payload['parameters'].items():
                original = getattr(module, key)
                tensor = restore(payload['state_before'][key], info['stride'])
                setattr(module, key, torch.nn.Parameter(tensor, requires_grad=original.requires_grad))
        info = payload['input_info'][0]
        x = restore(payload['inputs'][0], info['stride'] if a.layout == 'reference' else None).requires_grad_(True)
        torch.cuda.set_rng_state(native['rng_before'])
        rng_before = tensor_info(torch.cuda.get_rng_state())
        y = module(x)
        mask, origin = saved_mask(y) if expression.func.id == 'Dropout' else (None, None)
        # Same upstream operand on both GPUs, unrelated to their RNG streams.
        dy = (torch.arange(y.numel(), dtype=torch.float32).remainder(97) / 97).reshape(y.shape).to('cuda:0')
        y.backward(dy)
        result = snapshot({'output': y, 'input_gradient': x.grad,
                           'parameter_gradients': {key: value.grad for key, value in module.named_parameters()},
                           'state_after': module.state_dict(), 'mask': mask})
        fingerprint = digest(result)
        record = {'fingerprint': fingerprint, 'rng_before': rng_before,
                  'rng_after': tensor_info(torch.cuda.get_rng_state()), 'input': tensor_info(x),
                  'output': tensor_info(y), 'mask_source': origin}
        if first is None:
            first = fingerprint
            torch.save(result, a.output / 'first.pt')
        record['exact_repeat'] = first == fingerprint
        records.append(record)
    report = {'scope': 'Torch-only captured semantic leaf forward/backward, no UP import',
              'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
              'layout': a.layout, 'repr': payload['repr'], 'rtol': 1e-4, 'atol': 1e-5,
              'common_capture_sha256': hashlib.sha256(a.capture.read_bytes()).hexdigest(),
              'own_rng_capture_sha256': hashlib.sha256(a.own_rng.read_bytes()).hexdigest(),
              'all_repeats_exact': all(r['exact_repeat'] for r in records), 'repeats': records,
              'up_imported': 'up' in sys.modules, 'up_platform_imported': 'up_platform' in sys.modules}
    (a.output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: report[key] for key in ('repr', 'layout', 'all_repeats_exact', 'up_imported', 'up_platform_imported')}))


if __name__ == '__main__':
    main()
