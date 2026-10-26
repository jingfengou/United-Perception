"""Capture explicit platform calls and replay common logical operands.

Diagnostic artifacts only. CPU copies store evidence; all operator execution
uses the selected backend. This observer is never installed in accuracy runs.
"""
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path

import torch
from torch.utils._pytree import tree_flatten, tree_map


COMPUTE = {'matrix', 'convolution', 'normalization', 'activation', 'pooling',
           'loss', 'attention', 'detection', 'deformable', 'focal', 'pointcloud', 'sparse'}
RTOL, ATOL = 1e-4, 1e-5


def load_cases(directory):
    directory = Path(directory)
    record = json.loads((directory/'observation.json').read_text())
    if not record['completed_model_observation'] or not record['cases']:
        raise ValueError('Incomplete/empty observation is not numerical coverage')
    for case in record['cases']:
        path = directory/case['file']
        if path.name != case['file'] or hashlib.sha256(path.read_bytes()).hexdigest() != case['sha256']:
            raise ValueError('Captured operands changed: '+case['file'])
    return record


def tensor_info(value):
    return {'shape': list(value.shape), 'stride': list(value.stride()),
        'storage_offset': value.storage_offset(), 'dtype': str(value.dtype),
        'device_type': value.device.type, 'requires_grad': value.requires_grad}


def copy_tree(value):
    def clone(child):
        if isinstance(child, torch.Tensor):
            return child.detach().cpu().clone()
        if child is not None and not isinstance(child, (str, int, float, bool, torch.dtype, torch.device, torch.Size)):
            raise TypeError('Nonportable argument: '+type(child).__module__+'.'+type(child).__name__)
        return child
    return tree_map(clone, value)


def fingerprint(value):
    checksum = hashlib.sha256()
    leaves, spec = tree_flatten(value)
    checksum.update(str(spec).encode())
    for child in leaves:
        if isinstance(child, torch.Tensor):
            checksum.update(str((tuple(child.shape), child.dtype)).encode())
            raw = child.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            checksum.update(raw.numpy().tobytes())
        else:
            checksum.update(repr(child).encode())
    return checksum.hexdigest()


def restore_tensor(value, info, device):
    target = device if info['device_type'] != 'cpu' else 'cpu'
    shape, stride, offset = info['shape'], info['stride'], info['storage_offset']
    length = offset+1+sum(max(size-1, 0)*step for size, step in zip(shape, stride))
    storage = torch.empty(length, dtype=value.dtype, device=target)
    result = storage.as_strided(shape, stride, offset)
    source, destination = value, result
    for axis, (size, step) in enumerate(zip(shape, stride)):
        if size > 1 and step == 0:
            source = source.narrow(axis, 0, 1)
            destination = destination.narrow(axis, 0, 1)
    destination.copy_(source)
    return result


class PlatformCalls:
    def __init__(self, platform, output, *, capture=True, cases_per_op=2):
        self.platform, self.output, self.capture = platform, Path(output), capture
        self.limit, self.phase = cases_per_op, None
        self.counts, self.cases, self.variants, self.skipped = {}, [], {}, []
        self.saved, self.handles = [], []
        self.catalog = platform.capabilities()
        self.completed = False

    def note(self, name):
        key = (self.phase or 'setup')+'/'+name
        self.counts[key] = self.counts.get(key, 0)+1

    def record(self, name, arguments, *, constructor=None, module=None):
        self.note(name)
        if not self.phase or not self.capture:
            return
        tensors = [x for x in tree_flatten(arguments)[0] if isinstance(x, torch.Tensor)]
        if not tensors:
            return
        key = self.phase+'/'+name
        variant = repr([tensor_info(x) for x in tensors])
        variants = self.variants.setdefault(key, {})
        variants[variant] = variants.get(variant, 0)+1
        if any(row['key'] == key and row['variant'] == variant for row in self.cases):
            return
        if sum(row['key'] == key for row in self.cases) >= self.limit:
            return
        try:
            payload = {'operation': name, 'phase': self.phase,
                'kind': 'module' if module is not None else 'function',
                'arguments': copy_tree(arguments),
                'argument_info': tree_map(lambda x: tensor_info(x) if isinstance(x, torch.Tensor) else None, arguments),
                'autocast': {'enabled': torch.is_autocast_enabled(self.platform.device.device_type),
                    'dtype': str(torch.get_autocast_dtype(self.platform.device.device_type))},
                'grad_enabled': torch.is_grad_enabled(),
                'stochastic': name.split('.')[-1] in ('Dropout', 'Dropout2d'),
                'inplace': bool(getattr(module, 'inplace', False) if module is not None else arguments[1].get('inplace', False))}
            if module is not None:
                payload.update(constructor=copy_tree(constructor), training=module.training,
                    state=copy_tree(module.state_dict()),
                    state_info={key: tensor_info(value) for key, value in module.state_dict().items()})
        except TypeError as error:
            self.skipped.append({'key': key, 'reason': str(error)})
            return
        filename = 'case-%04d.pt' % len(self.cases)
        path = self.output/filename
        torch.save(payload, path)
        self.cases.append({'key': key, 'operation': name, 'phase': self.phase,
            'variant': variant, 'file': filename, 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'operands_sha256': fingerprint(payload['arguments']), 'stochastic': payload['stochastic']})

    def wrap(self, category, method, original):
        name = category+'.'+method
        def observed(*args, **kwargs):
            if method[:1].isupper():
                result = original(*args, **kwargs)
                self.note(name)
                if isinstance(result, torch.nn.Module):
                    constructor = (args, kwargs)
                    handle = result.register_forward_pre_hook(lambda module, values, named:
                        self.record(name, (values, named), constructor=constructor, module=module), with_kwargs=True)
                    self.handles.append(handle)
                return result
            self.record(name, (args, kwargs))
            return original(*args, **kwargs)
        return observed

    def __enter__(self):
        for name in self.catalog:
            category, method = name.split('.', 1)
            if category not in COMPUTE:
                continue
            provider = self.platform.category(category)
            old = getattr(provider, method)
            self.saved.append((provider, method, method in vars(provider), old))
            setattr(provider, method, self.wrap(category, method, old))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        for provider, method, own, old in reversed(self.saved):
            if own:
                setattr(provider, method, old)
            else:
                delattr(provider, method)

    @contextmanager
    def observing(self, phase):
        previous, self.phase = self.phase, phase
        try:
            yield
        finally:
            self.phase = previous

    def report(self):
        numeric = {case['key'] for case in self.cases}
        return {'completed_model_observation': self.completed,
            'scope': 'observed platform calls; at most two actual shape/layout variants per operation/phase by default',
            'vendor': self.platform.vendor, 'purpose': self.platform.purpose,
            'cases_per_operation_phase': self.limit, 'counts': self.counts,
            'cases': self.cases, 'nonportable_calls': self.skipped,
            'variant_counts': {key: len(value) for key, value in self.variants.items()},
            'observed_without_numeric_case': [key for key in self.counts if not key.startswith('setup/') and key not in numeric],
            'not_observed': [name for name in self.catalog if not any(key.endswith('/'+name) for key in self.counts)]}


def execute_case(platform, payload, *, direct=False, seed=131):
    category, name = payload['operation'].split('.')
    provider = importlib.import_module('up_platform.ops.'+category+'.torch_impl') if direct else platform.category(category)
    operation = getattr(provider, name)
    device = platform.device.device()
    leaves = []
    tensor_index = [0]
    def operand(value, info):
        if not isinstance(value, torch.Tensor):
            return value
        result = restore_tensor(value, info, device)
        first = tensor_index[0] == 0
        tensor_index[0] += 1
        # Also exercise the leading floating operand's derivative when a real
        # image batch did not require gradients. Never enable running-stat or
        # target gradients merely because those tensors are floating point.
        needs_grad = payload['grad_enabled'] and value.is_floating_point() and (info['requires_grad'] or first)
        if needs_grad:
            result.requires_grad_()
            leaves.append(result)
            if payload['inplace']:
                # A copy into fresh strided storage is a nonleaf connected to
                # the logical input, retaining its exact offset and strides.
                result = restore_tensor(result, info, device)
        return result
    args, kwargs = tree_map(operand, payload['arguments'], payload['argument_info'])
    module = None
    if payload['kind'] == 'module':
        constructor_args, constructor_kwargs = payload['constructor']
        constructor_kwargs = dict(constructor_kwargs)
        if constructor_kwargs.get('device') is not None:
            constructor_kwargs['device'] = device
        module = operation(*constructor_args, **constructor_kwargs).to(device)
        module.load_state_dict(payload['state'], strict=True)
        for key, value in list(module.named_parameters())+list(module.named_buffers()):
            value.data = restore_tensor(payload['state'][key],
                dict(payload['state_info'][key], device_type=platform.device.device_type), device)
        module.train(payload['training'])
        operation = module
    torch.manual_seed(seed)
    platform.device.manual_seed(seed)
    rng_before = platform.device.get_rng_state()
    dtype = getattr(torch, payload['autocast']['dtype'].split('.')[-1])
    with torch.set_grad_enabled(payload['grad_enabled']), torch.amp.autocast(
            platform.device.device_type, enabled=payload['autocast']['enabled'], dtype=dtype):
        output = operation(*args, **kwargs)
    floating = [x for x in tree_flatten(output)[0] if isinstance(x, torch.Tensor) and x.requires_grad]
    parameters = dict(module.named_parameters()) if module else {}
    differentiable = [*leaves, *(p for p in parameters.values() if p.requires_grad)]
    gradients = []
    if floating and differentiable:
        # Identical nonuniform upstream values, generated on the CPU rather
        # than drawing from incompatible vendor GPU random streams.
        upstream = [((torch.arange(x.numel(), device='cpu', dtype=torch.float32)%97-48)/97)
                    .reshape(x.shape).to(device=x.device, dtype=x.dtype) for x in floating]
        gradients = list(torch.autograd.grad(floating, differentiable, upstream, allow_unused=True))
    state = module.state_dict() if module else {}
    if payload['operation'] == 'normalization.batch_norm':
        state = {key: kwargs.get(key, args[index] if len(args) > index else None)
                 for index, key in ((1, 'running_mean'), (2, 'running_var'))}
    result = copy_tree({'output': output, 'input_gradients': gradients[:len(leaves)],
        'parameter_gradients': dict(zip((key for key, p in parameters.items() if p.requires_grad), gradients[len(leaves):])),
        'state_after': state,
        'arguments_after': (args, kwargs) if payload['inplace'] else None})
    return result, {'rng_before_sha256': fingerprint(rng_before),
                    'rng_after_sha256': fingerprint(platform.device.get_rng_state()),
                    'upstream_policy': '(CPU arange % 97 - 48) / 97, cast to output dtype',
                    'gradient_policy': 'observed differentiable operands plus leading floating operand',
                    'input_gradient_count': len(leaves)}


def compare_results(left, right):
    aa, spec_a = tree_flatten(left)
    bb, spec_b = tree_flatten(right)
    if spec_a != spec_b or len(aa) != len(bb):
        return {'passed': False, 'reason': 'result tree mismatch'}
    rows = []
    for index, (a, b) in enumerate(zip(aa, bb)):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            rows.append({'leaf': index, 'passed': type(a) is type(b) and a == b})
            continue
        same = a.shape == b.shape and a.dtype == b.dtype
        finite = same and (not a.is_floating_point() or bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
        close = finite and (torch.allclose(b, a, rtol=RTOL, atol=ATOL) if a.is_floating_point() else torch.equal(a, b))
        rows.append({'leaf': index, 'shape': list(a.shape), 'dtype': str(a.dtype),
            'passed': bool(close), 'bitwise_equal': same and torch.equal(a, b), 'finite': bool(finite),
            'max_abs': float((a.double()-b.double()).abs().max()) if finite and a.numel() else (0. if finite else None)})
    return {'passed': bool(rows) and all(row['passed'] for row in rows), 'leaves': rows}
