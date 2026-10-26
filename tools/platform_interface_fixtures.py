"""Small independent operands for the public Torch-compatible interfaces.

These are operator contract fixtures, not accuracy or performance benchmarks.
References resolve the declared Torch API, not the platform's torch_impl module.
"""
import importlib
from types import SimpleNamespace

import torch
from torch.utils._pytree import tree_map

from platform_operator_cases import copy_tree, tensor_info


TORCH_CATEGORIES = ('activation', 'attention', 'convolution', 'loss', 'matrix',
                    'normalization', 'pooling')
DISTRIBUTED_NORMALIZATION = {'sync_batch_norm', 'sync_batch_norm_forward'}


def interface_methods(category):
    module = importlib.import_module('up_platform.ops.'+category+'.base')
    base = next(value for key, value in vars(module).items() if key.endswith('Ops'))
    return base.__abstractmethods__


def resolve_torch_reference(category, name):
    module = importlib.import_module('up_platform.ops.'+category+'.base')
    base = next(value for key, value in vars(module).items() if key.endswith('Ops'))
    source = getattr(base, name).contract.source
    if not source.startswith('torch.'):
        raise ValueError('Not an independent Torch API reference: '+source)
    value = torch
    for part in source.split('.')[1:]:
        value = getattr(value, part)
    return value, source


def reference_platform(platform):
    providers = {}
    for category in TORCH_CATEGORIES:
        module = importlib.import_module('up_platform.ops.'+category+'.base')
        base = next(value for key, value in vars(module).items() if key.endswith('Ops'))
        providers[category] = SimpleNamespace(**{name: resolve_torch_reference(category, name)[0]
            for name in base.__abstractmethods__
            if name not in DISTRIBUTED_NORMALIZATION})
    return SimpleNamespace(device=platform.device, category=providers.__getitem__)


def fixture(operation, *, dtype=torch.float32, layout='contiguous', training=True, amp=False,
            device_type='cuda', inplace=False):
    """Return execute_case's portable payload with constructor/RNG/state contracts."""
    category, name = operation.split('.')
    torch.manual_seed(131)

    def tensor(shape, *, grad=True, positive=False):
        value = ((torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float64)
                  .reshape(shape) % 29)-14)/19
        if positive:
            value = value.abs()/2+.1
        return value.to(dtype=dtype).requires_grad_(training and grad)

    x = tensor((2, 4, 6, 6))
    args, kwargs, constructor, module = (x,), {}, None, None
    if category == 'activation':
        kwargs = {'dim': 1} if name in ('softmax', 'log_softmax', 'Softmax') else {}
        if name[:1].isupper():
            constructor = ((), kwargs)
            kwargs = {}
    elif category == 'matrix':
        a, b = tensor((3, 4)), tensor((4, 5))
        args = (a, b)
        if name in ('linear', 'Linear'):
            args = (a, tensor((5, 4)), tensor((5,)))
            if name == 'Linear':
                constructor, args = ((4, 5), {}), (a,)
        elif name == 'bmm':
            args = (tensor((2, 3, 4)), tensor((2, 4, 5)))
        elif name == 'addmm':
            args, kwargs = (tensor((3, 5)), a, b), {'beta': .5, 'alpha': 1.5}
        elif name == 'einsum':
            args = ('ij,jk->ik', a, b)
    elif category == 'convolution':
        if name[:1].isupper():
            constructor = ((4, 6, 3), {'padding': 1, 'groups': 2})
        elif name == 'conv2d':
            args, kwargs = (x, tensor((6, 2, 3, 3)), tensor((6,))), {'padding': 1, 'groups': 2}
        else:
            args, kwargs = (x, tensor((4, 3, 3, 3)), tensor((6,))), {'padding': 1, 'groups': 2}
    elif category == 'attention':
        constructor = ((8, 2), {'dropout': .2, 'batch_first': True})
        query = tensor((2, 3, 8))
        args = (query, query, query)
        kwargs = {'need_weights': True}
    elif category == 'normalization':
        if name in DISTRIBUTED_NORMALIZATION | {'batch_norm_module_forward'}:
            raise ValueError('This interface needs the explicit state/distributed fixture: '+operation)
        if name in ('BatchNorm1d', 'BatchNorm2d', 'LayerNorm'):
            constructor = (((4,) if name != 'LayerNorm' else ((6, 6),)), {})
            if name == 'BatchNorm1d':
                args = (tensor((2, 4, 6)),)
        elif name == 'batch_norm':
            args = (x, torch.zeros(4, dtype=dtype), torch.ones(4, dtype=dtype),
                    tensor((4,)), tensor((4,)))
            kwargs = {'training': training, 'momentum': .2}
        elif name == 'layer_norm':
            args = (x, (6, 6), tensor((6, 6)), tensor((6, 6)))
        elif name == 'group_norm':
            args = (x, 2, tensor((4,)), tensor((4,)))
        else:
            kwargs = {'p': 2., 'dim': 1}
    elif category == 'pooling':
        if name in ('AdaptiveAvgPool2d', 'adaptive_avg_pool2d'):
            constructor = ((2,), {}) if name[:1].isupper() else None
            args = (x,) if constructor else (x, 2)
        elif name in ('AvgPool2d', 'MaxPool2d', 'max_pool2d'):
            constructor = ((2,), {}) if name[:1].isupper() else None
            args = (x,) if constructor else (x, 2)
        elif name in ('Upsample', 'UpsamplingBilinear2d', 'interpolate'):
            config = {'size': (8, 8)}
            if name in ('Upsample', 'interpolate'):
                config.update(mode='bilinear', align_corners=False)
            constructor = ((), config) if name[:1].isupper() else None
            kwargs = {} if constructor else config
        elif name == 'ZeroPad2d':
            constructor = (((1, 2, 0, 1),), {})
        else:
            args, kwargs = (x, (1, 2, 0, 1)), {'mode': 'constant', 'value': .25}
    elif category == 'loss':
        logits, target = tensor((3, 5)), tensor((3, 5), grad=False, positive=True)
        args = (logits, target)
        if name in ('cross_entropy', 'CrossEntropyLoss'):
            args = (logits, torch.tensor([0, 4, -100], dtype=torch.long))
            kwargs = {'label_smoothing': .1}
        elif name in ('binary_cross_entropy',):
            args = (logits.detach().sigmoid().requires_grad_(training), target)
            kwargs = {'reduction': 'none'}
        elif name in ('kl_div', 'KLDivLoss'):
            args = (logits.detach().log_softmax(1).requires_grad_(training), target.softmax(1))
            kwargs = {'reduction': 'batchmean'}
        if name[:1].isupper():
            constructor = ((), kwargs)
            kwargs = {}
    else:
        raise ValueError(operation)

    if layout == 'strided':
        changed = False
        def stride(value):
            nonlocal changed
            if changed or not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.ndim == 0:
                return value
            changed = True
            storage = torch.empty((*value.shape[:-1], value.shape[-1]*2+1), dtype=value.dtype)
            view = storage[..., 1::2]
            view.copy_(value.detach())
            return view.requires_grad_(value.requires_grad)
        args = tree_map(stride, args)
    elif layout == 'empty':
        leading = next(value for value in args if isinstance(value, torch.Tensor)).shape[0]
        args = tree_map(lambda value: value[:0] if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == leading else value, args)
    if inplace:
        if constructor is not None:
            constructor = (constructor[0], dict(constructor[1], inplace=True))
        else:
            kwargs['inplace'] = True
    if constructor is not None:
        reference, _ = resolve_torch_reference(category, name)
        module = reference(*constructor[0], **constructor[1]).to(dtype=dtype)
        module.train(training)
    payload = {'operation': operation, 'phase': 'training' if training else 'inference',
        'kind': 'module' if module is not None else 'function',
        'arguments': copy_tree((args, kwargs)),
        'argument_info': tree_map(lambda value: dict(tensor_info(value), device_type=device_type) if isinstance(value, torch.Tensor) else None,
                                  (args, kwargs)),
        'autocast': {'enabled': amp, 'dtype': 'torch.float16'},
        'grad_enabled': training, 'stochastic': name in ('Dropout', 'Dropout2d', 'MultiheadAttention'),
        'inplace': inplace}
    if module is not None:
        payload.update(constructor=constructor, training=training, state=copy_tree(module.state_dict()),
            state_info={key: tensor_info(value) for key, value in module.state_dict().items()})
    return payload
