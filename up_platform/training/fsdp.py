"""Shared FSDP2 implementation, using only the selected vendor's PyTorch.

No UP dependency, CUDA binary selection, CPU model computation or eager SDK import.
"""
from contextlib import contextmanager
from copy import copy
from types import SimpleNamespace

import torch

from up_platform.contracts import UnsupportedOperation


def _api(device):
    try:
        from torch.distributed.fsdp import fully_shard, FSDPModule
        from torch.distributed.device_mesh import init_device_mesh
        if not hasattr(FSDPModule, 'set_reshard_after_forward'):
            raise ImportError('FSDP2 runtime reshard control requires PyTorch 2.10+')
    except ImportError as error:
        raise UnsupportedOperation(device.vendor, 'training', 'fsdp', str(error)) from error
    return fully_shard, FSDPModule, init_device_mesh


def wrap(device, module, *, shard_modules=(), reshard_after_forward=True):
    import torch.distributed as dist
    if device.device_type == 'cpu':
        raise UnsupportedOperation(device.vendor, 'training', 'fsdp', 'FSDP2 requires an accelerator')
    fully_shard, FSDPModule, init_device_mesh = _api(device)
    if not dist.is_initialized() or dist.get_world_size() < 2:
        raise ValueError('FSDP2 requires an initialized process group with at least two ranks')
    if type(reshard_after_forward) is not bool:
        raise ValueError('reshard_after_forward must be bool')
    if isinstance(shard_modules, str) or len(set(shard_modules)) != len(shard_modules):
        raise ValueError('shard_modules must be a list of distinct module paths')
    modules = dict(module.named_modules())
    for name in shard_modules:
        if not name or name not in modules:
            raise ValueError('Unknown FSDP shard module: ' + str(name))
    if any(isinstance(child, FSDPModule) for child in modules.values()):
        raise ValueError('Model has already been sharded')
    # DDP also synchronizes rank-0 initialization. FSDP2 does not do so itself.
    with torch.no_grad():
        for tensor in (*module.parameters(), *module.buffers()):
            dist.broadcast(tensor, src=0)
        # FSDP2 requires contiguous parameter storage. Vendor Conv.to(device)
        # may choose channels-last; normalize values/layout before any optimizer
        # exists. This changes neither logical weights nor the model algorithm.
        for parameter in module.parameters():
            if not parameter.is_contiguous():
                parameter.data = parameter.data.contiguous()
    mesh = init_device_mesh(device.device_type, (dist.get_world_size(),))
    for name in sorted(shard_modules, key=lambda n: n.count('.'), reverse=True) + ['']:
        child = modules[name]
        fully_shard(child, mesh=mesh, reshard_after_forward=reshard_after_forward)
        # Use the public setter also on the root; no automatic root exception.
        child.set_reshard_after_forward(reshard_after_forward, recurse=False)
        child._up_reshard_after_forward = reshard_after_forward
    return module


@contextmanager
def validation(module):
    """Gather once, so unequal numbers of validation batches do not deadlock.

    Validation temporarily needs full model memory on every rank. No parameter
    synchronization occurs inside per-rank forward loops; restore shards on exit.
    """
    sharded = [m for m in module.modules() if hasattr(m, '_up_reshard_after_forward')]
    try:
        for child in sharded:
            child.set_reshard_after_forward(False, recurse=False)
            child.unshard()
        yield
    finally:
        for child in reversed(sharded):
            child.reshard()
            child.set_reshard_after_forward(child._up_reshard_after_forward, recurse=False)


def scaler(device, **kwargs):
    _api(device)
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    from torch.distributed.tensor import DTensor

    class DTensorGradScaler(ShardedGradScaler):
        def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16=True):
            # The fused foreach AMP kernel operates on local Tensors. The views
            # share real GPU gradient storage; the original optimizer is untouched.
            local = SimpleNamespace(param_groups=[{'params': [SimpleNamespace(
                grad=p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad)
                for p in group['params']]} for group in optimizer.param_groups])
            return super()._unscale_grads_(local, inv_scale, found_inf, allow_fp16)

    return DTensorGradScaler(device=device.device_type, **kwargs)


def full_state_dict(module, optimizer=None):
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict, get_optimizer_state_dict, StateDictOptions,
    )
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    result = {'model': get_model_state_dict(module, options=options)}
    if optimizer is not None:
        result['optimizer'] = get_optimizer_state_dict(module, optimizer, options=options)
    return result


def _named_optimizer_state(module, optimizer, state):
    """Convert legacy IDs with the same ordered parameter-group contract as Torch."""
    groups = state['param_groups']
    if len(groups) != len(optimizer.param_groups):
        raise ValueError('Checkpoint optimizer parameter-group count differs')
    names = {id(p): name for name, p in module.named_parameters()}
    mapping = {}
    for saved, current in zip(groups, optimizer.param_groups):
        if len(saved['params']) != len(current['params']):
            raise ValueError('Checkpoint optimizer parameter-group size differs')
        for key, param in zip(saved['params'], current['params']):
            name = names[id(param)]
            if isinstance(key, str) and key != name:
                raise ValueError(f'Checkpoint optimizer parameter {key!r} does not match {name!r}')
            mapping[key] = name
    return {'state': {mapping[k]: v for k, v in state['state'].items()},
            'param_groups': [{**g, 'params': [mapping[k] for k in g['params']]} for g in groups]}


def _copy_containers(value):
    """DCP may replace dictionary entries with DTensors; keep caller state intact.

    Preserve metadata and Tensor storage without duplicating large CPU weights.
    """
    if isinstance(value, dict):
        result = copy(value)
        result.update((key, _copy_containers(item)) for key, item in value.items())
        return result
    if isinstance(value, list):
        return [_copy_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_containers(item) for item in value)
    return value


def load_full_state_dict(module, state, optimizer=None, *, strict=True):
    from torch.distributed.checkpoint.state_dict import (
        set_model_state_dict, set_optimizer_state_dict, StateDictOptions,
    )
    options = StateDictOptions(full_state_dict=True, strict=strict)
    if 'model' in state:
        set_model_state_dict(module, _copy_containers(state['model']), options=options)
    if 'optimizer' in state:
        if optimizer is None:
            raise ValueError('Checkpoint contains optimizer state but no optimizer was supplied')
        target = module.module if isinstance(module, torch.nn.parallel.DistributedDataParallel) else module
        named_state = _named_optimizer_state(target, optimizer, _copy_containers(state['optimizer']))
        set_optimizer_state_dict(module, optimizer, named_state, options=options)
