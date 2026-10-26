"""UP configuration stays in UP; the platform package has no YAML/CLI dependency."""
from up_platform import configure_platform
from contextlib import ExitStack
from functools import wraps


def fsdp_options(config, *, backend, purpose, runner_type):
    """Select the training strategy independently of vendor/communication backend."""
    options = dict(config.get('runtime', {}).get('distributed', {}))
    strategy = options.pop('strategy', 'ddp')
    if strategy not in ('ddp', 'fsdp2'):
        raise ValueError('runtime.distributed.strategy must be ddp or fsdp2')
    if strategy == 'ddp' or purpose == 'inference':
        if strategy == 'ddp' and options:
            raise ValueError('FSDP options require strategy: fsdp2')
        return None
    if options.keys() - {'shard_modules', 'reshard_after_forward'}:
        raise ValueError('Unknown FSDP options: ' + str(sorted(options)))
    if backend != 'dist' or runner_type != 'base':
        raise ValueError('FSDP2 requires --backend dist and the base runner; NAS/QAT/pruning runners are not supported')
    if config.get('runtime', {}).get('model_helper', {}).get('type', 'base') != 'base':
        raise ValueError('FSDP2 currently supports the base ModelHelper')
    optimizer = config.get('trainer', {}).get('optimizer', {})
    if optimizer.get('type', 'SGD') not in ('SGD', 'Adam', 'AdamW') or optimizer.get('register_type', 'base') != 'base':
        raise ValueError('FSDP2 currently supports base SGD/Adam/AdamW optimizers')
    if config.get('runtime', {}).get('async_norm', False):
        raise ValueError('FSDP2 does not support async_norm; choose the model normalization explicitly')
    if any(h['type'] == 'auto_checkpoint' for h in config.get('hooks', [])):
        raise ValueError('FSDP2 cannot collect a checkpoint from a rank-local signal handler; remove auto_checkpoint')
    return options


def fsdp_rank_state(model, scaler=None):
    """Gather rank-local buffers/RNG/scaler without CPU model computation."""
    import random
    import numpy as np
    import torch
    from up_platform import get_platform
    platform = get_platform()
    comm = platform.category('communication')
    rng = np.random.get_state()
    local = dict(buffers={k: v.detach().cpu().clone() for k, v in model.named_buffers()},
                 torch_rng=torch.get_rng_state(), device_rng=platform.device.get_rng_state(),
                 python_rng=random.getstate(), numpy_rng=(rng[0], rng[1].tolist(), *rng[2:]),
                 scaler=scaler.state_dict() if scaler is not None else None)
    states = [None] * comm.get_world_size()
    comm.all_gather_object(states, local)
    return states if comm.get_rank() == 0 else []


def restore_fsdp_rank_state(model, states, scaler=None):
    import random
    import numpy as np
    import torch
    from up_platform import get_platform
    platform = get_platform()
    comm = platform.category('communication')
    if len(states) != comm.get_world_size():
        raise ValueError('FSDP exact continuation requires the saved world size; use pretrain_model to load weights only')
    local = states[comm.get_rank()]
    buffers = dict(model.named_buffers())
    if buffers.keys() != local['buffers'].keys():
        raise ValueError('Checkpoint rank-local buffer names differ')
    with torch.no_grad():
        for name, buffer in buffers.items():
            buffer.copy_(local['buffers'][name])
    if scaler is not None and local['scaler'] is not None:
        scaler.load_state_dict(local['scaler'])
    torch.set_rng_state(local['torch_rng'])
    platform.device.set_rng_state(local['device_rng'])
    random.setstate(local['python_rng'])
    rng = local['numpy_rng']
    np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), *rng[2:]))


def training_validation(function):
    """Restore model/EMA modes around Runner evaluation, including exceptions."""
    @wraps(function)
    def wrapped(runner, *args, **kwargs):
        from up_platform import get_platform
        platform = get_platform()
        if platform.purpose != 'training':
            raise RuntimeError('Runner validation uses the training platform, not deployment inference')
        with ExitStack() as stack:
            stack.enter_context(platform.execution.validation(runner.model))
            if getattr(runner, 'ema', None) is not None:
                stack.enter_context(platform.execution.validation(runner.ema.model))
            return function(runner, *args, **kwargs)
    return wrapped


def configure_from_up(config, *, purpose):
    runtime = config.get('runtime', {})
    selection = dict(runtime.get('platform', {}))
    if 'device' in runtime:
        selection.setdefault('device', runtime['device'])
    if selection.get('device') == 'cpu':
        selection.setdefault('backend', 'cpu')
    return configure_platform(selection, purpose=purpose)


def configure_command(args, *, purpose):
    from up.utils.general.yaml_loader import load_yaml
    from up.utils.general.cfg_helper import merge_opts_into_cfg
    cfg = load_yaml(args.config, getattr(args, 'cfg_type', 'up'))
    cfg = merge_opts_into_cfg(getattr(args, 'opts', None), cfg)
    context = configure_from_up(cfg, purpose=purpose)
    args.platform_config = dict(cfg.get('runtime', {}).get('platform', {}))
    args.platform_config.setdefault('backend', context.vendor)
    if 'device' in cfg.get('runtime', {}):
        args.platform_config.setdefault('device', cfg['runtime']['device'])
    args.platform_purpose = purpose
    return context
