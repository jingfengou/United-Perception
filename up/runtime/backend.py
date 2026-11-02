"""Device-specific PyTorch services; model and optimizer operations stay in PyTorch."""
import importlib

import torch


_PLATFORMS = {
    'nvidia': ('cuda', 'nccl', None),
    'ascend_npu': ('npu', 'hccl', 'torch_npu'),
    'cambricon_mlu': ('mlu', 'cncl', 'torch_mlu'),
    'metax_cuda': ('cuda', 'nccl', None),
}
_runtime = None


class RuntimeBackend:
    def __init__(self, platform):
        if platform not in _PLATFORMS:
            raise ValueError('Unknown runtime.platform: {}'.format(platform))
        self.platform = platform
        self.device_type, self.dist_backend, extension = _PLATFORMS[platform]
        if extension is not None:
            importlib.import_module(extension)
        self.module = getattr(torch, self.device_type)

    @property
    def device(self):
        return torch.device(self.device_type, self.module.current_device())

    @property
    def amp(self):
        return self.module.amp

    def set_device(self, local_rank):
        if not self.module.is_available():
            raise RuntimeError('{} is unavailable in this PyTorch environment'.format(self.platform))
        if local_rank >= self.module.device_count():
            raise RuntimeError('Local rank {} exceeds visible device count'.format(local_rank))
        self.module.set_device(local_rank)


def setup_runtime(config, opts=None):
    # Resolve the same --opts overrides as the runner, before creating workers/groups.
    if opts:
        from up.utils.general.cfg_helper import merge_opts_into_cfg
        config = merge_opts_into_cfg(opts, config)
    platform = config.get('runtime', {}).get('platform', 'nvidia')
    global _runtime
    if _runtime is None or _runtime.platform != platform:
        _runtime = RuntimeBackend(platform)
    return _runtime


def get_runtime():
    global _runtime
    if _runtime is None:
        _runtime = RuntimeBackend('nvidia')
    return _runtime
