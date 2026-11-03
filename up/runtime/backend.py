"""Device-specific PyTorch services; model and optimizer operations stay in PyTorch."""
import importlib

import torch


# 每项依次指定设备 API、通信后端、注册设备所需的扩展；厂商 Torch 须在启动前安装好。
# NVIDIA 和沐曦虽然都使用 cuda 名称，实际由各自环境中的 Torch/SDK 执行。
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
            # 先导入 torch_npu/torch_mlu 注册设备，之后才能取得 torch.npu/torch.mlu。
            try:
                importlib.import_module(extension)
            except ImportError as error:
                raise RuntimeError('{} requires a matching {} installation: {}'.format(
                    platform, extension, error)) from error
        self.module = getattr(torch, self.device_type, None)
        if self.module is None:
            raise RuntimeError('{} does not provide torch.{}'.format(platform, self.device_type))

    @property
    def device(self):
        return torch.device(self.device_type, self.module.current_device())

    @property
    def amp(self):
        # AMP 使用当前设备自己的实现，模型和优化器仍按原来的 PyTorch 流程工作。
        return self.module.amp

    def set_device(self, local_rank):
        if not self.module.is_available():
            raise RuntimeError('{} is unavailable in this PyTorch environment'.format(self.platform))
        if local_rank < 0 or local_rank >= self.module.device_count():
            raise RuntimeError('Local rank {} exceeds visible device count'.format(local_rank))
        self.module.set_device(local_rank)

    def memory_available(self, device, needed_memory, reserve_bytes=0.5 * 1024 ** 3):
        """Keep UP's allocation estimate; query the selected device API, not CUDA globally."""
        if device.type == 'cpu' or not self.module.is_available():
            return False
        total = self.module.get_device_properties(device.index).total_memory
        allocated = self.module.memory_allocated(device)
        return needed_memory < total - allocated - reserve_bytes

    def configure_dnn(self, options=None, legacy_nocudnn=None):
        """Apply only requested policies; CANN/CNNL do not inherit cuDNN switches.

        MetaX exposes CUDA-compatible PyTorch settings. Its owner must verify
        that these settings control mcDNN in the installed MACA release.
        Non-CUDA vendors can override this method once their SDK policy is verified.
        """
        options = dict(options or {})
        valid = {'enabled', 'benchmark', 'deterministic', 'allow_tf32'}
        if set(options) - valid or any(type(value) is not bool for value in options.values()):
            raise ValueError('runtime.dnn accepts boolean {}'.format(sorted(valid)))
        if legacy_nocudnn:
            if options.get('enabled') is True:
                raise ValueError('--nocudnn conflicts with runtime.dnn.enabled=true')
            options['enabled'] = False
        elif legacy_nocudnn is not None and self.device_type == 'cuda':
            # Preserve the original Runner default without overriding explicit YAML.
            options.setdefault('enabled', True)
        if not options:
            return
        if self.device_type != 'cuda':
            raise NotImplementedError('{}: runtime.dnn {} needs a verified vendor policy; '
                                      'cuDNN flags cannot control CANN/CNNL'.format(
                                          self.platform, sorted(options)))
        api = torch.backends.cudnn
        for name in options:
            if not hasattr(api, name):
                raise NotImplementedError('{} does not expose the DNN policy {}'.format(self.platform, name))
        for name, value in options.items():
            setattr(api, name, value)


def setup_runtime(config, opts=None):
    # 必须在创建工作进程/通信组之前解析厂商，训练和推理入口共用这一步。
    # Resolve the same --opts overrides as the runner, before creating workers/groups.
    if opts:
        from up.utils.general.cfg_helper import merge_opts_into_cfg
        config = merge_opts_into_cfg(opts, config)
    platform = config.get('runtime', {}).get('platform', 'nvidia')
    expected = _PLATFORMS.get(platform)
    device = config.get('runtime', {}).get('device')
    if expected and device is not None and str(device).split(':')[0] != expected[0]:
        raise ValueError('runtime.device={} conflicts with runtime.platform={}; expected {}'.format(
            device, platform, expected[0]))
    global _runtime
    if _runtime is None or _runtime.platform != platform:
        # 缓存已解析的运行时，后续设备查询无需反复导入 SDK。
        _runtime = RuntimeBackend(platform)
    return _runtime


def get_runtime():
    global _runtime
    if _runtime is None:
        _runtime = RuntimeBackend('nvidia')
    return _runtime
