"""Observe actual YOLO GPU indexed writes and random calls for five updates.

Every model call runs once and returns its original result. Mutable operands
are copied before execution so standalone replays can reset the destination.
"""
import argparse
import collections
import collections.abc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from observe_vendor_prefix import snapshot, tensor_info


class Writes(TorchDispatchMode):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.step = 0
        self.records, self.random_calls = [], []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        gpu = any(isinstance(x, torch.Tensor) and x.is_cuda for x in tree_flatten((args, kwargs))[0])
        requested_device = kwargs.get('device')
        if isinstance(requested_device, (str, torch.device)):
            gpu = gpu or torch.device(requested_device).type == 'cuda'
        if not gpu:
            return func(*args, **kwargs)
        if any(token in name for token in ('rand', 'dropout', 'bernoulli', 'normal_', 'uniform_')):
            self.random_calls.append({'step': self.step, 'operation': name})
        if name not in ('aten.index_put_.default', 'aten._index_put_impl_.default'):
            return func(*args, **kwargs)
        before = snapshot((args, kwargs))
        metadata = tensor_info((args, kwargs))
        result = func(*args, **kwargs)
        index = len(self.records)
        filename = 'write-%04d.pt' % index
        torch.save({'operation': name, 'arguments': before, 'input_info': metadata,
                    'output': snapshot(result)}, self.directory / filename)
        indices = before[0][1]
        duplicate_count = None
        if len(indices) == args[0].ndim and all(isinstance(x, torch.Tensor) and x.dtype == torch.long for x in indices):
            broadcast = torch.broadcast_tensors(*indices)
            linear = torch.zeros_like(broadcast[0])
            for size, value in zip(args[0].shape, broadcast):
                linear = linear * size + value
            duplicate_count = linear.numel() - torch.unique(linear).numel()
        self.records.append({'index': index, 'step': self.step, 'operation': name, 'capture': filename,
                             'target_shape': list(args[0].shape), 'duplicate_indices': duplicate_count,
                             'value': tensor_info(before[0][2]), 'output': tensor_info(result)})
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'frozen-tools', 'data', 'initial', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--lane', choices=('new_nvidia', 'new_metax'), required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.frozen_tools))
    from accuracy_train import configure_numerics, run
    configure_numerics(True)
    collections.Iterable = collections.abc.Iterable
    np.float, np.int, np.bool = float, int, bool
    sys.path.insert(0, str(a.source.resolve()))
    from up_platform import configure_platform
    configure_platform({'backend': 'metax' if a.lane == 'new_metax' else 'nvidia', 'device': 'cuda:0'}, purpose='training')
    from up.runner.base_runner import BaseRunner
    forward = BaseRunner.forward_train
    observer = Writes(a.output)

    def observed_forward(runner, batch):
        observer.step += 1
        with observer:
            result = forward(runner, batch)
        (a.output / 'writes.json').write_text(json.dumps({
            'scope': 'Synchronized first-five-update forward diagnostic, not a new accuracy run',
            'observer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'records': observer.records, 'gpu_random_calls': observer.random_calls}, indent=2, sort_keys=True) + '\n')
        return result

    BaseRunner.forward_train = observed_forward
    a.model, a.mode, a.probe_steps, a.deterministic = 'yolov5s', 'train', 5, True
    run(a)


if __name__ == '__main__':
    main()
