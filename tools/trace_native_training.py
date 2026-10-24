"""Observe three real-data updates without changing model/optimizer algorithms.

Use the frozen accuracy worker. CPU gradient snapshots synchronize CUDA, so this
diagnostic is separate from the uninstrumented complete native training cohort.
"""
import argparse
import collections
import collections.abc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'frozen-tools', 'data', 'initial', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--lane', choices=('original_nvidia', 'new_nvidia'), required=True)
    parser.add_argument('--model', choices=('yolov5s', 'vit_b16'), required=True)
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()
    sys.path.insert(0, str(args.frozen_tools))
    from accuracy_train import run, CONTROLS
    if any(os.environ.get(key) is not None for key in CONTROLS):
        raise ValueError('Native diagnostics must not inherit determinism controls')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    collections.Iterable = collections.abc.Iterable
    np.float, np.int, np.bool = float, int, bool
    sys.path.insert(0, str(args.source.resolve()))
    if args.lane == 'new_nvidia':
        from up_platform import configure_platform
        configure_platform({'backend': 'nvidia', 'device': 'cuda:0'}, purpose='training')
    from up.runner.base_runner import BaseRunner
    from up.utils.general.hook_helper import Hook

    class Observe(Hook):
        def __init__(self, owner):
            super().__init__(owner)
            self.records = []
            self.current = None

        def before_forward(self, cur_iter, batch):
            self.current = {'step': len(self.records)+1,
                            'cpu_rng': digest_tensor(torch.get_rng_state()),
                            'gpu_rng': digest_tensor(torch.cuda.get_rng_state()),
                            'module_modes': {name: child.training for name, child in self.runner_ref().model.named_modules()}}

        def after_backward(self, cur_iter, loss):
            gradients = {name: parameter.grad.detach().cpu().clone()
                         for name, parameter in self.runner_ref().model.named_parameters() if parameter.grad is not None}
            path = args.output/('gradients-step'+str(self.current['step'])+'.pt')
            torch.save(gradients, path)
            self.current.update(loss=float(loss.detach()), gradients={name: digest_tensor(value) for name, value in gradients.items()},
                                gpu_rng_after=digest_tensor(torch.cuda.get_rng_state()))
            self.records.append(self.current)
            (args.output/'observations.json').write_text(json.dumps({'observer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'scope': 'CPU gradient snapshots synchronize CUDA; diagnostic only', 'records': self.records}, indent=2, sort_keys=True)+'\n')

    train = BaseRunner.train

    def observed_train(runner):
        modules = {}
        for name, module in runner.model.named_modules():
            if any(module.children()):
                continue
            try:
                source = inspect.getsource(type(module).forward).encode()
                forward_sha = hashlib.sha256(source).hexdigest()
            except (OSError, TypeError):
                forward_sha = None
            modules[name] = {'type': type(module).__module__+'.'+type(module).__qualname__,
                             'repr': repr(module), 'forward_sha256': forward_sha}
        (args.output/'leaf-modules.json').write_text(json.dumps(modules, indent=2, sort_keys=True)+'\n')
        runner._hooks.hooks.append(Observe(runner))
        return train(runner)

    BaseRunner.train = observed_train
    args.mode, args.probe_steps = 'train', args.steps
    run(args)


if __name__ == '__main__':
    main()
