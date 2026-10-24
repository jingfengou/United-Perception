"""Observe a real-data training prefix without replacing any model operation.

CPU evidence copies synchronize the device. Compare these diagnostic prefixes
with the uninstrumented full-budget reports before interpreting an observation.
GPU RNG bytes are recorded per vendor, never transferred between vendors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils._pytree import tree_map


def snapshot(value):
    return tree_map(lambda x: x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x, value)


def tensor_info(value):
    def describe(x):
        if not isinstance(x, torch.Tensor):
            return repr(x)
        raw = x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return {'shape': list(x.shape), 'stride': list(x.stride()), 'dtype': str(x.dtype),
                'device': str(x.device), 'sha256': hashlib.sha256(raw).hexdigest()}
    return tree_map(describe, value)


def saved_mask(output):
    """Read the actual autograd-saved mask, without drawing new random values."""
    node = output.grad_fn
    if node is None:
        return None, None
    for name in ('_saved_result1', '_saved_other'):
        value = getattr(node, name, None)
        if isinstance(value, torch.Tensor):
            return value, type(node).__name__ + '.' + name
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'frozen-tools', 'data', 'initial', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--lane', choices=('original_nvidia', 'new_nvidia', 'new_metax'), required=True)
    parser.add_argument('--model', choices=('yolov5s', 'vit_b16'), required=True)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--leaves', type=int, default=8,
                        help='Save inputs, state and outputs of this first-step leaf prefix; zero is control')
    args = parser.parse_args()
    if args.steps < 1 or args.leaves < 0:
        parser.error('steps must be positive and leaves must be nonnegative')
    sys.path.insert(0, str(args.frozen_tools.resolve()))
    from accuracy_train import run, configure_numerics, batch_hash, state_hash, write
    configure_numerics(True)
    # The worker loads the correct source and platform before constructing Runner.
    # Patch only the train entry, after its initial eval has consumed host RNG.
    sys.path.insert(0, str(args.source.resolve()))
    import collections
    import collections.abc
    import numpy as np
    collections.Iterable = collections.abc.Iterable
    np.float, np.int, np.bool = float, int, bool
    if args.lane != 'original_nvidia':
        from up_platform import configure_platform
        configure_platform({'backend': 'metax' if args.lane == 'new_metax' else 'nvidia',
                            'device': 'cuda:0'}, purpose='training')
    from up.runner.base_runner import BaseRunner
    from up.utils.general.hook_helper import Hook

    class Observe(Hook):
        def __init__(self, owner):
            super().__init__(owner)
            self.steps = []
            self.leaves = []
            self.active = False
            self.calls = 0
            self.pending = {}

        def before_forward(self, cur_iter, batch):
            self.active = not self.steps
            self.current = {'step': len(self.steps) + 1, 'batch_sha256': batch_hash(batch),
                            'cpu_rng': tensor_info(torch.get_rng_state()),
                            'gpu_rng': tensor_info(torch.cuda.get_rng_state())}
            if self.active:
                torch.save(snapshot(batch), args.output / 'first-batch.pt')
                torch.save({'cpu': torch.get_rng_state(), 'gpu': torch.cuda.get_rng_state()},
                           args.output / 'rng-before.pt')

        def pre(self, name, module, inputs):
            if not self.active or args.leaves == 0:
                return
            index = self.calls
            self.calls += 1
            stochastic = isinstance(module, torch.nn.Dropout) or type(module).__name__ == 'DropPath'
            record = {'index': index, 'name': name, 'type': type(module).__module__ + '.' + type(module).__qualname__,
                      'training': module.training, 'stochastic': stochastic}
            if index < args.leaves or stochastic:
                record.update(input=tensor_info(inputs), gpu_rng_before=tensor_info(torch.cuda.get_rng_state()))
                payload = {'inputs': snapshot(inputs), 'input_info': tensor_info(inputs),
                           'state_before': snapshot(module.state_dict()),
                           'parameters': {key: tensor_info(value) for key, value in module.named_parameters(recurse=False)},
                           'repr': repr(module), 'rng_before': torch.cuda.get_rng_state().cpu()}
                self.pending[id(module)] = (record, payload)
            else:
                self.pending[id(module)] = (record, None)

        def post(self, name, module, inputs, output):
            if not self.active or args.leaves == 0:
                return
            record, payload = self.pending.pop(id(module))
            if payload is not None:
                record.update(output=tensor_info(output), gpu_rng_after=tensor_info(torch.cuda.get_rng_state()))
                payload.update(output=snapshot(output), state_after=snapshot(module.state_dict()))
                if record['stochastic']:
                    mask, origin = saved_mask(output)
                    record.update(mask_source=origin, probability=getattr(module, 'p', getattr(module, 'drop_prob', None)))
                    if mask is not None:
                        payload['mask'] = snapshot(mask)
                        record['mask'] = tensor_info(mask)
                        probability = record['probability']
                        # A consistency observation, never substitute this for output.
                        reconstructed = inputs[0] / (1 - probability) * mask
                        record['mask_reconstructs_output'] = bool(torch.allclose(
                            reconstructed, output, rtol=1e-4, atol=1e-5))
                record['capture'] = 'leaf-%04d.pt' % record['index']
                torch.save(payload, args.output / record['capture'])
            self.leaves.append(record)

        def after_backward(self, cur_iter, loss):
            if self.active:
                gradients = {name: p.grad.detach().cpu().clone()
                             for name, p in self.runner_ref().model.named_parameters() if p.grad is not None}
                torch.save(gradients, args.output / 'gradients-step1.pt')
                self.current['gradient_sha256'] = state_hash(gradients)
            self.current.update(loss=float(loss.detach()), gpu_rng_after=tensor_info(torch.cuda.get_rng_state()))
            self.steps.append(self.current)
            self.active = False
            write(args.output / 'observations.json', {
                'scope': 'Synchronized diagnostic prefix, not new full-budget accuracy or performance',
                'observer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'rtol': 1e-4, 'atol': 1e-5, 'steps': self.steps, 'leaves': self.leaves})

    train = BaseRunner.train

    def observed_train(runner):
        observer = Observe(runner)
        runner._hooks.hooks.append(observer)
        handles = []
        for name, module in runner.model.named_modules():
            if not any(module.children()):
                handles.append(module.register_forward_pre_hook(
                    lambda m, inputs, name=name: observer.pre(name, m, inputs)))
                handles.append(module.register_forward_hook(
                    lambda m, inputs, output, name=name: observer.post(name, m, inputs, output)))
        try:
            return train(runner)
        finally:
            for handle in handles:
                handle.remove()

    BaseRunner.train = observed_train
    args.mode, args.probe_steps, args.deterministic = 'train', args.steps, True
    run(args)


if __name__ == '__main__':
    main()
