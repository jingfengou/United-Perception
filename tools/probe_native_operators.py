"""Repeat ATen operations on identical inputs; keep the first real result.

This is a separate synchronization-heavy diagnostic, never the full training
worker. Captured inputs can be replayed in a process that never imports UP.
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
from torch.utils._pytree import tree_flatten, tree_map

PURE = {'aten.convolution_backward.default', 'aten.cudnn_batch_norm_backward.default',
        'aten.native_batch_norm_backward.default', 'aten.native_layer_norm_backward.default',
        'aten.mm.default', 'aten.bmm.default', 'aten.addmm.default',
        'aten._softmax_backward_data.default', 'aten._log_softmax_backward_data.default',
        'aten.upsample_nearest2d_backward.default', 'aten.max_pool2d_with_indices_backward.default',
        'aten.gelu_backward.default', 'aten.sum.dim_IntList', 'aten.sum.default',
        'aten.index_put.default', 'aten.scatter_add.default', 'aten.index_add.default'}
MUTATES_FIRST = {'aten._index_put_impl_.default', 'aten.index_put_.default',
                 'aten.scatter_add_.default', 'aten.index_add_.default'}


def input_hash(arguments):
    leaves, _ = tree_flatten(arguments)
    return [{'shape': list(v.shape), 'dtype': str(v.dtype), 'stride': list(v.stride()),
             'sha256': hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            for v in leaves if isinstance(v, torch.Tensor)]


def compare(a, b):
    aa, _ = tree_flatten(a)
    bb, _ = tree_flatten(b)
    rows = []
    for x, y in zip(aa, bb):
        if not isinstance(x, torch.Tensor):
            if x != y:
                raise ValueError('Non-tensor output changed')
            continue
        exact = torch.equal(x, y)
        delta = 0.0 if exact or not x.numel() else float((x.double()-y.double()).abs().max())
        rows.append({'shape': list(x.shape), 'dtype': str(x.dtype), 'bitwise_equal': exact,
                     'max_abs': delta, 'within_tolerance': torch.allclose(x, y, rtol=1e-4, atol=1e-5)})
    return rows


class RepeatPure(TorchDispatchMode):
    def __init__(self, output):
        super().__init__()
        self.output, self.counts, self.findings = output, {}, []
        self.saved = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        before = args[0].clone() if name in MUTATES_FIRST and name not in self.saved else None
        result = func(*args, **kwargs)
        self.counts[name] = self.counts.get(name, 0)+1
        if name in PURE | MUTATES_FIRST and name not in self.saved:
            repeat_args = (before.clone(), *args[1:]) if before is not None else args
            again = func(*repeat_args, **kwargs)
            differences = compare(result, again)
            if any(not row['bitwise_equal'] for row in differences):
                capture = self.output/('capture-'+str(len(self.findings))+'.pt')
                original_args = (before, *args[1:]) if before is not None else args
                arguments = tree_map(lambda v: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v,
                                     (original_args, kwargs))
                torch.save({'operator': name, 'arguments': arguments, 'mutates_first': name in MUTATES_FIRST}, capture)
                self.findings.append({'operator': name, 'occurrence': self.counts[name],
                                      'capture': capture.name, 'outputs': differences,
                                      'schema': str(func._schema)})
                self.saved.add(name)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    for name in ('source', 'frozen-tools', 'data', 'initial'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--lane', choices=('original_nvidia', 'new_nvidia'))
    parser.add_argument('--model', choices=('yolov5s', 'vit_b16'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    if args.capture:
        if args.output.exists():
            raise FileExistsError(args.output)
        captured = torch.load(args.capture, map_location='cpu', weights_only=True)
        raw_args, kwargs = tree_map(lambda v: v.to('cuda:0') if isinstance(v, torch.Tensor) else v,
                                   captured['arguments'])
        namespace, operation, overload = captured['operator'].split('.')
        func = getattr(getattr(getattr(torch.ops, namespace), operation), overload)
        input_sha = input_hash((raw_args, kwargs))
        mutates = captured['mutates_first']
        def trial():
            call = (raw_args[0].clone(), *raw_args[1:]) if mutates else raw_args
            if input_hash((call, kwargs)) != input_sha:
                raise ValueError('Replay arguments changed')
            return func(*call, **kwargs)
        with torch.no_grad():
            first = trial()
            repeats = [compare(first, trial()) for _ in range(args.trials-1)]
        assert input_hash((raw_args, kwargs)) == input_sha, 'Input state changed across replays'
        report = {'operator': captured['operator'], 'deterministic': args.deterministic,
                  'trials': args.trials, 'repeats': repeats, 'rtol': 1e-4, 'atol': 1e-5,
                  'up_imported': 'up' in sys.modules, 'up_platform_imported': 'up_platform' in sys.modules,
                  'capture_sha256': hashlib.sha256(args.capture.read_bytes()).hexdigest(),
                  'input_tensor_hashes': input_sha, 'mutates_first': mutates,
                  'reset_mutable_target_each_trial': mutates}
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
        print(json.dumps({'operator': report['operator'], 'up_imported': report['up_imported'],
            'deterministic': args.deterministic,
            'nonidentical_repeats': sum(any(not row['bitwise_equal'] for row in repeat) for repeat in repeats)}))
        return
    if args.deterministic:
        raise ValueError('Capture the actual native operation before separate controlled replay')
    sys.path.insert(0, str(args.frozen_tools))
    from accuracy_train import run
    collections.Iterable = collections.abc.Iterable
    np.float, np.int, np.bool = float, int, bool
    sys.path.insert(0, str(args.source.resolve()))
    if args.lane == 'new_nvidia':
        from up_platform import configure_platform
        configure_platform({'backend': 'nvidia', 'device': 'cuda:0'}, purpose='training')
    from up.runner.base_runner import BaseRunner
    backward = BaseRunner.backward
    probe = RepeatPure(args.output)

    def observed_backward(runner, loss):
        with probe:
            return backward(runner, loss)

    BaseRunner.backward = observed_backward
    args.mode, args.probe_steps = 'train', args.steps
    run(args)
    report = {'operator_counts': probe.counts, 'findings': probe.findings,
              'scope': 'Repeated pure operations on identical arguments; original first result returned unchanged',
              'observer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'operator-probe.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'findings': probe.findings}, indent=2))


if __name__ == '__main__':
    main()
