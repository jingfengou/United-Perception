"""Trace same-source GPU calls and replay the first changed ATen operation.

Diagnostic only: CPU copies record evidence, never replace model computation.
Use a fresh process for each trace. The existing model protocol and tolerances
are unchanged. Captures are local, trusted torch.save artifacts, not downloads.
"""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

from compare_platform_models import MODELS, compare, worker


def snapshot(value):
    def copy_tensor(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().clone()
        return tensor
    return tree_map(copy_tensor, value)


def signature(value):
    def describe(tensor):
        if not isinstance(tensor, torch.Tensor):
            return repr(tensor)
        # contiguous() can preserve a non-unit stride on a singleton view.
        data = torch.empty(tensor.numel(), dtype=tensor.dtype).copy_(tensor.detach().cpu().reshape(-1)).view(torch.uint8)
        return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                'stride': list(tensor.stride()), 'device': str(tensor.device),
                'sha256': hashlib.sha256(data.numpy().tobytes()).hexdigest()}
    return tree_map(describe, value)


class FirstDifference(Exception):
    pass


class OperatorTrace(TorchDispatchMode):
    def __init__(self, directory, reference=None, limit=1000, device_type='cuda'):
        super().__init__()
        self.directory = directory
        self.reference = reference
        self.entries = []
        self.expected = json.loads((reference/'trace.json').read_text())['entries'] if reference else None
        self.first = None
        self.first_bitwise = None
        self.limit = limit
        self.limited = False
        self.completed = False
        self.error = None
        self.device_type = device_type
        self.recording = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.recording:
            return func(*args, **kwargs)
        # CPU initialization and host evidence copies are not GPU kernels.
        leaves = tree_flatten((args, kwargs))[0]
        active = any(isinstance(t, torch.Tensor) and t.device.type == self.device_type for t in leaves)
        if not active or str(func) in ('aten.detach.default', 'aten._to_copy.default'):
            return func(*args, **kwargs)
        return self.observe(str(func), args, kwargs, lambda: func(*args, **kwargs))

    @contextmanager
    def native_boundaries(self, extension):
        """Observe logical outputs, excluding uninitialized NMS scratch tails."""
        contracts = (
            ('roi_align.forward_avg_cuda', tuple(range(7)), lambda a, r: a[7]),
            ('roi_align.backward_avg_cuda', tuple(range(8)), lambda a, r: a[7]),
            ('overlaps.iou', (0, 1, 3), lambda a, r: a[2]),
            ('naive_nms.gpu_nms', (2, 3, 4), lambda a, r: (a[0][:int(a[1][0])], a[1])),
        )
        saved = []
        try:
            for path, readonly, output in contracts:
                module_name, method = path.split('.')
                module = getattr(extension, module_name)
                original = getattr(module, method)
                def wrapped(*args, _original=original, _path=path, _readonly=readonly, _output=output, **kwargs):
                    return self.observe('native.'+_path, args, kwargs,
                                        lambda: _original(*args, **kwargs), _readonly, _output)
                saved.append((module, method, original))
                setattr(module, method, wrapped)
            yield
        finally:
            for module, method, original in reversed(saved):
                setattr(module, method, original)

    def observe(self, name, args, kwargs, call, readonly=None, output=None):
        self.recording = True
        try:
            return self._observe(name, args, kwargs, call, readonly, output)
        finally:
            self.recording = False

    def _observe(self, name, args, kwargs, call, readonly, output):
        index = len(self.entries)
        if index == self.limit:
            self.limited = True
            raise FirstDifference
        inputs = signature((args, kwargs))
        readonly_inputs = signature((tuple(args[i] for i in readonly), kwargs)) if readonly is not None else inputs
        saved = snapshot((args, kwargs)) if self.reference else None
        result = call()
        actual = snapshot(output(args, result) if output else result)
        entry = {'index': index, 'operation': name,
                 'input_sha256': hashlib.sha256(json.dumps(readonly_inputs, sort_keys=True).encode()).hexdigest()}
        self.entries.append(entry)
        filename = f'{index:05d}.pt'
        if self.reference is None:
            torch.save(actual, self.directory/filename)
            return result
        if index >= len(self.expected) or self.expected[index]['operation'] != name:
            self.first = {'index': index, 'operation': name, 'kind': 'call_sequence'}
            raise FirstDifference
        inputs_equal = entry['input_sha256'] == self.expected[index]['input_sha256']
        expected = torch.load(self.reference/filename, weights_only=False)
        comparison = compare(expected, actual)
        if self.first_bitwise is None and signature(expected) != signature(actual):
            self.first_bitwise = {'index': index, 'operation': name,
                                  'inputs_bitwise_equal': inputs_equal,
                                  'within_tolerance': comparison['passed']}
            torch.save({'operation': name, 'args': saved[0], 'kwargs': saved[1],
                        'input_signature': inputs, 'expected': expected, 'actual': actual},
                       self.directory/'first-bitwise.pt')
        if not comparison['passed']:
            self.first = {'index': index, 'operation': name, 'kind': 'numeric',
                          'inputs_bitwise_equal': inputs_equal, **comparison}
            torch.save({'operation': name, 'args': saved[0], 'kwargs': saved[1],
                        'input_signature': inputs, 'expected': expected, 'actual': actual},
                       self.directory/'first-difference.pt')
            raise FirstDifference
        return result

    def report(self):
        complete = self.completed and not self.limited and self.first is None and (self.expected is None or len(self.entries) == len(self.expected))
        return {'scope': 'GPU ATen trace; instrumented execution, not accuracy or performance',
                'rtol': 1e-4, 'atol': 1e-5, 'reference': str(self.reference) if self.reference else None,
                'limit': self.limit, 'limit_reached': self.limited,
                'execution_error': self.error,
                'matched_complete_trace': complete, 'first_difference': self.first,
                'first_bitwise_difference': self.first_bitwise, 'entries': self.entries}


def restore_inputs(value, descriptions, device):
    def restore(tensor, info):
        if not isinstance(tensor, torch.Tensor):
            return tensor
        target = device if info['device'].startswith('cuda') else 'cpu'
        result = torch.empty_strided(info['shape'], info['stride'], dtype=tensor.dtype, device=target)
        result.copy_(tensor)
        return result
    return tree_map(restore, value, descriptions)


def replay(args):
    payload = torch.load(args.capture, weights_only=False)
    native = payload['operation'].startswith('native.')
    if native and payload['operation'] in ('native.roi_align.forward_avg_cuda', 'native.roi_align.backward_avg_cuda'):
        if args.native_binary is None:
            raise ValueError('Native replay requires --native-binary from the matching vendor build')
        spec = importlib.util.spec_from_file_location(args.native_binary.name.split('.')[0], args.native_binary)
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        _, module, method = payload['operation'].split('.')
        operation = getattr(getattr(extension, module), method)
    elif payload['operation'] in ('aten.convolution.default', 'aten.convolution_backward.default',
                                  'aten.mm.default', 'aten.addmm.default', 'aten.bmm.default'):
        namespace, name, overload = payload['operation'].split('.')
        operation = getattr(getattr(getattr(torch.ops, namespace), name), overload)
    else:
        raise ValueError('Replay currently supports only pure convolution/matrix calls; '
                         'native RoIAlign supports its explicit output-buffer contract')
    torch.use_deterministic_algorithms(not args.replay_atomic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    values = restore_inputs((payload['args'], payload['kwargs']), payload['input_signature'], 'cuda:0')
    rng = torch.cuda.get_rng_state()
    outputs = []
    for _ in range(args.repeats):
        torch.cuda.set_rng_state(rng)
        call_args, kwargs = restore_inputs((payload['args'], payload['kwargs']), payload['input_signature'], 'cuda:0')
        value = operation(*call_args, **kwargs)
        outputs.append(snapshot(call_args[7] if native else value))
    comparisons = [compare(outputs[0], other) for other in outputs[1:]]
    report = {'scope': ('Explicit native RoIAlign replay' if native else 'ATen replay') + ' without importing UP or up_platform',
              'operation': payload['operation'], 'capture_sha256': hashlib.sha256(args.capture.read_bytes()).hexdigest(),
              'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
              'repeats': args.repeats, 'rtol': 1e-4, 'atol': 1e-5,
              'input_signature': signature(values), 'comparisons': comparisons,
              'bitwise_equal': [signature(outputs[0]) == signature(other) for other in outputs[1:]],
              'capture_reference_comparison': compare(payload['expected'], outputs[0]),
              'all_repeatable': all(item['passed'] for item in comparisons)}
    report['controls'] = controls()
    report['up_imported'] = 'up' in sys.modules
    report['up_platform_imported'] = 'up_platform' in sys.modules
    if native:
        report['native_binary_sha256'] = hashlib.sha256(args.native_binary.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('comparisons', 'input_signature')}))


def controls():
    return {'torch': torch.__version__, 'deterministic': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'environment': {key: os.environ.get(key) for key in (
                'CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED',
                'MCDNN_USE_DETERMINISTIC_ALGO', 'MCDNN_STABLE_RESULT_MOD', 'MCDNN_POOLING_MAX_DETERMINISTIC')},
            'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--vendor', choices=('nvidia', 'metax'))
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--variant', choices=('original', 'repeat', 'routed'), default='original')
    parser.add_argument('--purpose', choices=('training', 'inference'), default='training')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='Required for inference; use the same trusted original training checkpoint')
    parser.add_argument('--capture', type=Path, help='Replay an existing first-difference.pt with plain ATen')
    parser.add_argument('--native-ops', action='store_true', help='Also trace Faster R-CNN RoIAlign/overlap/NMS pybind boundaries')
    parser.add_argument('--native-binary', type=Path, help='Matching trusted native extension for isolated RoIAlign replay')
    parser.add_argument('--replay-atomic', action='store_true', help='Replay only: disable the deterministic selector to audit the retained atomic path')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--limit', type=int, default=1000, help='Maximum recorded GPU calls; reaching it is not a complete pass')
    args = parser.parse_args()
    for name in ('root', 'baseline', 'output', 'reference', 'capture', 'checkpoint', 'native_binary'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.repeats < 2 or args.limit < 1:
        parser.error('--repeats must be at least 2 and --limit must be positive')
    if args.replay_atomic and (not args.capture or not args.native_binary):
        parser.error('--replay-atomic requires --capture and --native-binary')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.capture:
        replay(args)
        return
    if not all((args.root, args.baseline, args.vendor, args.model)):
        parser.error('Tracing requires --root, --baseline, --vendor and --model')
    if args.purpose == 'inference' and args.checkpoint is None:
        parser.error('Inference requires --checkpoint from the original training run')
    args.output.mkdir(parents=True)
    if args.checkpoint:
        shutil.copyfile(args.checkpoint, args.output/f'{args.model}-original-training.checkpoint.pt')
    from run_platform_validation import source_manifest
    (args.output/'source.json').write_text(json.dumps(source_manifest(args.root), indent=2)+'\n')
    trace = OperatorTrace(args.output, args.reference, args.limit)
    args.deterministic = True
    args.reference_defer_kitti = args.model == 'pointpillar'
    directory = args.output
    args.output = directory/'worker.json'
    if args.native_ops:
        from up_platform.ops._native.loader import load_native
        boundary = trace.native_boundaries(load_native(args.vendor))
    else:
        boundary = nullcontext()
    try:
        with boundary, trace:
            worker(args)
        trace.completed = True
    except FirstDifference:
        pass
    except Exception as error:
        trace.error = f'{type(error).__name__}: {error}'
        raise
    finally:
        report = trace.report()
        report['controls'] = controls()
        report['model'] = args.model
        report['variant'] = args.variant
        report['command'] = sys.argv
        (directory/'trace.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'calls': len(trace.entries), 'first_difference': trace.first}))


if __name__ == '__main__':
    main()
