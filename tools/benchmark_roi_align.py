"""Compare trusted RoIAlign binaries on identical captures in fresh processes."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import torch

from compare_platform_models import compare
from diagnose_platform_operator import restore_inputs


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def load_operation(path):
    spec = importlib.util.spec_from_file_location(path.name.split('.')[0], path)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    return getattr(extension, 'roi_align', extension).backward_avg_cuda


def capture_paths(directory):
    paths = sorted(directory.glob('case-*.pt'))
    manifest = json.loads((directory/'cases.json').read_text())['captures']
    if {path.name for path in paths} != set(manifest):
        raise ValueError('Capture file set does not match cases.json')
    for path in paths:
        if digest(path) != manifest[path.name]:
            raise ValueError('Capture checksum mismatch: '+path.name)
    return paths


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    paths = capture_paths(args.cases)
    torch.use_deterministic_algorithms(True)
    operation = load_operation(args.binary)
    records, raw = [], {}
    for path in paths:
        payload = torch.load(path, weights_only=False)
        if payload['operation'] != 'native.roi_align.backward_avg_cuda':
            raise ValueError('Expected a trusted RoIAlign backward capture')
        values, kwargs = restore_inputs((payload['args'], payload['kwargs']), payload['input_signature'], 'cuda:0')
        initial = values[7].clone()
        initial_zero = not torch.count_nonzero(payload['args'][7]).item()
        def invoke():
            values[7].zero_() if initial_zero else values[7].copy_(initial)
            operation(*values, **kwargs)
        outputs = []
        for _ in range(3):
            invoke()
            outputs.append(values[7].detach().cpu().clone())
        large = len(values[6]) > 64
        iterations, warmup = (5, 3) if large else (100, 20)
        for _ in range(warmup):
            invoke()
        torch.cuda.synchronize()
        timings = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                invoke()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end)/iterations)
        raw[path.name] = outputs
        record = {'case': path.name, 'capture_sha256': digest(path), 'parameters': list(values[:5]),
                  'top_shape': list(values[5].shape), 'bottom_shape': list(values[7].shape),
                  'dtype': str(values[5].dtype), 'iterations': iterations, 'warmup': warmup,
                  'milliseconds': timings, 'median_ms': statistics.median(timings),
                  'bitwise_repeats': [torch.equal(outputs[0], value) for value in outputs[1:]]}
        records.append(record)
        print(json.dumps(record), flush=True)
    if not records:
        raise ValueError('No case-*.pt captures found')
    raw_path = args.output/'values.pt'
    torch.save(raw, raw_path)
    report = {'binary': str(args.binary), 'binary_sha256': digest(args.binary), 'raw_sha256': digest(raw_path),
              'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
              'controls': {key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED',
                   'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG',
                   'MCDNN_USE_DETERMINISTIC_ALGO', 'MCDNN_STABLE_RESULT_MOD')},
              'scope': 'Hot GPU backward including identical gradient-buffer reset, not end-to-end model throughput',
              'records': records}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-binary', type=Path)
    parser.add_argument('--candidate-binary', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--binary', type=Path)
    args = parser.parse_args()
    for key in ('cases', 'output', 'baseline_binary', 'candidate_binary', 'binary'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.worker:
        if not args.binary:
            parser.error('--worker requires --binary')
        worker(args)
        return
    if not args.baseline_binary or not args.candidate_binary:
        parser.error('Provide baseline and candidate binaries')
    args.output.mkdir(parents=True, exist_ok=False)
    reports, values = {}, {}
    for label, binary in (('baseline', args.baseline_binary), ('candidate', args.candidate_binary)):
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--binary', str(binary),
                   '--cases', str(args.cases), '--output', str(args.output/label)]
        with (args.output/(label+'.log')).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        reports[label] = json.loads((args.output/label/'summary.json').read_text())
        values[label] = torch.load(args.output/label/'values.pt', weights_only=False)
    records = []
    for left, right in zip(reports['baseline']['records'], reports['candidate']['records'], strict=True):
        assert left['capture_sha256'] == right['capture_sha256'] and left['case'] == right['case']
        result = compare(values['baseline'][left['case']], values['candidate'][right['case']])
        records.append({'case': left['case'], 'capture_sha256': left['capture_sha256'],
                        'comparison': result, 'bitwise_old_new': all(torch.equal(a, b) for a, b in
                            zip(values['baseline'][left['case']], values['candidate'][right['case']], strict=True)),
                        'baseline_ms': left['median_ms'], 'candidate_ms': right['median_ms'],
                        'speedup': left['median_ms']/right['median_ms'],
                        'baseline_repeatable': all(left['bitwise_repeats']), 'candidate_repeatable': all(right['bitwise_repeats'])})
    report = {'rtol': 1e-4, 'atol': 1e-5, 'reports': reports, 'records': records,
              'passed': all(row['comparison']['passed'] and row['baseline_repeatable'] and row['candidate_repeatable'] for row in records)}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': report['passed'], 'records': records}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
