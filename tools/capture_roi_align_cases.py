"""Capture first-seen RoIAlign backward shapes from the fixed UP model protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from compare_platform_models import worker
from diagnose_platform_operator import snapshot, signature
from run_platform_validation import source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--vendor', choices=('nvidia', 'metax'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.root, args.baseline, directory = args.root.resolve(), args.baseline.resolve(), args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    from up_platform.ops._native.loader import load_native
    extension = load_native(args.vendor)
    original = extension.roi_align.backward_avg_cuda
    records, captured = [], {}
    def observed(*values):
        key = (tuple(values[:5]), tuple(values[5].shape), tuple(values[7].shape), str(values[5].dtype))
        first = key not in captured
        payload = {'operation': 'native.roi_align.backward_avg_cuda', 'args': snapshot(values),
                   'kwargs': {}, 'input_signature': signature((values, {}))} if first else None
        result = original(*values)
        if first:
            path = directory/f'case-{len(captured):02d}.pt'
            payload['expected'] = snapshot(values[7])
            torch.save(payload, path)
            captured[key] = path
        path = captured[key]
        records.append({'call': len(records), 'case': path.name, 'first_seen': first,
                        'parameters': list(values[:5]), 'top_shape': list(values[5].shape),
                        'roi_shape': list(values[6].shape), 'bottom_shape': list(values[7].shape),
                        'dtype': str(values[5].dtype)})
        print(json.dumps(records[-1]), flush=True)
        return result
    args.model, args.variant, args.purpose = 'faster_rcnn', 'routed', 'training'
    args.deterministic, args.reference_defer_kitti = True, False
    args.output = directory/'model.json'
    extension.roi_align.backward_avg_cuda = observed
    try:
        worker(args)
    finally:
        extension.roi_align.backward_avg_cuda = original
        report = {'vendor': args.vendor, 'seed': 131, 'protocol': 'unchanged three-update UP compatibility worker',
                  'source': source_manifest(args.root), 'calls': records,
                  'binary_sha256': hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
                  'captures': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in captured.values()},
                  'environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
                      'MKL_NUM_THREADS', 'PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG',
                      'MCDNN_USE_DETERMINISTIC_ALGO', 'MCDNN_STABLE_RESULT_MOD')}}
        (directory/'cases.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
