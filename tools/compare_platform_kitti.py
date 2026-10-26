"""Compare KITTI GPU evaluator with an unchanged NVIDIA Numba reference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from up_platform import configure_platform
from up_platform.ops._native.loader import load_native


def comparison_record(expected, actual):
    record = {'passed': False, 'expected_shape': list(expected.shape), 'actual_shape': list(actual.shape)}
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        record['reason'] = 'shape/dtype mismatch'
        return record
    bad = np.argwhere(~np.isclose(actual, expected, rtol=1e-4, atol=1e-5))
    record.update(passed=not len(bad), failed_pairs=len(bad),
                  max_abs=float(np.max(np.abs(expected-actual))) if actual.size else 0.,
                  examples=[{'index': [int(i), int(j)], 'expected': float(expected[i, j]),
                             'actual': float(actual[i, j])} for i, j in bad[:16]])
    return record


def self_overlap_sanity(boxes, queries, iou):
    pairs = [(i, j) for i, box in enumerate(boxes) for j, query in enumerate(queries)
             if np.array_equal(box, query) and np.isfinite(box).all() and (box[2:4] > 0).all()]
    bad = [(i, j) for i, j in pairs if not np.isclose(iou[i, j], 1., rtol=1e-4, atol=1e-5)]
    return {'scope': 'Diagnostic geometry invariant, not a replacement reference or relaxed gate',
            'identical_box_pairs': len(pairs), 'non_unit_self_iou': len(bad),
            'examples': [{'index': [i, j], 'box': boxes[i].tolist(), 'iou': float(iou[i, j])} for i, j in bad[:16]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--write-reference', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    p = configure_platform({'backend': args.vendor}, purpose='training')
    if args.write_reference:
        assert args.vendor == 'nvidia', 'The unmodified reference requires Numba CUDA'
        assert not args.reference.exists()
        from up_platform.ops.pointcloud._kitti_numba import rotate_iou_gpu_eval
        rng = np.random.default_rng(131)
        boxes = rng.uniform(-3, 3, size=(100, 5)).astype(np.float32)
        boxes[:, 2:4] = rng.uniform(.2, 4, (100, 2))
        queries = boxes[:71].copy()
        boxes[:4] = [[0, 0, 2, 2, 0], [2, 0, 2, 2, 0], [5, 5, 2, 2, 0], [0, 0, 1, 1, 0]]
        queries[:4] = boxes[:4]
        refs = {str(c): rotate_iou_gpu_eval(boxes, queries, c) for c in (-1, 0, 1, 2)}
        np.savez(args.reference, boxes=boxes, queries=queries, **refs)
    reference = np.load(args.reference)
    boxes, queries = reference['boxes'], reference['queries']
    records = []
    for criterion in (-1, 0, 1, 2):
        expected = reference[str(criterion)]
        try:
            routed = p.category('pointcloud').rotate_iou_eval(boxes, queries, criterion)
        except RuntimeError as error:
            routed = None
            records.append({'criterion': criterion, 'implementation': 'routed', 'passed': False, 'error': str(error)})
        port = load_native(args.vendor).kitti_rotate_iou(torch.tensor(boxes, device=p.device.device()),
                                                        torch.tensor(queries, device=p.device.device()), criterion).cpu().numpy()
        for name, actual in (('routed', routed), ('compiled_port', port)):
            if actual is None:
                continue
            records.append({'criterion': criterion, 'implementation': name, **comparison_record(expected, actual)})
    for n, k in ((0, 3), (3, 0), (0, 0)):
        from up_platform.ops.pointcloud._kitti_gpu import rotate_iou_eval
        value = rotate_iou_eval(boxes[:n], queries[:k], vendor=args.vendor)
        records.append({'empty': [n, k], 'passed': value.shape == (n, k) and value.dtype == np.float32})
    args.output.write_text(json.dumps({'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                                      'reference': 'Original upstream Numba CUDA, no numerical edits',
                                      'reference_self_overlap': self_overlap_sanity(boxes, queries, reference['-1']),
                                      'environment': p.device.environment(), 'rtol': 1e-4, 'atol': 1e-5,
                                      'records': records}, indent=2))
    print(json.dumps(records))
    raise SystemExit(0 if all(r['passed'] for r in records) else 1)


if __name__ == '__main__':
    main()
