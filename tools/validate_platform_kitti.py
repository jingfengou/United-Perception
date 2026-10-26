"""Audit corrected GPU KITTI geometry against independent GEOS and analytic cases.

Shapely is a test-only dependency. It is never a runtime fallback for UP.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from up_platform import configure_platform
from compare_platform_kitti import comparison_record, self_overlap_sanity
from run_platform_validation import source_manifest


def oracle(boxes, queries, criterion):
    import shapely

    def polygons(values):
        values = values.astype(np.float64)
        local = np.array([[-.5, -.5], [-.5, .5], [.5, .5], [.5, -.5]])
        xy = values[:, None, 2:4] * local
        c, s = np.cos(values[:, 4, None]), np.sin(values[:, 4, None])
        corners = np.stack((c*xy[:, :, 0] + s*xy[:, :, 1],
                            -s*xy[:, :, 0] + c*xy[:, :, 1]), axis=-1)
        corners += values[:, None, :2]
        return shapely.polygons(corners)

    overlap = shapely.area(shapely.intersection(polygons(boxes)[:, None], polygons(queries)[None, :], grid_size=0))
    area1 = boxes[:, 2].astype(np.float64) * boxes[:, 3]
    area2 = queries[:, 2].astype(np.float64) * queries[:, 3]
    denominator = {0: area2[None, :], 1: area1[:, None]}.get(criterion)
    if criterion == -1:
        denominator = area1[:, None] + area2[None, :] - overlap
    if denominator is not None:
        overlap = np.divide(overlap, denominator, out=np.zeros_like(overlap), where=denominator > 0)
    return overlap.astype(np.float32)


def fixtures(reference):
    yield 'original_reference_inputs', reference['boxes'], reference['queries']
    edges = np.array([[0, 0, 2, 2, 0], [2, 0, 2, 2, 0], [2, 2, 2, 2, 0],
                      [1, 0, 2, 2, 0], [0, 0, 1, 1, 0], [0, 0, 0, 2, 0],
                      [0, 0, 2, 0, .3], [0, 0, 2, 2, .3], [0, 0, 2, 2, np.pi/4],
                      [0, 0, 2, 2, 1e-7], [2-1e-6, 0, 2, 2, 0], [2+1e-6, 0, 2, 2, 0]], np.float32)
    yield 'closed_edges_and_degenerate', edges, edges.copy()
    rng = np.random.default_rng(131)
    values = rng.uniform(-100, 100, (150, 5)).astype(np.float32)
    values[:, 2:4] = rng.uniform(.001, 80, (150, 2))
    values[:, 4] = rng.uniform(-np.pi, np.pi, 150)
    yield 'random_rotations', values, values[:125].copy()
    shifted = values.copy()
    shifted[:, :2] += np.float32(1e6)
    yield 'large_centers', shifted, shifted[:125].copy()
    thin = values.copy()
    thin[:, 2] = np.float32(.001)
    yield 'thin_rectangles', thin, thin[:125].copy()
    yield 'noncontiguous', values[::2], values[1::2]


def analytic_checks(actual):
    # Unit/axis-aligned cases independently fix denominator order and contact rules.
    checks = {(-1, 0, 0): 1., (-1, 0, 1): 0., (-1, 0, 2): 0.,
              (-1, 0, 3): 1/3, (-1, 0, 4): .25, (0, 0, 4): 1.,
              (1, 0, 4): .25, (2, 0, 4): 1., (-1, 0, 5): 0., (-1, 6, 6): 0.}
    return [{'criterion': c, 'index': [i, j], 'expected': expected,
             'actual': float(actual[c][i, j]),
             'passed': bool(np.isclose(actual[c][i, j], expected, rtol=1e-4, atol=1e-5))}
            for (c, i, j), expected in checks.items()]


def main():
    import shapely
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor', choices=('nvidia', 'metax'), required=True)
    parser.add_argument('--purpose', choices=('training', 'inference'), required=True)
    parser.add_argument('--reference', type=Path, required=True, help='Preserved original Numba .npz, never overwritten')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    p = configure_platform({'backend': args.vendor}, purpose=args.purpose)
    evaluate = p.category('pointcloud').rotate_iou_eval
    reference = np.load(args.reference)
    records, arrays, legacy_changes, analytic = [], {}, [], []
    for name, boxes, queries in fixtures(reference):
        actual = {}
        for criterion in (-1, 0, 1, 2):
            expected = oracle(boxes, queries, criterion)
            value = evaluate(boxes, queries, criterion)
            actual[criterion] = value
            repeats = [evaluate(boxes, queries, criterion) for _ in range(2)]
            reverse = evaluate(queries, boxes, {0: 1, 1: 0}.get(criterion, criterion)).T
            record = {'case': name, 'criterion': criterion,
                      'oracle': comparison_record(expected, value),
                      'symmetry': comparison_record(value, reverse),
                      'bitwise_repeats': [bool(np.array_equal(value, repeat)) for repeat in repeats]}
            record['passed'] = record['oracle']['passed'] and record['symmetry']['passed'] and all(record['bitwise_repeats'])
            records.append(record)
            arrays[f'{name}_{criterion}'] = value
            if name == 'original_reference_inputs':
                legacy_changes.append({'criterion': criterion, **comparison_record(reference[str(criterion)], value)})
        sanity = self_overlap_sanity(boxes, queries, actual[-1])
        records.append({'case': name, 'self_overlap': sanity, 'passed': sanity['non_unit_self_iou'] == 0})
        if name == 'closed_edges_and_degenerate':
            analytic = analytic_checks(actual)
    for n, k in ((0, 3), (3, 0), (0, 0)):
        value = evaluate(reference['boxes'][:n], reference['queries'][:k])
        records.append({'empty': [n, k], 'passed': value.shape == (n, k) and value.dtype == np.float32})
    values_path = args.output/'values.npz'
    np.savez_compressed(values_path, **arrays)
    source = source_manifest(Path(__file__).resolve().parents[1])
    (args.output/'source.json').write_text(json.dumps(source, indent=2)+'\n')
    from up_platform.ops._native.loader import load_native
    binary = Path(load_native(args.vendor).__file__)
    report = {'scope': 'Corrected GPU evaluation geometry, intentionally not upstream numerical equivalence',
              'vendor': args.vendor, 'purpose': p.purpose, 'rtol': 1e-4, 'atol': 1e-5,
              'environment': p.device.environment(), 'torch': torch.__version__,
              'oracle': {'shapely': shapely.__version__, 'geos': shapely.geos_version_string,
                         'precision': 'double, grid_size=0; test-only CPU reference'},
              'source_sha256': source['sha256'], 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              'values_sha256': hashlib.sha256(values_path.read_bytes()).hexdigest(),
              'legacy_changes': legacy_changes, 'records': records, 'analytic': analytic,
              'passed': all(r['passed'] for r in records + analytic)}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('vendor', 'purpose', 'passed', 'legacy_changes')}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
