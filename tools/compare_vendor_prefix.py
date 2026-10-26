"""Compare trusted tensor captures from observe_vendor_prefix.py on the CPU."""
import argparse
import json
from pathlib import Path

import torch


def compare_tensor(reference, actual):
    row = {'shape': list(reference.shape), 'dtype': str(reference.dtype)}
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        return dict(row, passed=False, exact=False, kind='metadata')
    exact = torch.equal(reference, actual)
    if reference.is_floating_point():
        left, right = reference.double(), actual.double()
        difference = (left - right).abs()
        close = torch.isclose(right, left, rtol=1e-4, atol=1e-5, equal_nan=False)
        row.update(passed=bool(close.all()), exact=exact,
                   mismatched_elements=int((reference != actual).sum()),
                   outside_tolerance=int((~close).sum()),
                   max_abs=float(difference.max()) if reference.numel() else 0.)
    else:
        count = int((reference != actual).sum())
        row.update(passed=exact, exact=exact, mismatched_elements=count, outside_tolerance=count)
    return row


def compare_tree(reference, actual, path=''):
    if isinstance(reference, torch.Tensor):
        return [{'path': path, **compare_tensor(reference, actual)}]
    if isinstance(reference, dict):
        if list(reference) != list(actual):
            raise ValueError('Tree keys differ at ' + path)
        return [row for key in reference for row in compare_tree(reference[key], actual[key], path + '/' + str(key))]
    if isinstance(reference, (tuple, list)):
        if len(reference) != len(actual):
            raise ValueError('Tree lengths differ at ' + path)
        return [row for i, (left, right) in enumerate(zip(reference, actual))
                for row in compare_tree(left, right, path + '/' + str(i))]
    return [{'path': path, 'exact': reference == actual, 'passed': reference == actual}]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--actual', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--leaves', type=int, help='Explicit capture prefix limit, not a complete-forward claim')
    p.add_argument('--gradients-only', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(2)
    left = json.loads((a.reference / 'observations.json').read_text())
    right = json.loads((a.actual / 'observations.json').read_text())
    rows = []
    pairs = [] if a.gradients_only else list(zip(left['leaves'], right['leaves']))
    if not a.gradients_only and len(left['leaves']) != len(right['leaves']):
        raise ValueError('Leaf execution sequence lengths differ')
    for l, r in pairs:
        if (l['index'], l['name'], l['type']) != (r['index'], r['name'], r['type']):
            raise ValueError('Leaf execution sequence differs')
        if a.leaves is not None and l['index'] >= a.leaves:
            continue
        if 'capture' not in l or 'capture' not in r:
            continue
        x = torch.load(a.reference / l['capture'], map_location='cpu', weights_only=False)
        y = torch.load(a.actual / r['capture'], map_location='cpu', weights_only=False)
        fields = ['inputs', 'output', 'state_before', 'state_after']
        if 'mask' in x and 'mask' in y:
            fields.append('mask')
        rows.append({'index': l['index'], 'name': l['name'], 'type': l['type'],
                     **{key: compare_tree(x[key], y[key]) for key in fields}})
    x = torch.load(a.reference / 'gradients-step1.pt', map_location='cpu', weights_only=True)
    y = torch.load(a.actual / 'gradients-step1.pt', map_location='cpu', weights_only=True)
    gradients = compare_tree(x, y)
    report = {'scope': 'Observed first training forward and parameter gradients; CPU comparisons only',
              'rtol': 1e-4, 'atol': 1e-5, 'reference': str(a.reference), 'actual': str(a.actual),
              'compared_leaf_limit': a.leaves, 'gradients_only': a.gradients_only,
              'first_step': {'reference': left['steps'][0], 'actual': right['steps'][0]},
              'leaves': rows, 'gradients': gradients,
              'gradient_tensors': len(gradients),
              'gradient_tensors_exact': sum(r['exact'] for r in gradients),
              'gradient_tensors_within_tolerance': sum(r['passed'] for r in gradients)}
    for label, test in (('first_bitwise_leaf', lambda r: not all(v['exact'] for v in r['output'])),
                        ('first_outside_tolerance_leaf', lambda r: not all(v['passed'] for v in r['output']))):
        report[label] = next((r['index'] for r in rows if test(r)), None)
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('first_bitwise_leaf', 'first_outside_tolerance_leaf',
          'gradient_tensors', 'gradient_tensors_exact', 'gradient_tensors_within_tolerance')}))


if __name__ == '__main__':
    main()
