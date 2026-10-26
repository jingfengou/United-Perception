"""Pack and compare an explicitly bounded forward prefix, retaining FP32 values."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from compare_vendor_prefix import compare_tree


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--reference', type=Path, help='Omit to pack the reference outputs')
    p.add_argument('--leaves', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    observed = json.loads((a.captures / 'observations.json').read_text())
    prefix = observed['leaves'][:a.leaves]
    if len(prefix) != a.leaves:
        raise ValueError('Requested prefix was not observed')
    if a.reference is None:
        records = []
        for row in prefix:
            capture = a.captures / row['capture']
            payload = torch.load(capture, map_location='cpu', weights_only=False)
            records.append({'metadata': row, 'output': payload['output'],
                            'capture_sha256': hashlib.sha256(capture.read_bytes()).hexdigest()})
        torch.save({'records': records, 'steps': observed['steps'], 'leaves': a.leaves}, a.output)
        print(json.dumps({'packed_leaves': len(records), 'bytes': a.output.stat().st_size}))
        return
    expected = torch.load(a.reference, map_location='cpu', weights_only=False)
    if expected['leaves'] != a.leaves:
        raise ValueError('Reference coverage differs')
    rows = []
    for left, right in zip(expected['records'], prefix):
        meta = left['metadata']
        if (meta['index'], meta['name'], meta['type']) != (right['index'], right['name'], right['type']):
            raise ValueError('Forward sequence differs')
        actual = torch.load(a.captures / right['capture'], map_location='cpu', weights_only=False)
        rows.append({'index': right['index'], 'name': right['name'],
                     'comparison': compare_tree(left['output'], actual['output'])})
    report = {'scope': 'Explicit first-leaf prefix, not whole-model equivalence', 'rtol': 1e-4, 'atol': 1e-5,
              'leaves': a.leaves, 'rows': rows,
              'first_outside_tolerance': next((row['index'] for row in rows
                                               if not all(v['passed'] for v in row['comparison'])), None)}
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'leaves': a.leaves, 'first_outside_tolerance': report['first_outside_tolerance']}))


if __name__ == '__main__':
    main()
