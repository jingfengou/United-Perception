"""Replay real YOLO duplicate-target writes with common operands, without UP."""
import argparse
import json
from pathlib import Path
import sys

import torch

from compare_vendor_prefix import compare_tensor
from observe_vendor_prefix import snapshot
from replay_vendor_leaf import restore


def restore_tree(value, info):
    if isinstance(value, torch.Tensor):
        return restore(value, info['stride']) if info['device'].startswith('cuda') else value.clone()
    if isinstance(value, dict):
        return {key: restore_tree(child, info[key]) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(restore_tree(child, meta) for child, meta in zip(value, info))
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=10)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    source = json.loads((a.reference / 'writes.json').read_text())
    cases = []
    for row in source['records']:
        if len(row['target_shape']) != 5 or row['target_shape'][0] != 64:
            continue
        captured = torch.load(a.reference / row['capture'], map_location='cpu', weights_only=False)
        args, _ = captured['arguments']
        target, indices, values = args[:3]
        if not all(isinstance(i, torch.Tensor) and i.dtype == torch.long for i in indices):
            continue
        broadcast = torch.broadcast_tensors(*indices)
        linear = torch.zeros_like(broadcast[0])
        for size, index in zip(target.shape, broadcast):
            linear = linear * size + index
        duplicates = linear.numel() - torch.unique(linear).numel()
        if not duplicates or torch.unique(values).numel() <= 1:
            continue
        namespace, operation, overload = captured['operation'].split('.')
        function = getattr(getattr(getattr(torch.ops, namespace), operation), overload)
        repeats, first = [], None
        for _ in range(a.repeats):
            call, keywords = restore_tree(captured['arguments'], captured['input_info'])
            with torch.no_grad():
                result = snapshot(function(*call, **keywords))
            if first is None:
                first = result
                torch.save(result, a.output / row['capture'])
            repeats.append(compare_tensor(first, result))
        cases.append({'index': row['index'], 'step': row['step'], 'operation': captured['operation'],
                      'target_shape': row['target_shape'], 'duplicate_indices': duplicates,
                      'reference_comparison': compare_tensor(captured['output'], first),
                      'self_repeats_exact': all(r['exact'] for r in repeats), 'repeats': repeats})
    if not cases:
        raise ValueError('No duplicate-index nonconstant YOLO targets found')
    report = {'scope': 'Real target writes on identical captured operands; no UP import',
              'torch': torch.__version__, 'device': torch.cuda.get_device_name(), 'rtol': 1e-4, 'atol': 1e-5,
              'cases': cases, 'up_imported': 'up' in sys.modules, 'up_platform_imported': 'up_platform' in sys.modules}
    (a.output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'cases': len(cases), 'all_reference_exact': all(r['reference_comparison']['exact'] for r in cases),
                      'all_self_repeats_exact': all(r['self_repeats_exact'] for r in cases)}))


if __name__ == '__main__':
    main()
