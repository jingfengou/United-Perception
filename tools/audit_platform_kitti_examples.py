"""Reproduce original KITTI self-IoU failures and compare independent geometry."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from validate_platform_kitti import oracle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from up_platform import configure_platform
    p = configure_platform({'backend': 'nvidia', 'device': 'cuda:0'}, purpose='inference')
    source = args.baseline/'up/tasks/det_3d/data/metrics/kitti_object_eval_python/rotate_iou.py'
    spec = importlib.util.spec_from_file_location('pinned_original_kitti', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = np.load(args.reference)
    boxes, queries = fixture['boxes'], fixture['queries']
    records, arrays = [], {}
    for criterion in (-1, 0, 1, 2):
        original = module.rotate_iou_gpu_eval(boxes, queries, criterion)
        new = p.category('pointcloud').rotate_iou_eval(boxes, queries, criterion)
        expected = oracle(boxes, queries, criterion)
        arrays.update({str(criterion)+'_original': original, str(criterion)+'_new': new,
                       str(criterion)+'_oracle': expected})
        changed = np.abs(new-original) > 1e-5+1e-4*np.abs(original)
        row = {'criterion': criterion, 'changed_pairs': int(changed.sum()),
               'max_old_new': float(np.abs(new-original).max()),
               'max_new_oracle': float(np.abs(new-expected).max()),
               'new_within_original_tolerance': bool(np.allclose(new, expected, rtol=1e-4, atol=1e-5)),
               'old_vs_preserved_reference_max': float(np.abs(original-fixture[str(criterion)]).max())}
        records.append(row)
    examples = []
    for i in (4, 14):
        examples.append({'index': [i, i], 'boxes_xy_width_height_clockwise_angle': boxes[i].tolist(),
                         'query': queries[i].tolist(), 'same_box': bool(np.array_equal(boxes[i], queries[i])),
                         'old_iou': float(arrays['-1_original'][i, i]),
                         'new_iou': float(arrays['-1_new'][i, i]), 'oracle_iou': float(arrays['-1_oracle'][i, i])})
    np.savez_compressed(args.output/'values.npz', **arrays, boxes=boxes, queries=queries)
    report = {'interface': 'pointcloud.rotate_iou_eval', 'status': 'intentional_geometry_correction',
        'original_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        'scope': 'Original 7100 input pairs, four criteria; independent test-only GEOS oracle',
        'records': records, 'examples': examples}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
