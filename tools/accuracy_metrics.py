"""Recompute saved final predictions on CPU; no repeat GPU inference or tolerance relaxation."""
import argparse
import json
import math
from pathlib import Path

from accuracy_data import digest


def recompute(model, prediction, meta):
    rows = [json.loads(line) for line in prediction.read_text().splitlines()]
    if model in ('resnet18', 'vit_b16'):
        import torch
        labels = {name: int(value) for name, value in (line.rsplit(' ', 1) for line in meta.read_text().splitlines())}
        names = [row['filename'] for row in rows]
        if len(names) != len(labels) or len(set(names)) != len(names) or set(names) != set(labels):
            raise ValueError('Incorrect held-out classification image set')
        counts = {1: 0, 5: 0}
        for row in rows:
            target = labels[row['filename']]
            if row['label'] != target or row['prediction'] != row['topk_idx'][0]:
                raise ValueError('Saved label/prediction contract differs')
            if len(row['topk_idx']) != 5 or len(set(row['topk_idx'])) != 5:
                raise ValueError('Invalid top5 output')
            for k in counts:
                counts[k] += int(target in row['topk_idx'][:k])
        return {'top'+str(k): torch.tensor([n], dtype=torch.float32).mul_(100./len(labels)).item() for k, n in counts.items()}
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    truth = COCO(str(meta))
    for row in rows:
        if row['image_id'] not in truth.imgs or row['category_id'] not in truth.cats:
            raise ValueError('Detection outside the held-out images/categories')
        if len(row['bbox']) != 4 or not all(math.isfinite(float(x)) for x in (*row['bbox'], row['score'])):
            raise ValueError('Invalid detection values')
    if rows:
        detected = truth.loadRes(rows)
    else:
        detected = COCO()
        detected.dataset = {'images': list(truth.imgs.values()), 'categories': list(truth.cats.values()), 'annotations': []}
        detected.createIndex()
    evaluator = COCOeval(truth, detected, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ('AP', 'AP.5', 'AP.75', 'AP_S', 'AP_M', 'AP_L', 'AR_1', 'AR_10', 'AR', 'AR_S', 'AR_M', 'AR_L')
    return {'bbox.'+name: float(value) for name, value in zip(names, evaluator.stats)}


def audit(directory, data):
    report = json.loads((directory/'report.json').read_text())
    meta = Path(data['cifar10' if report['model'] in ('resnet18', 'vit_b16') else 'pennfudan']['test']['meta'])
    prediction = directory/'final-predictions/results.txt'
    actual = recompute(report['model'], prediction, meta)
    if actual != report['final_metrics']:
        raise ValueError('CPU metric recomputation differs from final Runner metrics')
    result = {'passed': True, 'model': report['model'], 'metrics': actual, 'report_sha256': digest(directory/'report.json'),
              'predictions_sha256': digest(prediction), 'labels_sha256': digest(meta), 'evaluator_sha256': digest(Path(__file__))}
    (directory/'metric-audit.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    for directory in args.runs:
        print(json.dumps(audit(directory, data)))
