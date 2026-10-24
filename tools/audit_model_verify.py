"""Audit model-verify's printed metrics without rounding away ratio-unit gaps."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


REFERENCE = '120d9af31fb022ccec71f1060eee892491341ca1'


def delta_pp(nvidia, metax, *, unit):
    if unit not in ('percent', 'ratio'):
        raise ValueError('Metric unit must be explicit: percent or ratio')
    return (metax-nvidia)*(100 if unit == 'ratio' else 1)


def audit(root):
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REFERENCE:
        raise ValueError('Use the pinned reference revision: '+REFERENCE)
    spec = importlib.util.spec_from_file_location('reference_summary', root/'generate_summary.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    def read(vendor, stem):
        return (root/vendor/'logs'/stem).read_text(encoding='utf-8')
    for model, stem in (('ResNet-50', 'resnet50-imagenet-1k-val'), ('ViT-B/16', 'VIT-B-16-imagenet-1k-val'),
                        ('CLIP ViT-B/32', 'clip-vit-b-32-imagenet-1k-val')):
        rows += module.classification_rows(model, 'ImageNet-1k val', read('cuda', stem+'-cuda.log'), read('mx-c500', stem+'-mx-c500.log'))
    rows += module.detr_rows(read('cuda', 'detr-resnet-50-coco-val-cuda.log'), read('mx-c500', 'detr-resnet-50-coco-val-mx-c500.log'))
    for size in ('n', 's', 'm', 'l', 'x'):
        rows += module.yolo_rows('YOLO26'+size, read('cuda', 'yolo26'+size+'-coco-val-cuda.log'),
                                 read('mx-c500', 'yolo26'+size+'-coco-val-mx-c500.log'))
    metrics = []
    for model, dataset, label, nvidia, metax, digits in rows:
        if not module.is_accuracy(label):
            continue
        if nvidia is None or metax is None:
            raise ValueError('A claimed reference metric is missing: '+model+'/'+label)
        unit = 'percent' if 'accuracy (%)' in label else 'ratio'
        metrics.append({'model': model, 'dataset': dataset, 'metric': label, 'unit': unit,
            'nvidia': nvidia, 'metax': metax, 'delta_pp': delta_pp(nvidia, metax, unit=unit),
            'reference_display': module.comparison(label, nvidia, metax), 'input_print_digits': digits,
            'scope': 'Numbers extracted from published logs; classification and DETR logs already round their outputs'})
    files = [root/'README.md', root/'generate_summary.py']
    files += sorted(p for folder in ('cuda', 'mx-c500', 'cuda_train', 'mx-c500_train')
                    for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.sh', '.md', '.log', '.csv'))
    return {'reference_url': 'https://github.com/xumy1995/model-verify', 'revision': revision,
        'metrics': metrics, 'files': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'interpretation': 'The inference table and independently trained checkpoints answer different questions; a rounded zero is not bitwise equivalence'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = audit(args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+'\n').encode())
    print(json.dumps({'metrics': len(value['metrics']), 'revision': value['revision']}))
