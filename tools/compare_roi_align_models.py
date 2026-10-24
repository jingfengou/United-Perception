"""Compare pre/post platform roots with identical UP model/data/seed contracts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from compare_platform_models import compare, flatten
from run_platform_validation import source_manifest


def compared(left, right):
    result = compare(left, right)
    a, b = flatten(left), flatten(right)
    result['bitwise_equal'] = a.keys() == b.keys() and all(
        torch.equal(a[key], b[key]) if isinstance(a[key], torch.Tensor) and isinstance(b[key], torch.Tensor)
        else type(a[key]) is type(b[key]) and a[key] == b[key] for key in a)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--upstream-root', type=Path, required=True)
    parser.add_argument('--vendor', choices=('nvidia', 'metax'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for key in ('baseline_root', 'candidate_root', 'upstream_root', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    executions, paths, timings = [], {}, {}
    def run(label, root, purpose, checkpoint=None):
        directory = args.output/label
        directory.mkdir()
        if checkpoint:
            shutil.copyfile(checkpoint, directory/'faster_rcnn-original-training.checkpoint.pt')
        path = directory/'model.json'
        command = [sys.executable, str(root/'tools/compare_platform_models.py'), '--worker',
                   '--root', str(root), '--baseline', str(args.upstream_root), '--vendor', args.vendor,
                   '--model', 'faster_rcnn', '--variant', 'routed', '--purpose', purpose,
                   '--deterministic', '--output', str(path)]
        if purpose == 'training':
            command += ['--measure-time']
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(root) + os.pathsep + environment.get('PYTHONPATH', '')
        with (directory/'worker.log').open('x') as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        executions.append({'label': label, 'command': command, 'exit_code': result.returncode})
        (args.output/'commands.json').write_text(json.dumps(executions, indent=2)+'\n')
        print(label, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError('Worker failed; retain this output and select a new stage after repair')
        paths[label] = path
        timings[label] = json.loads(path.read_text()).get('timing')
        return path
    baseline = run('baseline-training', args.baseline_root, 'training')
    run('candidate-training', args.candidate_root, 'training')
    run('candidate-repeat', args.candidate_root, 'training')
    checkpoint = baseline.with_suffix('.checkpoint.pt')
    run('baseline-inference', args.baseline_root, 'inference', checkpoint)
    run('candidate-inference', args.candidate_root, 'inference', checkpoint)
    records = []
    for left, right in (('baseline-training', 'candidate-training'), ('candidate-training', 'candidate-repeat'),
                        ('baseline-inference', 'candidate-inference')):
        result = compared(torch.load(paths[left].with_suffix('.pt'), weights_only=False),
                          torch.load(paths[right].with_suffix('.pt'), weights_only=False))
        records.append({'comparison': [left, right], **result})
    report = {'vendor': args.vendor, 'scope': 'same seed 131, batch 2, three updates, eval each step, restore after step 2; synthetic inputs',
              'rtol': 1e-4, 'atol': 1e-5, 'common_inference_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'source': {'baseline': source_manifest(args.baseline_root), 'candidate': source_manifest(args.candidate_root)},
              'records': records, 'timings': timings, 'executions': executions,
              'artifacts': {label: {'metadata_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                    'values_sha256': hashlib.sha256(path.with_suffix('.pt').read_bytes()).hexdigest()}
                            for label, path in paths.items()},
              'passed': all(r['passed'] for r in records)}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': report['passed'], 'records': records, 'timings': timings}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
