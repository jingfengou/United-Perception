"""Run one lane on one allocated GPU; failures are preserved, metrics never gated."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from accuracy_data import digest, verify
from accuracy_recipe import BUDGETS
from run_platform_validation import source_manifest

CONTROLS = ('CUBLAS_WORKSPACE_CONFIG', 'MCDNN_USE_DETERMINISTIC_ALGO', 'MCDNN_STABLE_RESULT_MOD',
            'PYTORCH_DISABLE_CUDA_CUDNN_TF32', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'data', 'initial', 'output', 'lock'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--lane', choices=('original_nvidia', 'new_nvidia', 'new_metax'), required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--models', choices=BUDGETS, nargs='+', default=list(BUDGETS))
    parser.add_argument('--probe-steps', type=int)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    with args.lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = {'lane': args.lane, 'physical_device': args.device, 'seed': 131,
            'models': args.models, 'probe_steps': args.probe_steps, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'source': source_manifest(args.source), 'data_sha256': digest(args.data),
            'data_verification': verify(json.loads(args.data.read_text())), 'jobs': [],
            'selection': 'final epoch only; one seed; no accuracy threshold; no automatic reruns'}
        def save():
            (args.output/'execution.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
        save()
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device), OMP_NUM_THREADS='2',
            MKL_NUM_THREADS='2', PYTHONHASHSEED='0', DEFAULT_TASKS='cls:det')
        for key in CONTROLS:
            environment.pop(key, None)
        # Dependency paths may follow the selected source; it must take precedence.
        environment['PYTHONPATH'] = str(args.source)+os.pathsep+environment.get('PYTHONPATH', '')
        for model in args.models:
            initial = args.initial/model/'initial.pt'
            command = [sys.executable, str(Path(__file__).with_name('accuracy_train.py')), '--source', str(args.source),
                '--data', str(args.data), '--output', str(args.output/model), '--initial', str(initial),
                '--lane', args.lane, '--model', model]
            if args.probe_steps:
                command += ['--probe-steps', str(args.probe_steps)]
            job = {'model': model, 'command': command, 'initial_file_sha256': digest(initial), 'status': 'running'}
            manifest['jobs'].append(job)
            save()
            with (args.output/(model+'.log')).open('x') as log:
                result = subprocess.run(command, env=environment, cwd=args.source,
                                        stdout=log, stderr=subprocess.STDOUT)
            job.update(exit_code=result.returncode, status='failed' if result.returncode else 'executed')
            if not result.returncode:
                report_path = args.output/model/'report.json'
                report = json.loads(report_path.read_text())
                expected = args.probe_steps or BUDGETS[model]['steps']
                if (report['training_steps'] != expected or report['initial_file_sha256'] != job['initial_file_sha256']
                        or report['data_sha256'] != manifest['data_sha256']):
                    raise ValueError('Run does not match the frozen inputs/budget')
                job.update(status=report['status'], report_sha256=digest(report_path))
                audit_command = [sys.executable, str(Path(__file__).with_name('accuracy_metrics.py')),
                                 '--data', str(args.data), '--runs', str(args.output/model)]
                with (args.output/(model+'-metric-audit.log')).open('x') as log:
                    audited = subprocess.run(audit_command, env=dict(environment, CUDA_VISIBLE_DEVICES=''),
                                             stdout=log, stderr=subprocess.STDOUT)
                job.update(metric_audit_command=audit_command, metric_audit_exit_code=audited.returncode)
                infer_command = list(command)
                infer_command[infer_command.index('--initial')+1] = str(args.output/model/'final.pt')
                infer_command[infer_command.index('--output')+1] = str(args.output/model/'standalone-inference')
                infer_command += ['--mode', 'inference']
                with (args.output/(model+'-inference.log')).open('x') as log:
                    inferred = subprocess.run(infer_command, env=environment, cwd=args.source,
                                              stdout=log, stderr=subprocess.STDOUT)
                job.update(inference_command=infer_command, inference_exit_code=inferred.returncode)
                if inferred.returncode:
                    job['status'] = 'standalone_inference_failed'
            if source_manifest(args.source) != manifest['source'] or digest(args.data) != manifest['data_sha256']:
                raise ValueError('Source or data manifest changed during the cohort')
            save()
            print(args.lane, model, job['status'], flush=True)
        manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),
            completed=all(job['exit_code'] == 0 and job.get('inference_exit_code') == 0
                          and job.get('metric_audit_exit_code') == 0 for job in manifest['jobs']))
        save()
        raise SystemExit(0 if manifest['completed'] else 1)


if __name__ == '__main__':
    main()
