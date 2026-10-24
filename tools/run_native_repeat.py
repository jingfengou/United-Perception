"""Same physical NVIDIA GPU: repeated native traces, full budgets and inference.

An outer GPU lock is held for the entire experiment. The original accuracy
worker and original/new runtime trees stay immutable; no retries or seed search.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

CONTROLS = ('CUBLAS_WORKSPACE_CONFIG', 'MCDNN_USE_DETERMINISTIC_ALGO', 'MCDNN_STABLE_RESULT_MOD',
            'PYTORCH_DISABLE_CUDA_CUDNN_TF32', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')
UUID = 'GPU-a11f730c-a975-2bf0-b765-3c24d83b7104'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    root, base = args.output.resolve(), args.base.resolve()
    if args.launch:
        receipt = root/'launch.json'
        if receipt.exists():
            raise FileExistsError(receipt)
        with (root/'controller.log').open('x') as stream:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--base', str(base),
                                        '--output', str(root)], stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        save(receipt, {'pid': process.pid, 'created_utc': datetime.now(timezone.utc).isoformat(),
                       'physical_device': 1, 'uuid': UUID})
        print(receipt.read_text())
        return
    sys.path.insert(0, str(base/'source-03/tools'))
    from accuracy_data import digest, verify
    from run_platform_validation import source_manifest
    plan = json.loads((base/'plan.json').read_text())
    with Path('/data2/oujingfeng/up-rearchitecture-gpu-1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        uuid = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        compute = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if uuid != UUID or compute:
            raise RuntimeError('GPU changed or has an active compute owner')
        assert source_manifest(base/'source-03')['sha256'] == plan['new_source_sha256']
        assert source_manifest(base/'original')['sha256'] == plan['original_source_sha256']
        assert digest(base/'data/data.json') == plan['data_sha256']
        verification = verify(json.loads((base/'data/data.json').read_text()))
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                           PYTHONHASHSEED='0', DEFAULT_TASKS='cls:det')
        for key in CONTROLS:
            environment.pop(key, None)
        state = {'pid': os.getpid(), 'physical_device': 1, 'uuid': uuid, 'seed': 131,
                 'started_utc': datetime.now(timezone.utc).isoformat(), 'mode': 'native',
                 'source_stage': str(base/'source-03'), 'source_sha256': plan['new_source_sha256'],
                 'original_sha256': plan['original_source_sha256'], 'data_sha256': plan['data_sha256'],
                 'data_verification': verification, 'commands': [], 'completed': False,
                 'observer_sha256': digest(root/'trace_native_training.py'),
                 'controller_sha256': digest(Path(__file__)),
                 'full_runs': 8, 'full_optimizer_updates': 14960,
                 'scope': 'Two repeats per source/model; not a multi-seed experiment or statistical equivalence proof'}
        def run(label, command):
            row = {'label': label, 'command': list(map(str, command)), 'status': 'running',
                   'started_utc': datetime.now(timezone.utc).isoformat()}
            state['commands'].append(row)
            save(root/'controller.json', state)
            with (root/(label+'.log')).open('x') as stream:
                result = subprocess.run(list(map(str, command)), env=environment, cwd=root,
                                        stdout=stream, stderr=subprocess.STDOUT)
            row.update(exit_code=result.returncode, status='completed' if result.returncode == 0 else 'failed',
                       finished_utc=datetime.now(timezone.utc).isoformat())
            assert source_manifest(base/'source-03')['sha256'] == plan['new_source_sha256']
            assert source_manifest(base/'original')['sha256'] == plan['original_source_sha256']
            save(root/'controller.json', state)
            print(label, result.returncode, flush=True)
        order = ('original-01', 'new-01', 'new-02', 'original-02')
        for model in ('yolov5s', 'vit_b16'):
            for variant in order:
                lane = 'original_nvidia' if variant.startswith('original') else 'new_nvidia'
                source = base/('original' if lane == 'original_nvidia' else 'source-03')
                run('trace-'+model+'-'+variant, [sys.executable, root/'trace_native_training.py', '--source', source,
                    '--frozen-tools', base/'source-03/tools', '--data', base/'data/data.json', '--initial', base/'initial'/model/'initial.pt',
                    '--output', root/'traces'/model/variant, '--lane', lane, '--model', model])
        for model in ('vit_b16', 'yolov5s'):
            for variant in order:
                lane = 'original_nvidia' if variant.startswith('original') else 'new_nvidia'
                source = base/('original' if lane == 'original_nvidia' else 'source-03')
                run('full-'+model+'-'+variant, [sys.executable, base/'source-03/tools/run_accuracy_cohort.py', '--source', source,
                    '--data', base/'data/data.json', '--initial', base/'initial', '--output', root/'full'/model/variant,
                    '--lock', root/'worker.lock', '--lane', lane, '--device', '1', '--models', model])
        for model in ('yolov5s', 'vit_b16'):
            for trained, runtime in (('original', 'new'), ('new', 'original')):
                checkpoint = root/'full'/model/(trained+'-01')/model/'final.pt'
                if not checkpoint.exists():
                    state['commands'].append({'label': 'cross-'+model+'-'+trained+'-to-'+runtime,
                                               'status': 'blocked_missing_checkpoint'})
                    continue
                lane = 'original_nvidia' if runtime == 'original' else 'new_nvidia'
                source = base/('original' if runtime == 'original' else 'source-03')
                run('cross-'+model+'-'+trained+'-to-'+runtime, [sys.executable, base/'source-03/tools/accuracy_train.py',
                    '--source', source, '--data', base/'data/data.json', '--initial', checkpoint,
                    '--output', root/'cross-inference'/model/(trained+'-to-'+runtime), '--lane', lane, '--model', model,
                    '--mode', 'inference'])
        state.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                     completed=all(row['status'] == 'completed' for row in state['commands']),
                     post_training_data=verify(json.loads((base/'data/data.json').read_text())))
        save(root/'controller.json', state)
        raise SystemExit(0 if state['completed'] else 1)


if __name__ == '__main__':
    main()
