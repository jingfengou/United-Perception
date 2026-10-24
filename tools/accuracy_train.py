"""One real UP Runner run: fixed budget, common initial tensors, native numerics."""
import argparse
import collections
import collections.abc
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import math
import numbers
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from accuracy_data import digest
from accuracy_recipe import BUDGETS, recipe
from compare_platform_models import cpu
from run_platform_validation import source_manifest

CONTROLS = ('CUBLAS_WORKSPACE_CONFIG', 'MCDNN_USE_DETERMINISTIC_ALGO', 'MCDNN_STABLE_RESULT_MOD',
            'PYTORCH_DISABLE_CUDA_CUDNN_TF32', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def state_hash(state):
    checksum = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        checksum.update(key.encode())
        checksum.update(str(tuple(tensor.shape)).encode())
        checksum.update(str(tensor.dtype).encode())
        checksum.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def batch_hash(value):
    def describe(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous().numpy()
        if isinstance(item, np.ndarray):
            return [str(item.dtype), list(item.shape), hashlib.sha256(item.tobytes()).hexdigest()]
        if isinstance(item, dict):
            return {str(k): describe(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [describe(child) for child in item]
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(type(item))
    return hashlib.sha256(json.dumps(describe(value), sort_keys=True).encode()).hexdigest()


def metrics(value):
    result = {str(key): float(child) for key, child in value.items() if isinstance(child, numbers.Real)}
    if not result or not all(math.isfinite(child) for child in result.values()):
        raise ValueError('Missing/non-finite validation metrics')
    return result


def finite(value):
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def run(args):
    args.source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write(args.output/'status.json', {'status': 'starting', 'lane': args.lane, 'model': args.model,
                                     'mode': args.mode, 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()})
    # Import-only Python/NumPy compatibility, identical in all three lanes.
    collections.Iterable = collections.abc.Iterable
    np.float, np.int, np.bool = float, int, bool
    os.environ['DEFAULT_TASKS'] = 'cls:det'
    for key in CONTROLS:
        if os.environ.get(key) is not None:
            raise ValueError('Native controller must remove inherited control: '+key)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    sys.path.insert(0, str(args.source))
    vendor = 'metax' if args.lane == 'new_metax' else 'nvidia'
    platform = None
    if args.lane != 'original_nvidia':
        from up_platform import configure_platform
        platform = configure_platform({'backend': vendor, 'device': 'cuda:0'}, purpose='inference' if args.mode == 'inference' else 'training')
        platform.device.set_device(0)
    import up
    if Path(up.__file__).resolve().parent != args.source/'up':
        raise RuntimeError('Wrong UP source tree imported')
    from up.utils.general.global_flag import DIST_BACKEND
    from up.utils.general.hook_helper import Hook
    from up.runner.base_runner import BaseRunner
    DIST_BACKEND.backend = 'dist'
    data = json.loads(args.data.read_text())
    config = recipe(args.source, data, args.model)
    algorithm_config = copy.deepcopy(config)
    algorithm_sha = hashlib.sha256(json.dumps(algorithm_config, sort_keys=True).encode()).hexdigest()
    if platform is not None:
        config['runtime']['platform'] = {'backend': vendor, 'device': 'cuda:0'}
        config['runtime']['device'] = 'cuda:0'
    config['saver'].update(save_dir=str(args.output/'checkpoints'), results_dir=str(args.output/'predictions'))
    write(args.output/'effective-config.json', config)
    runner = BaseRunner(config, work_dir=str(args.output), training=args.mode != 'inference')
    if args.mode != 'inference' and runner.max_iter != BUDGETS[args.model]['steps']:
        raise ValueError('Runner budget differs from the frozen plan: '+str(runner.max_iter))
    if torch.are_deterministic_algorithms_enabled() or torch.backends.cudnn.deterministic:
        raise ValueError('Native run unexpectedly enabled deterministic algorithms')
    if args.mode == 'inference':
        # Hash-pinned output of this task's own worker; UP schedulers retain NumPy scalars.
        saved = torch.load(args.initial, map_location='cpu', weights_only=False)
        runner.model.load_state_dict(saved['model'], strict=True)
        if state_hash(runner.model.state_dict()) != state_hash(saved['model']):
            raise ValueError('Standalone inference weights changed')
        with torch.no_grad():
            res_file, results = runner.inference()
            result = metrics(runner.data_loaders['test'].dataset.evaluate(res_file, results))
        write(args.output/'inference-report.json', {'status': 'completed', 'model': args.model, 'lane': args.lane,
            'purpose': platform.purpose if platform else 'original UP standalone inference', 'metrics': result,
            'checkpoint_sha256': digest(args.initial), 'weights_exact': True,
            'elapsed_seconds': time.monotonic()-started, 'source': source_manifest(args.source)})
        print(json.dumps({'mode': 'standalone-inference', 'model': args.model, 'metrics': result}), flush=True)
        return
    if args.mode == 'initialize':
        if args.lane != 'original_nvidia':
            raise ValueError('Generate common initialization once on the original NVIDIA model')
        torch.save(cpu(runner.model.state_dict()), args.output/'initial.pt')
        write(args.output/'initial.json', {'model': args.model, 'seed': 131, 'file_sha256': digest(args.output/'initial.pt'),
              'state_sha256': state_hash(runner.model.state_dict()), 'algorithm_config_sha256': algorithm_sha,
              'data_sha256': digest(args.data), 'source': source_manifest(args.source)})
        return
    if args.initial is None:
        raise ValueError('A common initial tensor file is required')
    initial = torch.load(args.initial, map_location='cpu', weights_only=True)
    runner.model.load_state_dict(initial, strict=True)
    initial_sha = state_hash(initial)
    if state_hash(runner.model.state_dict()) != initial_sha:
        raise ValueError('Common initialization was not loaded exactly')
    del initial

    class Recorder(Hook):
        def __init__(self, owner):
            super().__init__(owner)
            self.losses, self.learning_rates, self.evaluations, self.batches = [], [], [], []

        def before_forward(self, cur_iter, batch):
            self.learning_rates.append([float(g['lr']) for g in self.runner_ref().optimizer.param_groups])
            self.batches.append(batch_hash(batch))

        def after_backward(self, cur_iter, loss):
            value = float(loss.detach())
            if not math.isfinite(value):
                raise ValueError('Non-finite loss at step '+str(cur_iter))
            self.losses.append(value)
            if len(self.losses) == 1 or len(self.losses) % 50 == 0:
                write(args.output/'status.json', {'status': 'training', 'model': args.model, 'lane': args.lane,
                      'mode': args.mode, 'completed_steps': len(self.losses), 'expected_steps': runner.max_iter,
                      'loss': value, 'elapsed_seconds': time.monotonic()-started})

        def after_eval(self, value):
            self.evaluations.append({'after_steps': len(self.losses), 'metrics': metrics(value)})

    recorder = Recorder(runner)
    runner._hooks.hooks.append(recorder)
    if args.probe_steps:
        runner.max_iter = args.probe_steps
    initial_metrics = metrics(runner.evaluate())
    torch.cuda.reset_peak_memory_stats()
    train_started = time.monotonic()
    runner.train()
    torch.cuda.synchronize()
    train_seconds = time.monotonic()-train_started
    if len(recorder.losses) != runner.max_iter:
        raise ValueError('Incomplete optimizer budget')
    final_metrics = metrics(runner.evaluate())
    shutil.copytree(args.output/'predictions', args.output/'final-predictions')
    saved = cpu({'model': runner.model.state_dict(), 'optimizer': runner.optimizer.state_dict(),
                 'scheduler': runner.lr_scheduler.state_dict()})
    if not finite(saved):
        raise ValueError('Non-finite checkpoint state')
    torch.save(saved, args.output/'final.pt')
    restored = type(runner.model)(copy.deepcopy(config['net']))
    restored = platform.device.move(restored) if platform else restored.cuda()
    restored.load_state_dict(saved['model'], strict=True)
    if state_hash(restored.state_dict()) != state_hash(saved['model']):
        raise ValueError('Restored model state differs')
    runner.model = restored
    runner.optimizer = runner.build_optimizer(copy.deepcopy(config['trainer']['optimizer']))
    runner.optimizer.load_state_dict(saved['optimizer'])
    runner.lr_scheduler = runner.build_lr_scheduler(copy.deepcopy(config['trainer']['lr_scheduler']))
    runner.lr_scheduler.load_state_dict(saved['scheduler'])
    runner.optimizer.load_state_dict(saved['optimizer'])
    restored_metrics = metrics(runner.evaluate())
    packages = {name: metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'scipy',
                                                           'pycocotools', 'Pillow', 'opencv-python')}
    report = {'status': 'probe_completed' if args.probe_steps else 'completed', 'model': args.model,
        'lane': args.lane, 'seed': 131, 'mode': 'native', 'epochs': BUDGETS[args.model]['epochs'],
        'expected_steps': BUDGETS[args.model]['steps'], 'training_steps': len(recorder.losses),
        'initial_file_sha256': digest(args.initial), 'initial_state_sha256': initial_sha,
        'final_state_sha256': state_hash(saved['model']), 'checkpoint_sha256': digest(args.output/'final.pt'),
        'data_sha256': digest(args.data), 'algorithm_config_sha256': algorithm_sha,
        'source': source_manifest(args.source), 'driver_sha256': digest(Path(__file__)),
        'adapter_sha256': digest(Path(__file__).with_name('accuracy_detr.py')) if args.model == 'detr' else None,
        'losses': recorder.losses, 'learning_rates': recorder.learning_rates,
        'first_batch_sha256': recorder.batches[0], 'all_batches_sha256': hashlib.sha256(json.dumps(recorder.batches).encode()).hexdigest(),
        'initial_metrics': initial_metrics, 'epoch_metrics': recorder.evaluations,
        'final_metrics': final_metrics, 'restored_metrics': restored_metrics,
        'restored_metric_delta': {key: restored_metrics[key]-value for key, value in final_metrics.items()},
        'restore_weights_exact': True, 'checkpoint_all_states_finite': True,
        'training_wall_seconds': train_seconds, 'total_worker_seconds': time.monotonic()-started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'environment': {'python': sys.version, 'packages': packages, 'gpu': torch.cuda.get_device_name(),
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'deterministic': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'controls': {key: os.environ.get(key) for key in CONTROLS}},
        'scope': 'Complete fixed small-data budget; one seed; no statistical equivalence claim. Official DETR adapter is external to UP.',
        'selection': 'final epoch only; no best checkpoint or rerun selection',
        'model_provenance': 'official DETR adapter, unmodified official criterion' if args.model == 'detr' else 'native UP model'}
    write(args.output/'report.json', report)
    write(args.output/'status.json', {'status': report['status'], 'model': args.model, 'lane': args.lane,
          'completed_steps': len(recorder.losses), 'final_metrics': final_metrics})
    print(json.dumps({'status': report['status'], 'model': args.model, 'lane': args.lane,
                      'steps': report['training_steps'], 'metrics': final_metrics}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'data', 'output', 'initial'):
        parser.add_argument('--'+name, type=lambda value: Path(value).absolute(), required=name != 'initial')
    parser.add_argument('--lane', choices=('original_nvidia', 'new_nvidia', 'new_metax'), required=True)
    parser.add_argument('--model', choices=BUDGETS, required=True)
    parser.add_argument('--mode', choices=('initialize', 'train', 'inference'), default='train')
    parser.add_argument('--probe-steps', type=int)
    run(parser.parse_args())
