"""Fixed-weight real-data inference and fixed-state gradient diagnostics.

Inputs are prepared once with the frozen UP recipe. No optimizer updates,
alternative model operators, CPU model fallback or relaxed tolerance are used.
"""
import argparse
import collections
import collections.abc
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import warnings

import numpy as np
import torch

from accuracy_data import digest, verify
from accuracy_recipe import recipe
from accuracy_train import batch_hash, metrics, state_hash
from compare_platform_models import compare, cpu, flatten
from package_native_repeat import tree_hash
from run_platform_validation import source_manifest

MODELS = ('resnet18', 'yolov5s', 'detr', 'vit_b16')
CONTROLS = ('CUBLAS_WORKSPACE_CONFIG', 'MCDNN_USE_DETERMINISTIC_ALGO', 'MCDNN_STABLE_RESULT_MOD',
            'PYTORCH_DISABLE_CUDA_CUDNN_TF32', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())


def policy(profile):
    expected = ':4096:8' if profile != 'native' else None
    for key in CONTROLS:
        if os.environ.get(key) != (expected if key == 'CUBLAS_WORKSPACE_CONFIG' else None):
            raise ValueError('Unexpected inherited control: '+key)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(profile != 'native', warn_only=profile == 'ultralytics')
    torch.backends.cudnn.deterministic = profile != 'native'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(131)
    np.random.seed(131)
    torch.manual_seed(131)
    torch.cuda.manual_seed_all(131)


def difference(left, right):
    result = compare(left, right)
    a, b = flatten(left), flatten(right)
    deltas = [float((x.double()-b[key].double()).abs().max()) for key, x in a.items()
              if isinstance(x, torch.Tensor) and key in b and isinstance(b[key], torch.Tensor)
              and x.shape == b[key].shape and x.dtype == b[key].dtype and x.numel()]
    return {'bitwise_equal': tree_hash(left) == tree_hash(right), 'within_tolerance': result['passed'],
            'max_abs_matching_tensors': max(deltas, default=0.), 'tensor_count': result['tensor_count'],
            'failures': result['failures'], 'rtol': 1e-4, 'atol': 1e-5}


def predictions(output):
    selected = {name: cpu(output[name]) for name in ('logits', 'scores', 'preds', 'dt_bboxes') if name in output}
    if not selected:
        raise ValueError('No declared model prediction tensors in UP output')
    return selected


def rng_state():
    return (torch.get_rng_state(), torch.cuda.get_rng_state(), np.random.get_state(), random.getstate())


def restore_rng(saved):
    torch.set_rng_state(saved[0])
    torch.cuda.set_rng_state(saved[1])
    np.random.set_state(saved[2])
    random.setstate(saved[3])


def build(args, *, training):
    collections.Iterable = collections.abc.Iterable
    np.float, np.int, np.bool = float, int, bool
    os.environ['DEFAULT_TASKS'] = 'cls:det'
    policy(args.profile)
    torch.cuda.set_device(0)
    sys.path.insert(0, str(args.source.resolve()))
    from up_platform import configure_platform
    platform = configure_platform({'backend': args.vendor, 'device': 'cuda:0'},
                                  purpose='training' if training else 'inference')
    import up
    if Path(up.__file__).resolve().parent != args.source.resolve()/'up':
        raise ValueError('Wrong UP source imported')
    from up.runner.base_runner import BaseRunner
    from up.utils.general.global_flag import DIST_BACKEND
    DIST_BACKEND.backend = 'dist'
    data = json.loads(args.data.read_text())
    config = recipe(args.source, data, args.model)
    algorithm_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config['runtime'].update(device='cuda:0', platform={'backend': args.vendor, 'device': 'cuda:0'})
    config['saver'].update(save_dir=str(args.output/'checkpoints'), results_dir=str(args.output/'predictions'))
    runner = BaseRunner(config, work_dir=str(args.output), training=training)
    policy(args.profile)
    return runner, platform, algorithm_sha


def prepare(args):
    plan = json.loads(args.checkpoint_plan.read_text())
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'models': {},
                'data_sha256': digest(args.data), 'data_verification': verify(json.loads(args.data.read_text())),
                'source': source_manifest(args.source), 'seed': 131}
    # One fresh process per model avoids leaking UP's legacy global flags.
    for model in (args.model,):
        selected = plan[model]
        checkpoint = Path(selected['path'])
        if digest(checkpoint) != selected['sha256']:
            raise ValueError('Training checkpoint changed: '+model)
        args.model, args.output = model, args.assets/'build'/model
        args.output.mkdir(parents=True)
        runner, _, config_sha = build(args, training=True)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
        if state_hash(state) != selected['state_sha256']:
            raise ValueError('Checkpoint model state changed: '+model)
        runner.model.load_state_dict(state, strict=True)
        directory = args.assets/model
        directory.mkdir()
        torch.save({'model': state}, directory/'weights.pt')
        validation = []
        iterator = iter(runner.data_loaders['test'])
        for _ in range(runner.data_loaders['test'].get_epoch_size()):
            validation.append(cpu(dict(next(iterator))))
        train_loader = runner.data_loaders['train']
        positions = {0, 1, train_loader.get_epoch_size()-1}
        training = []
        iterator = iter(train_loader)
        for index in range(train_loader.get_epoch_size()):
            batch = dict(next(iterator))
            if index in positions:
                training.append({'batch_index': index+1, 'batch': cpu(batch)})
        torch.save({'batches': validation}, directory/'validation.pt')
        torch.save({'cases': training}, directory/'training.pt')
        manifest['models'][model] = {'checkpoint_origin': selected, 'state_sha256': state_hash(state),
            'algorithm_config_sha256': config_sha, 'validation_batches': len(validation),
            'validation_samples': sum(len(b['image']) for b in validation),
            'training_case_shapes': [list(c['batch']['image'].shape) for c in training],
            'training_batch_positions': [c['batch_index'] for c in training],
            'validation_input_sha256': batch_hash(validation), 'training_input_sha256': batch_hash(training),
            'files': {name: digest(directory/name) for name in ('weights.pt', 'validation.pt', 'training.pt')}}
        del runner, state, validation, training
        torch.cuda.empty_cache()
    write(args.assets/args.model/'manifest.json', manifest)
    print(json.dumps({'prepared_models': list(manifest['models'])}), flush=True)


def inference(args, runner, platform, batches):
    from up.utils.general.hook_helper import Hook
    captured = []
    class Observe(Hook):
        def after_eval_forward(self, cur_iter, output):
            if not torch.is_inference_mode_enabled():
                raise ValueError('Expected actual inference_mode during UP inference')
            captured.append(predictions(output))
    runner._hooks.hooks.append(Observe(runner))
    if len(batches) != runner.data_loaders['test'].get_epoch_size():
        raise ValueError('Frozen validation budget does not match the UP recipe')
    iterator = iter(batches)
    def frozen_batch(split='test'):
        if split != 'test':
            raise ValueError('Only held-out inputs may be used for inference')
        cur_iter = runner.local_eval_iter(step=1)
        runner._hooks('before_data', cur_iter)
        return runner.batch2device(copy.deepcopy(next(iterator)))
    runner.get_batch = frozen_batch
    before = state_hash(runner.model.state_dict())
    with torch.inference_mode():
        path, results = runner.inference()
        actual = metrics(runner.data_loaders['test'].dataset.evaluate(path, results))
        baseline = captured[0]
        repeated, times = [], []
        for _ in range(args.trials):
            batch = runner.batch2device(copy.deepcopy(batches[0]))
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            value = runner.model(batch)
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end))
            repeated.append(difference(baseline, predictions(value)))
    if state_hash(runner.model.state_dict()) != before or platform.purpose != 'inference':
        raise ValueError('Inference updated model state or platform purpose')
    torch.save(captured, args.output/'outputs.pt')
    return {'metrics': actual, 'validation_batches': len(batches), 'input_sha256': batch_hash(batches),
        'repeat_first_batch': repeated, 'forward_batch_ms': times, 'model_state_unchanged': True,
        'timing_scope': 'Separate CUDA-event forward probes on the first fixed batch; full-validation CPU audits are not a throughput benchmark'}


def gradients(args, runner, platform, cases, initial):
    records = []
    runner.model.train()
    saved_rng = rng_state()
    for index, case in enumerate(cases):
        baseline, baseline_loss = None, None
        repeats = []
        for trial in range(args.trials):
            runner.model.load_state_dict(initial, strict=True)
            runner.model.zero_grad(set_to_none=True)
            batch = runner.batch2device(copy.deepcopy(case['batch']))
            restore_rng(saved_rng)
            output = runner.model(batch)
            loss = sum(value for key, value in output.items() if 'loss' in key)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            current = {name: cpu(p.grad) for name,p in runner.model.named_parameters() if p.grad is not None}
            if not all(torch.isfinite(value).all() for value in current.values()):
                raise ValueError('Non-finite training gradient')
            if baseline is None:
                baseline = current
                baseline_loss = float(loss.detach())
                torch.save(current, args.output/f'gradients-case{index}.pt')
            else:
                row = difference(baseline, current)
                row.update(trial=trial, loss=float(loss.detach()), loss_exact=float(loss.detach()) == baseline_loss)
                repeats.append(row)
                if not row['bitwise_equal'] and not (args.output/f'first-difference-case{index}.pt').exists():
                    torch.save(current, args.output/f'first-difference-case{index}.pt')
        records.append({'batch_position': case['batch_index'], 'shape': list(case['batch']['image'].shape),
                        'input_sha256': batch_hash(case['batch']), 'repeats': repeats, 'baseline_loss': baseline_loss,
                        'baseline_gradients_sha256': digest(args.output/f'gradients-case{index}.pt')})
        write(args.output/'progress.json', {'completed_cases': len(records), 'cases': records})
    runner.model.load_state_dict(initial, strict=True)
    restore_rng(saved_rng)
    if platform.purpose != 'training' or state_hash(runner.model.state_dict()) != state_hash(initial):
        raise ValueError('Gradient diagnostic changed its initial state or training purpose')
    return {'cases': records, 'optimizer_updates': 0,
            'scope': 'Actual UP losses/backward at fixed model state and RNG, including the classification epoch-tail batch; not full training accuracy or a speed benchmark'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'inference', 'gradients'), required=True)
    for name in ('source', 'data', 'assets', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--checkpoint-plan', type=Path)
    parser.add_argument('--vendor', choices=('nvidia', 'metax'), required=True)
    parser.add_argument('--profile', choices=('native', 'ultralytics', 'strict'), default='native')
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()
    if args.phase == 'prepare':
        args.assets.mkdir(parents=True, exist_ok=True)
        prepare(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'model': args.model, 'vendor': args.vendor, 'profile': args.profile, 'phase': args.phase,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'trials': args.trials, 'status': 'starting'}
    caught = []
    try:
        manifest = json.loads((args.assets/args.model/'manifest.json').read_text())
        entry = manifest['models'][args.model]
        for name, sha in entry['files'].items():
            if digest(args.assets/args.model/name) != sha:
                raise ValueError('Frozen weight/input file changed: '+name)
        runner, platform, config_sha = build(args, training=args.phase == 'gradients')
        if config_sha != entry['algorithm_config_sha256']:
            raise ValueError('UP recipe changed')
        initial = torch.load(args.assets/args.model/'weights.pt', map_location='cpu', weights_only=False)['model']
        runner.model.load_state_dict(initial, strict=True)
        if state_hash(runner.model.state_dict()) != entry['state_sha256']:
            raise ValueError('Common model state did not load exactly')
        policy(args.profile)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if args.phase == 'inference':
                data = torch.load(args.assets/args.model/'validation.pt', map_location='cpu', weights_only=False)
                value = inference(args, runner, platform, data['batches'])
            else:
                data = torch.load(args.assets/args.model/'training.pt', map_location='cpu', weights_only=False)
                value = gradients(args, runner, platform, data['cases'], initial)
        report.update(status='completed', result=value, weights_exact=True, model_state_sha256=entry['state_sha256'])
    except Exception as error:
        report.update(status='failed', exception_type=type(error).__name__, exception=str(error))
        (args.output/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        report.update(warnings=[str(item.message) for item in caught],
            policy={'deterministic': torch.are_deterministic_algorithms_enabled(),
                    'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
                    'cudnn_deterministic': torch.backends.cudnn.deterministic,
                    'cudnn_benchmark': torch.backends.cudnn.benchmark,
                    'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
                    'controls': {key: os.environ.get(key) for key in CONTROLS}},
            source=source_manifest(args.source), driver_sha256=digest(Path(__file__)),
            environment={'python': sys.version, 'packages': {name: metadata.version(name) for name in
                ('torch', 'torchvision', 'numpy', 'scipy', 'pycocotools', 'Pillow', 'opencv-python')},
                'device': torch.cuda.get_device_name(), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
        write(args.output/'report.json', report)


if __name__ == '__main__':
    main()
