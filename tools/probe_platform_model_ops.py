"""Real-model platform-call capture, common-input replay and CPU comparisons.

Separate diagnostic processes only; do not attach this to accuracy training.
Capture preserves full per-rank batch, dtype, shapes and native policy. The
numeric cases are explicitly sampled, not an assertion about every variant.
"""
import argparse
import collections
import collections.abc
import importlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

from platform_operator_cases import ATOL, RTOL, PlatformCalls, compare_results, execute_case, load_cases


def write(path, record):
    path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')


def native_profile():
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def capture(args):
    import numpy as np
    collections.Iterable = collections.abc.Iterable
    np.float, np.int, np.bool = float, int, bool
    os.environ['DEFAULT_TASKS'] = 'cls:det'
    sys.path.insert(0, str(args.frozen_tools.resolve()))
    from accuracy_data import digest
    from accuracy_train import batch_hash, state_hash
    from full_accuracy_recipe import recipe
    from full_accuracy_train import iterator, unwrap
    from run_platform_validation import source_manifest
    plan = json.loads(args.plan.read_text())
    data = json.loads(args.data.read_text())
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    assert world == plan['world_size'] == 2, 'Use the original two-rank configuration'
    assert data['identity'] == plan['data_identity'] and digest(args.data) in plan['data_bindings_sha256']
    lane = 'new_'+args.vendor
    assert source_manifest(args.source)['sha256'] == plan['sources'][lane]['sha256']
    for name, sha in plan['tools'].items():
        assert digest(args.frozen_tools/name) == sha
    settings = plan['models'][args.model]
    assert digest(args.initial) == settings['initial_sha256']
    sys.path.insert(0, str(args.source.resolve()))
    from up_platform import configure_platform
    platform = configure_platform({'backend': args.vendor, 'device': f'cuda:{local_rank}'}, purpose=args.purpose)
    platform.device.set_device(local_rank)
    native_profile()
    dist.init_process_group('nccl')
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    observer = PlatformCalls(platform, args.output, capture=rank == 0 and not args.coverage_only,
                             cases_per_op=args.cases_per_op)
    phases = []
    try:
        with observer:
            import up
            assert Path(up.__file__).resolve().parent == args.source.resolve()/'up'
            from up.utils.general.global_flag import DIST_BACKEND
            from up.utils.env.gene_env import set_random_seed
            from up.runner.base_runner import BaseRunner
            from full_accuracy_data import register_full_coco
            DIST_BACKEND.backend = 'dist'
            register_full_coco()
            config = recipe(args.source, data, args.model, batch_size=settings['batch_size'],
                world_size=world, workers=plan['workers'], seed=plan['seed'])
            config['runtime'].update(platform={'backend': args.vendor, 'device': f'cuda:{local_rank}'},
                                     device=f'cuda:{local_rank}')
            config['saver'].update(save_dir=str(args.output/'unused-checkpoints'), results_dir=str(args.output/'unused-predictions'))
            runner = BaseRunner(config, work_dir=str(args.output), training=args.purpose == 'training')
            initial = torch.load(args.initial, map_location='cpu', weights_only=True)
            unwrap(runner.model).load_state_dict(initial, strict=True)
            if runner.ema:
                runner.ema.model.load_state_dict(initial, strict=True)
            assert state_hash(unwrap(runner.model).state_dict()) == settings['initial_state_sha256']
            del initial
            set_random_seed(plan['seed'], config['runtime'].get('rank_init', False))
            if args.purpose == 'training':
                iterator(runner, 'train', seed=plan['seed'], rank=rank, epoch=0)
                runner.model.train()
                batch = runner.get_batch('train')
                input_sha = batch_hash({key: value for key, value in batch.items() if key not in ('filenames', 'filename')})
                with observer.observing('training'):
                    loss = runner.forward_train(batch)
                    runner.backward(loss)
                phases.append({'phase': 'training', 'forward_backward_completed': True,
                    'batch_sha256': input_sha, 'optimizer_updates': 0})
                del batch, loss
                runner.model.zero_grad(set_to_none=True)
                before = state_hash(unwrap(runner.model).state_dict())
                ema_before = state_hash(runner.ema.model.state_dict()) if runner.ema else None
                modes = [module.training for module in runner.model.modules()]
                iterator(runner, 'test', seed=plan['seed'], rank=rank, epoch=0)
                batch = runner.get_batch('test')
                with platform.execution.validation(runner.model), observer.observing('training_eval'):
                    runner.forward_eval(batch)
                assert before == state_hash(unwrap(runner.model).state_dict())
                assert ema_before == (state_hash(runner.ema.model.state_dict()) if runner.ema else None)
                assert modes == [module.training for module in runner.model.modules()]
                phases.append({'phase': 'training_eval', 'model_ema_and_modes_preserved': True})
            else:
                iterator(runner, 'test', seed=plan['seed'], rank=rank, epoch=0)
                runner.model.eval()
                batch = runner.get_batch('test')
                with torch.no_grad(), observer.observing('independent_inference'):
                    runner.forward_eval(batch)
                phases.append({'phase': 'independent_inference', 'fresh_process': True})
            observer.completed = True
        if rank == 0:
            report = observer.report()
            report.update(model=args.model, phases=phases, parent_plan_sha256=digest(args.plan),
                source_sha256=source_manifest(args.source)['sha256'], initial_sha256=digest(args.initial),
                initial_state_sha256=settings['initial_state_sha256'], world_size=world,
                batch_size_per_rank=settings['batch_size'], native_profile={'benchmark': True,
                    'deterministic': False, 'tf32': False}, torch=torch.__version__,
                coverage_only=args.coverage_only, seed=plan['seed'],
                scope='one real training/test batch; no optimizer update and no dataset accuracy metric')
            write(args.output/'observation.json', report)
    except BaseException as error:
        if rank == 0:
            report = observer.report()
            report['error'] = repr(error)
            write(args.output/'observation-failed.json', report)
        raise
    finally:
        dist.destroy_process_group()


def replay(args):
    from accuracy_data import digest
    observed = load_cases(args.captures)
    sys.path.insert(0, str(args.source.resolve()))
    from up_platform import configure_platform
    platform = configure_platform({'backend': args.vendor, 'device': 'cuda:0'}, purpose=observed['purpose'])
    platform.device.set_device(0)
    native_profile()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in observed['cases']:
        path = args.captures/case['file']
        assert digest(path) == case['sha256'], 'Captured operands changed'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        first, first_meta = execute_case(platform, payload)
        filename = case['file'].replace('case-', 'result-')
        torch.save(first, args.output/filename)
        repeats = []
        for _ in range(args.repeats-1):
            again, _ = execute_case(platform, payload)
            repeats.append(compare_results(first, again))
            del again
        category, name = payload['operation'].split('.')
        try:
            shared = importlib.import_module('up_platform.ops.'+category+'.torch_impl')
        except ModuleNotFoundError as error:
            if error.name != 'up_platform.ops.'+category+'.torch_impl':
                raise
            shared = None
        direct = {'available': shared is not None and hasattr(shared, name)}
        if direct['available']:
            baseline, _ = execute_case(platform, payload, direct=True)
            direct['comparison'] = compare_results(baseline, first)
            if not direct['comparison']['passed']:
                torch.save(baseline, args.output/filename.replace('result-', 'direct-'))
            del baseline
        else:
            direct['reason'] = 'UP-specific operation; no torch_impl oracle. Cross-vendor result still compared.'
        rows.append({'operation': payload['operation'], 'phase': payload['phase'], 'input_case': case['file'],
            'input_sha256': case['sha256'], 'stochastic': payload['stochastic'],
            'result_file': filename, 'result_sha256': digest(args.output/filename),
            'same_vendor_repeats': repeats, 'platform_vs_direct_vendor_torch': direct, **first_meta})
        del first
        write(args.output/'progress.json', {'completed_cases': len(rows), 'expected_cases': len(observed['cases'])})
    write(args.output/'replay.json', {'completed': len(rows) == len(observed['cases']),
        'vendor': args.vendor, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
        'model': observed['model'], 'purpose': observed['purpose'], 'repeats': args.repeats,
        'rtol': RTOL, 'atol': ATOL, 'observation_sha256': digest(args.captures/'observation.json'),
        'common_capture_vendor': observed['vendor'], 'cases': rows,
        'scope': 'sampled semantic operations, forward/gradients/state; instrumented timings are not benchmarks',
        'rng_policy': 'same external seed; vendor RNG states never copied between platforms'})


def compare(args):
    from accuracy_data import digest
    left = json.loads((args.reference/'replay.json').read_text())
    right = json.loads((args.actual/'replay.json').read_text())
    if not left['completed'] or not right['completed'] or not left['cases']:
        raise ValueError('Incomplete replays cannot establish numerical coverage')
    assert left['observation_sha256'] == right['observation_sha256']
    assert len(left['cases']) == len(right['cases'])
    rows = []
    for a, b in zip(left['cases'], right['cases']):
        assert (a['operation'], a['phase'], a['input_sha256']) == (b['operation'], b['phase'], b['input_sha256'])
        for directory, entry in ((args.reference, a), (args.actual, b)):
            assert digest(directory/entry['result_file']) == entry['result_sha256']
        expected = torch.load(args.reference/a['result_file'], map_location='cpu', weights_only=False)
        actual = torch.load(args.actual/b['result_file'], map_location='cpu', weights_only=False)
        rows.append({'operation': a['operation'], 'phase': a['phase'], 'stochastic': a['stochastic'],
            'input_case': a['input_case'], 'comparison': compare_results(expected, actual)})
        del expected, actual
    write(args.output, {'scope': 'common operands/weights/upstream gradients, sampled real-model platform operations',
        'model': left['model'], 'purpose': left['purpose'], 'rtol': RTOL, 'atol': ATOL,
        'reference_vendor': left['vendor'], 'actual_vendor': right['vendor'],
        'numerical_cases': len(rows), 'cases_within_tolerance': sum(row['comparison']['passed'] for row in rows),
        'stochastic_cases': sum(row['stochastic'] for row in rows), 'cases': rows,
        'limitation': 'Equal seed does not imply equal cross-vendor random masks. No tolerance relaxation or forced mask replacement.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    c = commands.add_parser('capture')
    for name in ('source', 'frozen-tools', 'data', 'plan', 'initial', 'output'):
        c.add_argument('--'+name, type=Path, required=True)
    c.add_argument('--vendor', choices=('nvidia', 'metax'), required=True)
    c.add_argument('--model', choices=('resnet50', 'yolov5s', 'vit_b16'), required=True)
    c.add_argument('--purpose', choices=('training', 'inference'), required=True)
    c.add_argument('--coverage-only', action='store_true')
    c.add_argument('--cases-per-op', type=int, default=2)
    r = commands.add_parser('replay')
    for name in ('source', 'captures', 'output'):
        r.add_argument('--'+name, type=Path, required=True)
    r.add_argument('--vendor', choices=('nvidia', 'metax'), required=True)
    r.add_argument('--repeats', type=int, default=3)
    s = commands.add_parser('compare')
    for name in ('reference', 'actual', 'output'):
        s.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, 'cases_per_op', 1) < 1 or getattr(args, 'repeats', 2) < 2:
        parser.error('Positive case limit and at least two repeats are required')
    torch.set_num_threads(2)
    {'capture': capture, 'replay': replay, 'compare': compare}[args.command](args)


if __name__ == '__main__':
    main()
