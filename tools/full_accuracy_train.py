"""Resumable full-data UP comparison; run only on allocated devices.

The same driver calls native UP forward/backward/update/EMA/LR in every lane.
No model operation or vendor numerical kernel is replaced. Data workers use
their own per-epoch RNG; model RNG and all training states survive a restart.
Preflight outputs are explicitly separated from full-budget accuracy evidence.
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

import numpy as np
import torch
import torch.distributed as dist

from accuracy_data import digest
from accuracy_train import batch_hash, configure_numerics, finite, metrics, state_hash
from compare_platform_models import cpu
from full_accuracy_recipe import RECIPES, recipe
from run_platform_validation import source_manifest


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def unwrap(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch already initializes its CPU RNG from the full worker seed.
    import cv2
    cv2.setNumThreads(1)


def local_state(runner):
    return cpu({'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(), 'torch_gpu': torch.cuda.get_rng_state(),
        'buffers': dict(unwrap(runner.model).named_buffers()),
        'ema_buffers': dict(runner.ema.model.named_buffers()) if runner.ema else {},
        'scaler': runner.scaler.state_dict() if runner.fp16 else None})


def restore_buffers(runner, state):
    with torch.no_grad():
        for name, value in unwrap(runner.model).named_buffers():
            value.copy_(state['buffers'][name])
        if runner.ema:
            for name, value in runner.ema.model.named_buffers():
                value.copy_(state['ema_buffers'][name])


def restore_local(runner, state):
    restore_buffers(runner, state)
    if runner.fp16:
        runner.scaler.load_state_dict(state['scaler'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state(state['torch_gpu'])


def iterator(runner, split, *, seed, rank, epoch, skip=0):
    loader = runner.data_loaders[split]
    loader.generator = torch.Generator().manual_seed(seed+1000003*epoch+1009*rank+(0 if split == 'train' else 500000001))
    loader.worker_init_fn = worker_seed
    loader.batch_sampler.set_start_iter(epoch*loader.get_epoch_size())
    if hasattr(runner, 'data_iterators'):
        runner.data_iterators.pop(split, None)
    else:
        runner.data_iterators = {}
    stream = iter(loader)
    for _ in range(skip):
        next(stream)
    runner.data_iterators[split] = stream


def canonical_config(config, data, source):
    replacements = {str(source): '@UP@'}
    def paths(value, key='data'):
        if isinstance(value, dict):
            for name, child in value.items():
                paths(child, key+'.'+name)
        elif isinstance(value, str) and value.startswith('/'):
            replacements[value] = '@'+key+'@'
    paths(data)
    text = json.dumps(config, sort_keys=True)
    for path in sorted(replacements, key=len, reverse=True):
        text = text.replace(path, replacements[path])
    return hashlib.sha256(text.encode()).hexdigest()


def checkpoint(runner, directory, progress, rank, world, identity):
    """Two slots: publish a manifest only once every rank's files are complete."""
    slot = progress['checkpoint_generation'] % 2
    rank_path = directory/f'checkpoint-{slot}.rank{rank}.pt'
    temporary = rank_path.with_suffix('.pt.writing')
    torch.save(local_state(runner), temporary)
    temporary.replace(rank_path)
    if rank == 0:
        path = directory/f'checkpoint-{slot}.pt'
        temporary = path.with_suffix('.pt.writing')
        saved = cpu({'model': unwrap(runner.model).state_dict(), 'optimizer': runner.optimizer.state_dict(),
            'scheduler': runner.lr_scheduler.state_dict(), 'ema': runner.ema.state_dict() if runner.ema else None,
            'progress': progress, 'identity': identity})
        if not finite(saved):
            raise ValueError('Non-finite model/optimizer/EMA checkpoint state')
        torch.save(saved, temporary)
        temporary.replace(path)
    if world > 1:
        dist.barrier()
    if rank == 0:
        files = [directory/f'checkpoint-{slot}.pt',
                 *(directory/f'checkpoint-{slot}.rank{index}.pt' for index in range(world))]
        write(directory/'checkpoint.json', {'slot': slot, 'progress': progress, 'identity': identity,
              'files': {path.name: digest(path) for path in files}})
    if world > 1:
        dist.barrier()


def restore_checkpoint(runner, directory, rank, identity):
    marker = json.loads((directory/'checkpoint.json').read_text())
    if marker['identity'] != identity:
        raise ValueError('Checkpoint source/plan/data/lane does not match this run')
    for name, expected in marker['files'].items():
        if digest(directory/name) != expected:
            raise ValueError('Checkpoint hash differs: '+name)
    saved = torch.load(directory/f"checkpoint-{marker['slot']}.pt", map_location='cpu', weights_only=False)
    unwrap(runner.model).load_state_dict(saved['model'], strict=True)
    runner.optimizer.load_state_dict(saved['optimizer'])
    runner.lr_scheduler.load_state_dict(saved['scheduler'])
    if runner.ema:
        runner.ema.load_state_dict(saved['ema'])
    state = torch.load(directory/f"checkpoint-{marker['slot']}.rank{rank}.pt", map_location='cpu', weights_only=False)
    restore_local(runner, state)
    if runner.cur_iter != saved['progress']['completed_steps']:
        raise ValueError('Scheduler and completed-step counts differ')
    return saved['progress'], state


def load_inference_checkpoint(runner, directory, identity, rank=0):
    marker = json.loads((directory/'checkpoint.json').read_text())
    if marker['identity'] != identity:
        raise ValueError('Inference source/plan/data/lane differs from training')
    path = directory/f"checkpoint-{marker['slot']}.pt"
    if digest(path) != marker['files'][path.name]:
        raise ValueError('Inference checkpoint hash differs')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    unwrap(runner.model).load_state_dict(saved['model'], strict=True)
    if state_hash(unwrap(runner.model).state_dict()) != state_hash(saved['model']):
        raise ValueError('Inference model was not restored exactly')
    if runner.ema:
        runner.ema.load_state_dict(saved['ema'])
        if state_hash(runner.ema.model.state_dict()) != state_hash(saved['ema']['ema_state_dict']):
            raise ValueError('Inference EMA was not restored exactly')
    # UP DDP deliberately uses broadcast_buffers=False. Native training eval
    # therefore uses rank-local BN/EMA buffers; preserve that exact protocol in
    # the fresh distributed inference process as well as in training resume.
    local_path = directory/f"checkpoint-{marker['slot']}.rank{rank}.pt"
    if digest(local_path) != marker['files'][local_path.name]:
        raise ValueError('Inference rank-local state hash differs')
    restore_buffers(runner, torch.load(local_path, map_location='cpu', weights_only=False))
    return marker['files'][path.name]


def evaluate(runner, args, rank, world, *, epoch):
    before = state_hash(unwrap(runner.model).state_dict())
    ema_before = state_hash(runner.ema.model.state_dict()) if runner.ema else None
    scaler = copy.deepcopy(runner.scaler.state_dict()) if runner.fp16 else None
    modes = [(module, module.training) for module in runner.model.modules()]
    # Reset the finite validation view; InfiniteBatchSampler must not advance
    # to another permutation or reuse a prefetched tail after a prior eval.
    iterator(runner, 'test', seed=args.seed, rank=rank, epoch=0)
    result = runner.evaluate()
    for module, training in modes:
        module.training = training
    if (state_hash(unwrap(runner.model).state_dict()) != before
            or (runner.ema and state_hash(runner.ema.model.state_dict()) != ema_before)
            or (runner.fp16 and runner.scaler.state_dict() != scaler)):
        raise ValueError('Evaluation updated model/scaler state')
    values = metrics(result) if rank == 0 else {}
    if rank == 0:
        record = {'after_epoch': epoch, 'metrics': values, 'weights_unchanged': True,
                  'purpose': 'training', 'ema': runner.ema is not None}
        with (args.output/'evaluations.jsonl').open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True)+'\n')
    if world > 1:
        values = [values]
        dist.broadcast_object_list(values, src=0)
        values = values[0]
    return values


def run(args):
    collections.Iterable = collections.abc.Iterable
    np.float, np.int, np.bool = float, int, bool
    os.environ['DEFAULT_TASKS'] = 'cls:det'
    configure_numerics(False)
    # Match model-verify's native high-throughput convolution selection; this
    # permits native fluctuation and does not claim strict determinism.
    torch.backends.cudnn.benchmark = True
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    args.source = args.source.resolve()
    sys.path.insert(0, str(args.source))
    platform = None
    vendor = 'metax' if args.lane == 'new_metax' else 'nvidia'
    if args.lane != 'original_nvidia':
        from up_platform import configure_platform
        platform = configure_platform({'backend': vendor, 'device': f'cuda:{local_rank}'},
                                      purpose='inference' if args.mode == 'inference' else 'training')
        platform.device.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    import up
    if Path(up.__file__).resolve().parent != args.source/'up':
        raise RuntimeError('Wrong UP checkout imported')
    from up.utils.general.global_flag import DIST_BACKEND
    from up.utils.env.gene_env import set_random_seed
    from up.runner.base_runner import BaseRunner
    from full_accuracy_data import register_full_coco
    DIST_BACKEND.backend = 'dist'
    register_full_coco()
    plan = json.loads(args.plan.read_text())
    data = json.loads(args.data.read_text())
    for name, expected_hash in plan['tools'].items():
        if digest(Path(__file__).with_name(name)) != expected_hash:
            raise ValueError('Frozen validation tool changed: '+name)
    settings = plan['models'][args.model]
    if world != plan['world_size']:
        raise ValueError('Actual world size differs from frozen plan')
    if args.seed != plan['seed'] or data['identity'] != plan['data_identity']:
        raise ValueError('Data/seed differs from frozen plan')
    if args.mode != 'initialize' and not args.probe_steps:
        if digest(args.data) not in plan['data_bindings_sha256']:
            raise ValueError('Machine-local data bindings changed after the audit/plan freeze')
        counts = {'imagenet/train': 1281167, 'imagenet/test': 50000,
                  'coco/train': 118287, 'coco/test': 5000}
        if any(data['identity']['splits'][key]['images'] != count for key, count in counts.items()):
            raise ValueError('A preflight/subset cannot be reported as complete full-data training')
        if not data['identity'].get('all_image_bytes_verified') or not data['identity'].get('all_images_rgb_decoded'):
            raise ValueError('Full training requires a complete data integrity audit')
    config = recipe(args.source, data, args.model, batch_size=settings['batch_size'],
                    world_size=world, workers=plan['workers'], seed=args.seed)
    config_sha = canonical_config(config, data, args.source)
    source = source_manifest(args.source)
    if source['sha256'] != plan['sources'][args.lane]['sha256']:
        raise ValueError('Frozen source changed')
    identity = {'lane': args.lane, 'model': args.model, 'seed': args.seed, 'world_size': world,
        'plan_sha256': digest(args.plan), 'data_identity': data['identity'],
        'source_sha256': source['sha256'], 'algorithm_config_sha256': config_sha,
        'driver_sha256': digest(Path(__file__)), 'preflight_steps': args.probe_steps}
    if args.resume:
        if not args.output.is_dir():
            raise FileNotFoundError(args.output)
    elif rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    if platform:
        config['runtime'].update(platform={'backend': vendor, 'device': f'cuda:{local_rank}'},
                                 device=f'cuda:{local_rank}')
    config['saver'].update(save_dir=str(args.output/'native-checkpoints'), results_dir=str(args.output/'predictions'))
    if rank == 0:
        write(args.output/'effective-config.json', config)
        write(args.output/'identity.json', identity)
    runner = BaseRunner(config, work_dir=str(args.output), training=args.mode != 'inference')
    if args.mode == 'initialize':
        if args.lane != 'original_nvidia':
            raise ValueError('Common initial tensors must be generated on original UP/NVIDIA')
        if rank == 0:
            torch.save(cpu(unwrap(runner.model).state_dict()), args.output/'initial.pt')
            write(args.output/'initial.json', dict(identity, file_sha256=digest(args.output/'initial.pt'),
                  state_sha256=state_hash(unwrap(runner.model).state_dict())))
        return
    if args.initial is None:
        raise ValueError('A hash-pinned common initial tensor file is required')
    if digest(args.initial) != settings['initial_sha256']:
        raise ValueError('Common initial file hash differs')
    initial = torch.load(args.initial, map_location='cpu', weights_only=True)
    unwrap(runner.model).load_state_dict(initial, strict=True)
    if runner.ema:
        runner.ema.model.load_state_dict(initial, strict=True)
    initial_state_sha = state_hash(initial)
    if state_hash(unwrap(runner.model).state_dict()) != initial_state_sha:
        raise ValueError('Common initial tensors differ after loading')
    del initial
    set_random_seed(args.seed, config['runtime'].get('rank_init', False))
    progress = {'completed_steps': 0, 'optimizer_updates': 0, 'checkpoint_generation': 0,
                'training_seconds': 0., 'evaluation_seconds': 0.}
    pending_state = None
    if args.resume:
        progress, pending_state = restore_checkpoint(runner, args.output, rank, identity)
        runner.start_iter = progress['completed_steps']
    if args.mode == 'inference':
        if args.checkpoint is None or args.resume:
            raise ValueError('Standalone inference requires --checkpoint and forbids --resume')
        checkpoint_sha = load_inference_checkpoint(runner, args.checkpoint, identity, rank)
        with torch.no_grad():
            iterator(runner, 'test', seed=args.seed, rank=rank, epoch=0)
            file, results = runner.inference()
            result = runner.data_loaders['test'].dataset.evaluate(file, results) if rank == 0 else {}
        if rank == 0:
            write(args.output/'inference.json', {'metrics': metrics(result), 'identity': identity,
                  'checkpoint_sha256': checkpoint_sha, 'weights_exact': True,
                  'purpose': 'inference', 'precision': 'fp32', 'ema': runner.ema is not None,
                  'rank_local_buffers_restored': True})
        return
    epoch_steps = runner.train_epoch_size
    expected = RECIPES[args.model]['epochs']*epoch_steps
    if runner.max_iter != expected or expected != settings['expected_steps']:
        raise ValueError('Native UP step budget differs from frozen plan')
    target = args.probe_steps or expected
    stop = args.checkpoint_stop_after or target
    if args.checkpoint_stop_after and not args.probe_steps:
        raise ValueError('--checkpoint-stop-after is only allowed for a preflight resume audit')
    if not 0 < stop <= target <= expected:
        raise ValueError('Invalid preflight/update budget')
    if progress['completed_steps'] >= stop:
        raise ValueError('Requested budget already completed; reuse evidence rather than restart')
    runner._hooks('before_train')
    successful = [progress['optimizer_updates']]
    step_hook = runner.optimizer.register_step_post_hook(lambda *_: successful.__setitem__(0, successful[0]+1))
    torch.cuda.reset_peak_memory_stats()
    trace = (args.output/f'train.rank{rank}.jsonl').open('a')
    try:
        start_epoch = progress['completed_steps']//epoch_steps
        for epoch in range(start_epoch, RECIPES[args.model]['epochs']):
            skip = progress['completed_steps'] % epoch_steps
            iterator(runner, 'train', seed=args.seed, rank=rank, epoch=epoch, skip=skip)
            if pending_state:
                restore_local(runner, pending_state)
                pending_state = None
            runner.model.train()
            epoch_begin = time.monotonic()
            samples = hashlib.sha256()
            for _ in range(skip, epoch_steps):
                index = progress['completed_steps']
                if index >= stop:
                    break
                if runner.cur_iter != index:
                    raise ValueError('UP scheduler iteration drift')
                step_began = time.monotonic()
                batch = runner.get_batch('train')
                names = batch.get('image_id', batch.get('filenames'))
                samples.update(json.dumps(names, sort_keys=True).encode())
                # Image roots differ by host; compare tensors/labels/IDs rather
                # than embedding local absolute paths in the tensor digest.
                input_hash = batch_hash({key: child for key, child in batch.items()
                    if key not in ('filenames', 'filename')}) if index < 3 else None
                loss = runner.forward_train(batch)
                runner.backward(loss)
                runner.update()
                if runner.ema:
                    runner.ema.step(runner.model, curr_step=index)
                value = float(loss.detach())
                if not np.isfinite(value):
                    raise ValueError('Non-finite loss at step '+str(index))
                # UP evaluates/saves before LR.step. The shared driver evaluates
                # outside the step and checkpoints afterwards, so resume restores
                # the exact LR intended for the next update.
                runner.lr_scheduler.step()
                progress.update(completed_steps=index+1, optimizer_updates=successful[0])
                if args.probe_steps or index < 3 or (index+1) % 100 == 0:
                    record = {'step': index+1, 'epoch': epoch+1, 'loss': value,
                        'step_seconds': time.monotonic()-step_began,
                        'lr_next': [group['lr'] for group in runner.optimizer.param_groups],
                        'optimizer_updates': successful[0], 'input_sha256': input_hash,
                        'sample_chunk_sha256': samples.hexdigest()}
                    trace.write(json.dumps(record, sort_keys=True)+'\n')
                    trace.flush()
                    samples = hashlib.sha256()
                    if rank == 0:
                        write(args.output/'status.json', dict(progress, status='preflight' if args.probe_steps else 'training',
                            expected_steps=expected, epoch=epoch+1, loss=value))
            torch.cuda.synchronize()
            progress['training_seconds'] += time.monotonic()-epoch_begin
            final = progress['completed_steps'] == stop
            if final or (epoch+1) % 5 == 0:
                begin = time.monotonic()
                values = evaluate(runner, args, rank, world, epoch=epoch+1)
                progress['evaluation_seconds'] += time.monotonic()-begin
            progress['checkpoint_generation'] += 1
            checkpoint(runner, args.output, progress, rank, world, identity)
            if rank == 0:
                if final and source_manifest(args.source) != source:
                    raise ValueError('Source changed during execution')
                write(args.output/'status.json', dict(progress,
                    status=('preflight_checkpoint' if stop < target else 'preflight_completed') if final and args.probe_steps else ('completed' if final else 'training'),
                    expected_steps=expected, last_epoch=epoch+1,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            if final:
                if rank == 0:
                    write(args.output/'report.json', {'identity': identity, 'progress': progress,
                        'status': ('preflight_checkpoint' if stop < target else 'preflight_completed') if args.probe_steps else 'completed',
                        'final_metrics': values, 'initial_state_sha256': initial_state_sha,
                        'expected_steps': expected, 'selection': 'final epoch; native EMA where configured',
                        'precision': RECIPES[args.model]['precision'], 'benchmark': torch.backends.cudnn.benchmark,
                        'deterministic': torch.are_deterministic_algorithms_enabled(),
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                        'environment': {'python': sys.version, 'device': torch.cuda.get_device_name(),
                            'packages': {name: metadata.version(name) for name in
                                ('torch', 'torchvision', 'numpy', 'scipy', 'Pillow', 'pycocotools', 'opencv-python')}}})
                break
        runner._hooks('after_train')
    finally:
        step_hook.remove()
        trace.close()
    if source_manifest(args.source) != source:
        raise ValueError('Source changed during execution')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'plan', 'data', 'output', 'initial', 'checkpoint'):
        parser.add_argument('--'+name, type=lambda value: Path(value).absolute(), required=name not in ('initial', 'checkpoint'))
    parser.add_argument('--model', choices=RECIPES, required=True)
    parser.add_argument('--lane', choices=('original_nvidia', 'new_nvidia', 'new_metax'), required=True)
    parser.add_argument('--mode', choices=('initialize', 'train', 'inference'), default='train')
    parser.add_argument('--seed', type=int, default=131)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--probe-steps', type=int, help='Preflight only; never certified as full training')
    parser.add_argument('--checkpoint-stop-after', type=int, help='Preflight only: exit at a saved state, then test --resume')
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        if args.output.is_dir():
            write(args.output/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+
                              f'.rank{os.environ.get("RANK", "0")}.pid{os.getpid()}.json'),
                  {'type': type(error).__name__, 'error': str(error), 'pid': os.getpid()})
            if int(os.environ.get('RANK', 0)) == 0:
                write(args.output/'status.json', {'status': 'failed', 'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
