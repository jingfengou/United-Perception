"""Two-device FSDP2 acceptance, run with the selected vendor's torchrun.

Synthetic inputs test execution/continuation, not dataset convergence or speed.
The output directory must be shared by this job's ranks and previously absent.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from up_platform import configure_platform


def model():
    return torch.nn.Sequential(torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4), torch.nn.ReLU(),
        torch.nn.Flatten(), torch.nn.Linear(4*8*8, 16), torch.nn.LayerNorm(16)), torch.nn.Linear(16, 3))


def full(tensor):
    return tensor.full_tensor() if hasattr(tensor, 'full_tensor') else tensor


def check(left, right):
    torch.testing.assert_close(full(left), full(right), rtol=1e-4, atol=1e-5)


def main(args):
    platform = configure_platform({'backend': args.vendor}, purpose=args.purpose)
    device = platform.device
    rank = int(os.environ.get('RANK', 0))
    device.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    output = Path(args.output)
    checkpoint = output/'checkpoint.pth'
    if args.purpose == 'inference':
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        candidate = device.move(model()).eval()
        candidate.load_state_dict(state['model'])
        x = device.move(state['input'])
        with torch.no_grad():
            check(candidate(x).cpu(), state['prediction'])
        (output/'inference.json').write_text(json.dumps({'purpose': platform.purpose,
            'vendor': platform.vendor, 'checkpoint': 'full CPU tensors, unchanged model keys', 'passed': True})+'\n')
        return

    comm, training = platform.category('communication'), platform.execution
    comm.init_process_group(backend='nccl')
    torch.set_num_threads(2)
    torch.manual_seed(131 + rank)
    reference = device.move(model())
    candidate = copy.deepcopy(reference)
    ema = copy.deepcopy(reference).eval().requires_grad_(False)
    ddp = training.wrap_ddp(reference, device_ids=[device.current_device()], broadcast_buffers=False)
    training.wrap_fsdp(candidate, shard_modules=['0'])
    training.wrap_fsdp(ema, shard_modules=['0'])
    assert all(hasattr(p, 'to_local') for p in candidate.parameters())
    assert list(reference.state_dict()) == list(candidate.state_dict())
    make_optimizer = lambda m: training.optimizer('AdamW', params=[
        {'params': m[0].parameters(), 'lr': .001}, {'params': m[1].parameters(), 'lr': .002}])
    ref_opt, opt = make_optimizer(reference), make_optimizer(candidate)
    ref_scaler = training.grad_scaler(init_scale=32, enabled=args.amp)
    scaler = training.sharded_grad_scaler(init_scale=32, enabled=args.amp)
    generator = torch.Generator().manual_seed(99+rank)
    x, target = device.move(torch.randn(4, 3, 8, 8, generator=generator)), device.move(torch.tensor([0, 1, 2, 0]))

    def step(m, optimizer, scale):
        optimizer.zero_grad(set_to_none=True)
        with training.autocast(enabled=args.amp):
            loss = torch.nn.functional.cross_entropy(m(x), target)
        scale.scale(loss).backward()
        scale.unscale_(optimizer)
        norm = training.clip_grad_norm_(m.parameters(), 1.)
        scale.step(optimizer)
        scale.update()
        return loss.detach(), full(norm)

    for _ in range(3):
        a, na = step(ddp, ref_opt, ref_scaler)
        b, nb = step(candidate, opt, scaler)
        check(a, b)
        check(na, nb)
        for p, q in zip(reference.parameters(), candidate.parameters()):
            check(p, q)
    with torch.no_grad():
        for k, v in ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(.9).add_(candidate.state_dict()[k], alpha=.1)

    modes = [m.training for m in candidate.modules()]
    before_buffers = {k: v.clone() for k, v in candidate.named_buffers()}
    before_scaler = scaler.state_dict()
    # Deliberately unequal batch counts, including a rank with no forward calls.
    with training.validation(candidate):
        assert platform.purpose == 'training'
        for _ in range(2 if rank == 0 else 0):
            check(candidate(x), reference.eval()(x))
    reference.train()
    assert [m.training for m in candidate.modules()] == modes
    assert before_scaler == scaler.state_dict()
    assert all(torch.equal(v, dict(candidate.named_buffers())[k]) for k, v in before_buffers.items())
    with training.validation(ema):
        if rank == 0:
            assert torch.isfinite(ema(x)).all()

    # UP checkpoint adapters retain each rank's BN/RNG/scaler, without importing UP.
    spec = importlib.util.spec_from_file_location('checkpoint_adapter',
        Path(__file__).resolve().parents[2]/'up/utils/env/platform.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    states = adapter.fsdp_rank_state(candidate, scaler)
    state = training.full_state_dict(candidate, opt)
    with training.validation(candidate):
        prediction = candidate(x).cpu()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        torch.save({**state, 'rank_states': states, 'scaler': scaler.state_dict(),
                    'input': x.cpu(), 'prediction': prediction}, checkpoint)
    comm.barrier()
    restored = device.move(model())
    training.wrap_fsdp(restored, shard_modules=['0'])
    restored_opt = make_optimizer(restored)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    training.load_full_state_dict(restored, saved, restored_opt)
    restored_scaler = training.sharded_grad_scaler(init_scale=32, enabled=args.amp)
    adapter.restore_fsdp_rank_state(restored, saved['rank_states'], restored_scaler)
    a, na = step(candidate, opt, scaler)
    b, nb = step(restored, restored_opt, restored_scaler)
    check(a, b)
    check(na, nb)
    for p, q in zip(candidate.parameters(), restored.parameters()):
        check(p, q)

    # Nonfinite gradients injected into one local shard must skip ALL ranks.
    if args.amp:
        before = [p.to_local().detach().clone() for p in restored.parameters()]
        restored_opt.zero_grad(set_to_none=True)
        with training.autocast():
            loss = torch.nn.functional.cross_entropy(restored(x), target)
        restored_scaler.scale(loss).backward()
        if rank == 0:
            next(restored.parameters()).grad.to_local().fill_(float('inf'))
        old_scale = restored_scaler.get_scale()
        restored_scaler.step(restored_opt)
        restored_scaler.update()
        assert restored_scaler.get_scale() == old_scale*.5
        assert all(torch.equal(a, b.to_local()) for a, b in zip(before, restored.parameters()))

    if rank == 0:
        (output/'training.json').write_text(json.dumps(dict(vendor=platform.vendor, torch=str(torch.__version__),
            world_size=comm.get_world_size(), amp=args.amp, passed=True, rtol=1e-4, atol=1e-5,
            checks=['DDP loss/parameters/clipping', 'DTensor shards', 'parameter groups', 'sharded EMA',
                    'uneven eval', 'mode/buffer/scaler preservation', 'full checkpoint keys',
                    'optimizer/BN/RNG/scaler continuation', 'one-rank overflow skips all ranks']))+'\n')
    comm.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor', required=True, choices=['nvidia', 'metax'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--purpose', default='training', choices=['training', 'inference'])
    parser.add_argument('--amp', action='store_true')
    main(parser.parse_args())
