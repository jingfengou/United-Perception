"""Actual two-GPU comparisons for every communication and SyncBN entry."""
import argparse
import copy
from datetime import timedelta
import os
from pathlib import Path

import torch
import torch.distributed as dist

from audit_platform_interfaces import Recorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('nvidia', 'metax'), default='nvidia')
    parser.add_argument('--rendezvous-dir', type=Path,
                        help='Shared local directory supporting POSIX flock; avoid incompatible network filesystems')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from up_platform import configure_platform
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    platform = configure_platform({'backend': args.backend, 'device': f'cuda:{rank}'}, purpose='training')
    ops = platform.category('communication')
    init_count = [0]
    rendezvous = args.rendezvous_dir or args.output
    rendezvous.mkdir(parents=True, exist_ok=True)
    def init_kwargs():
        # Each lifecycle comparison uses its own rendezvous. Reusing torchrun's
        # TCPStore across successive destroy/reinitialize cycles can mix keys.
        path = (rendezvous/('rendezvous-'+str(init_count[0]))).resolve()
        init_count[0] += 1
        return {'backend': 'nccl', 'device_id': torch.device('cuda', rank),
                'timeout': timedelta(seconds=60), 'rank': rank, 'world_size': 2,
                'init_method': path.as_uri()}
    audit = Recorder(args.output/f'rank{rank}', 'distributed', 'training', backend=args.backend)
    audit.pair('communication.is_available', 'compiled_backend', dist.is_available, ops.is_available)
    audit.pair('communication.is_initialized', 'before_init', dist.is_initialized, ops.is_initialized)

    def initialize(function):
        function(**init_kwargs())
        result = {'initialized': dist.is_initialized(), 'rank': dist.get_rank(), 'size': dist.get_world_size()}
        dist.barrier()
        dist.destroy_process_group()
        return result
    audit.pair('communication.init_process_group', 'actual_nccl',
               lambda: initialize(dist.init_process_group), lambda: initialize(ops.init_process_group))
    dist.init_process_group(**init_kwargs())
    retained_groups = []
    try:
        for name in ('is_initialized', 'get_rank', 'get_world_size', 'get_backend'):
            audit.pair('communication.'+name, 'live_group', getattr(dist, name), getattr(ops, name))
        def world(function):
            group = function()
            return {'same_WORLD': group is dist.group.WORLD, 'rank': dist.get_rank(group),
                    'size': dist.get_world_size(group)}
        audit.pair('communication.world_group', 'actual_group', lambda: world(lambda: dist.group.WORLD),
                   lambda: world(ops.world_group))
        audit.pair('communication.sum_op', 'actual_enum', lambda: str(dist.ReduceOp.SUM), lambda: str(ops.sum_op()))
        def subgroup(function):
            group = function(ranks=[0, 1])
            retained_groups.append(group)
            result = {'rank': dist.get_rank(group), 'size': dist.get_world_size(group)}
            dist.barrier(group)
            return result
        audit.pair('communication.new_group', 'two_rank_subgroup', lambda: subgroup(dist.new_group), lambda: subgroup(ops.new_group))
        def barrier(function):
            work = function(async_op=True)
            work.wait()
            return {'waited': True}
        audit.pair('communication.barrier', 'async_wait', lambda: barrier(dist.barrier), lambda: barrier(ops.barrier))
        for method in ('all_reduce', 'broadcast', 'all_gather', 'gather'):
            for asynchronous in (False, True):
                def collective(function, name=method, async_op=asynchronous):
                    value = torch.tensor([rank+1.], device=f'cuda:{rank}')
                    output = value
                    if name == 'all_reduce':
                        work = function(value, op=dist.ReduceOp.SUM, async_op=async_op)
                    elif name == 'broadcast':
                        work = function(value, src=0, async_op=async_op)
                    else:
                        values = [torch.zeros_like(value) for _ in range(2)]
                        if name == 'all_gather':
                            work = function(values, value, async_op=async_op)
                            output = values
                        else:
                            work = function(value, gather_list=values if rank == 0 else None, dst=0, async_op=async_op)
                            output = values if rank == 0 else value
                    if work is not None:
                        work.wait()
                    return output
                audit.pair('communication.'+method, 'async_'+str(asynchronous),
                    lambda m=method: collective(getattr(dist, m)), lambda m=method: collective(getattr(ops, m)))
        def object_gather(function):
            values = [None, None]
            function(values, {'rank': rank, 'text': 'UP'})
            return values
        audit.pair('communication.all_gather_object', 'rank_metadata',
            lambda: object_gather(dist.all_gather_object), lambda: object_gather(ops.all_gather_object))
        def object_broadcast(function):
            values = [{'message': 'UP', 'number': 131} if rank == 0 else None]
            function(values, src=0)
            return values
        audit.pair('communication.broadcast_object_list', 'rank_metadata',
            lambda: object_broadcast(dist.broadcast_object_list), lambda: object_broadcast(ops.broadcast_object_list))
        normalization = platform.category('normalization')
        from torch.nn.modules._functions import SyncBatchNorm
        for method in ('sync_batch_norm', 'sync_batch_norm_forward'):
            def sync(function, lowlevel=method == 'sync_batch_norm'):
                torch.manual_seed(131)
                module = torch.nn.SyncBatchNorm(4).cuda(rank).train()
                torch.manual_seed(131+rank)
                value = torch.randn(2, 4, 4, 4, device=f'cuda:{rank}', requires_grad=True)
                if lowlevel:
                    output = function(value, module.weight, module.bias, module.running_mean, module.running_var,
                                      module.eps, module.momentum, dist.group.WORLD, 2)
                else:
                    output = function(module, value)
                gradients = torch.autograd.grad(output, (value, *module.parameters()), torch.ones_like(output))
                return {'output': output, 'gradients': gradients, 'state': module.state_dict()}
            direct = SyncBatchNorm.apply if method == 'sync_batch_norm' else torch.nn.SyncBatchNorm.forward
            audit.pair('normalization.'+method, 'world2_backward_and_stats',
                lambda: sync(direct), lambda m=method: sync(getattr(normalization, m)))
        def ddp(function):
            torch.manual_seed(131)
            module = torch.nn.Linear(4, 2).cuda(rank)
            wrapped = function(module, device_ids=[rank])
            optimizer = torch.optim.SGD(wrapped.parameters(), lr=.01)
            torch.manual_seed(711+rank)
            value = torch.randn(3, 4, device=f'cuda:{rank}')
            output = wrapped(value)
            output.square().sum().backward()
            gradients = {name: value.grad for name, value in module.named_parameters()}
            optimizer.step()
            state = copy.deepcopy(wrapped.state_dict())
            wrapped.load_state_dict(state)
            return {'output': output, 'gradients': gradients, 'state': state,
                    'parameter_names': list(dict(wrapped.named_parameters()))}
        audit.pair('training.wrap_ddp', 'gradient_update_and_restore',
            lambda: ddp(torch.nn.parallel.DistributedDataParallel), lambda: ddp(platform.execution.wrap_ddp))
    finally:
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
            for group in retained_groups:
                dist.destroy_process_group(group)
            dist.destroy_process_group()
    def destroy(function):
        dist.init_process_group(**init_kwargs())
        dist.barrier()
        function()
        return dist.is_initialized()
    audit.pair('communication.destroy_process_group', 'actual_nccl',
        lambda: destroy(dist.destroy_process_group), lambda: destroy(ops.destroy_process_group))
    audit.finish()


if __name__ == '__main__':
    main()
