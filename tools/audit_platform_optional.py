"""Per-entry optional-toolchain snapshots, retaining unavailable policy variants.

Direct/routed pipelines execute in separate processes because ASP has global
state. A snapshot belongs to the named operation within its stated pipeline;
it is not an isolated benchmark or a claim about every dependency API.
"""
import argparse
import copy
import importlib
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from platform_operator_cases import compare_results, copy_tree


def snapshot(value):
    if isinstance(value, torch.nn.Module):
        return {'state': copy_tree(value.state_dict()), 'type': type(value).__module__+'.'+type(value).__name__}
    if isinstance(value, torch.optim.Optimizer):
        return copy_tree(value.state_dict())
    if isinstance(value, (torch.Tensor, tuple, list, dict)) or value is None or isinstance(value, (str, int, float, bool)):
        return copy_tree(value)
    return {'type': type(value).__module__+'.'+type(value).__name__}


def execute(args):
    from up_platform import configure_platform
    p = configure_platform({'backend': 'nvidia', 'device': 'cuda:0'}, purpose='training')
    p.device.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(131)
    np.random.seed(131)
    values, errors = {}, {}
    routed = args.route == 'routed'
    def save(name, value):
        values[name] = snapshot(value)
    def attempt(name, function):
        try:
            save(name, function())
        except Exception as error:
            errors[name] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}

    if args.case == 'sparse':
        import spconv.pytorch as package
        ops = p.category('sparse')
        indices = torch.tensor([[0, z, y, x] for z in range(2, 5) for y in range(2, 5) for x in range(2, 5)], device='cuda', dtype=torch.int32)
        features = torch.randn(len(indices), 4, device='cuda', requires_grad=True)
        def factory(name, *aa, **kk):
            return (getattr(ops, name) if routed else getattr(package, name))(*aa, **kk)
        tensor = factory('SparseConvTensor', features, indices, [8, 8, 8], 1)
        save('sparse.SparseConvTensor', {'features': tensor.features, 'indices': tensor.indices, 'spatial_shape': tensor.spatial_shape})
        for name, construct in [('SubMConv3d', lambda: factory('SubMConv3d', 4, 4, 3, padding=1, indice_key='s')),
                                ('SparseConv3d', lambda: factory('SparseConv3d', 4, 4, 3, stride=2, padding=1, indice_key='down'))]:
            module = construct().cuda()
            output = module(tensor)
            gradients = torch.autograd.grad(output.features.sum(), (features, *module.parameters()), retain_graph=True)
            save('sparse.'+name, {'output': output.features, 'indices': output.indices,
                                 'gradients': gradients, 'state': module.state_dict()})
            if name == 'SparseConv3d':
                down = output
        inverse = factory('SparseInverseConv3d', 4, 4, 3, indice_key='down').cuda()
        output = inverse(down)
        gradients = torch.autograd.grad(output.features.sum(), (features, *inverse.parameters()), retain_graph=True)
        save('sparse.SparseInverseConv3d', {'output': output.features, 'gradients': gradients, 'state': inverse.state_dict()})
        seq = factory('SparseSequential', torch.nn.ReLU()).cuda()
        save('sparse.SparseSequential', seq(tensor).features)
        save('sparse.features', ops.features(tensor) if routed else tensor.features)
        save('sparse.dense', ops.dense(tensor) if routed else tensor.dense())
        replacement = features*2
        changed = ops.replace_feature(tensor, replacement) if routed else tensor.replace_feature(replacement)
        save('sparse.replace_feature', {'features': changed.features, 'indices': changed.indices,
                                       'aliases_replacement': changed.features is replacement})
        save('sparse.is_sparse_convolution', ops.is_sparse_convolution(inverse) if routed else isinstance(inverse, package.conv.SparseConvolution))
        class Demo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2, device='cuda'))
        if routed:
            instance = ops.module_instance(Demo)
        else:
            bound = type('Demo', (Demo, package.SparseModule), {'__module__': Demo.__module__})
            instance = object.__new__(bound)
        Demo.__init__(instance)
        save('sparse.module_instance', {'is_sparse_module': isinstance(instance, package.SparseModule), 'state': instance.state_dict()})
        def voxel():
            import cumm.tensorview as tv
            from spconv.utils import Point2VoxelCPU3d
            generator = Point2VoxelCPU3d(vsize_xyz=[.5, .5, .5], coors_range_xyz=[0, 0, 0, 4, 4, 4],
                                        num_point_features=4, max_num_voxels=16, max_num_points_per_voxel=4)
            points = np.array([[.1, .1, .1, 1], [.2, .1, .1, 2], [1., 1., 1., 3]], np.float32)
            if routed:
                result = ops.voxelize_cpu(generator, points)
            else:
                result = tuple(value.numpy() for value in generator.point_to_voxel(tv.from_numpy(points)))
            saved = [torch.from_numpy(value.copy()) for value in result]
            generator.point_to_voxel(tv.from_numpy(points[:1]))
            return {'values': saved, 'independent_of_reused_scratch': all(np.array_equal(value, copy.numpy()) for value, copy in zip(result, saved))}
        attempt('sparse.voxelize_cpu', voxel)
    elif args.case in ('quantization', 'qdrop'):
        from mqbench.prepare_by_platform import prepare_by_platform, BackendType
        from mqbench.utils.state import enable_calibration_woquantization, enable_quantization
        from mqbench.advanced_ptq import ptq_reconstruction
        from mqbench.convert_deploy import convert_deploy
        ops = p.category('quantization')
        if args.case == 'quantization':
            types = ops.backend_types() if routed else BackendType
            save('quantization.backend_types', {name: str(value) for name, value in types.__members__.items()})
            paths = {'FrozenBatchNorm2d': 'mqbench.nn.modules', 'ConvFreezebn2d': 'mqbench.nn.intrinsic.qat.modules',
                     'ConvFreezebnReLU2d': 'mqbench.nn.intrinsic.qat.modules', 'FakeQuantizeBase': 'mqbench.fake_quantize.quantize_base'}
            for name, path in paths.items():
                value = ops.module_type(name) if routed else getattr(importlib.import_module(path), name)
                save('quantization.module_type/'+name, value.__module__+'.'+value.__name__)
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
                                    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 4)).cuda().eval()
        config = {}
        backend = BackendType.Tensorrt
        if args.case == 'qdrop':
            backend = BackendType.Academic
            config = {'extra_qconfig_dict': {'w_observer': 'MSEObserver', 'a_observer': 'EMAMSEObserver',
                'a_fakequantize': 'QDropFakeQuantize', 'w_fakequantize': 'AdaRoundFakeQuantize',
                'w_qscheme': {'bit': 2, 'symmetry': False, 'per_channel': True, 'pot_scale': False, 'p': 2.4},
                'a_qscheme': {'bit': 4, 'symmetry': False, 'per_channel': False, 'pot_scale': False, 'p': 2.4}}}
        model = (ops.prepare if routed else prepare_by_platform)(model, backend, config).cuda()
        save('quantization.prepare', model)
        (ops.enable_calibration if routed else enable_calibration_woquantization)(model)
        batches = [torch.randn(2, 3, 8, 8, device='cuda') for _ in range(2)]
        for batch in batches:
            model(batch)
        save('quantization.enable_calibration', model)
        if args.case == 'qdrop':
            from easydict import EasyDict
            config = EasyDict(pattern='block', scale_lr=4e-5, warm_up=.25, weight=.005,
                max_count=4, b_range=[20, 2], keep_gpu=True, round_mode='learned_hard_sigmoid', prob=.5)
            model = (ops.reconstruct if routed else ptq_reconstruction)(model, batches, config)
            save('quantization.reconstruct', {'state': model.state_dict(), 'output': model(batches[0])})
        else:
            (ops.enable_quantization if routed else enable_quantization)(model)
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            value = model(batches[0])
            value.square().mean().backward()
            optimizer.step()
            save('quantization.enable_quantization', {'state': model.state_dict(), 'output': value,
                'gradients': {name: parameter.grad for name, parameter in model.named_parameters()}})
            model.eval()
            def deploy():
                folder = args.output/'onnx'
                folder.mkdir()
                (ops.convert_deploy if routed else convert_deploy)(model, backend, dummy_input=(batches[0],),
                    model_name='operator_audit', output_path=str(folder))
                from compare_platform_exports import graph_signature
                return {path.name: graph_signature(path.read_bytes()) for path in sorted(folder.glob('*.onnx'))}
            attempt('quantization.convert_deploy', deploy)
    elif args.case == 'base_export':
        from msbench.scheduler import build_sparse_scheduler
        ops = p.category('pruning')
        config = {'mask_generator': {'type': 'NormalMaskGenerator', 'kwargs': {'structured': False}},
                  'fake_sparse': {'type': 'DefaultFakeSparse'}, 'scheduler': {'type': 'BaseScheduler'}}
        scheduler = (ops.build_scheduler if routed else build_sparse_scheduler)(copy.deepcopy(config))
        model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8)).cuda()
        model = ops.prepare(scheduler, model) if routed else scheduler.prepare_sparse_model(model)
        scheduler.update_sparsity_per_layer(model, .5)
        (ops.export(scheduler, model) if routed else scheduler.export_sparse_model(model))
        zero_counts = {name: int((parameter == 0).sum()) for name, parameter in model.named_parameters() if name.endswith('weight')}
        assert zero_counts and all(value > 0 for value in zero_counts.values())
        save('pruning.export', {'state': model.state_dict(), 'actual_zero_elements': zero_counts,
                               'reference_policy': 'BaseScheduler/DefaultFakeSparse; not Amba'})
    elif args.case in ('pruning', 'amba'):
        from msbench.scheduler import build_sparse_scheduler
        ops = p.category('pruning')
        from msbench.nn.modules import FrozenBatchNorm2d
        value = ops.module_type('FrozenBatchNorm2d') if routed else FrozenBatchNorm2d
        save('pruning.module_type', value.__module__+'.'+value.__name__)
        config = {'scheduler': {'type': 'AmpereScheduler', 'kwargs': {'allow_permutation': False, 'verbosity': 0}}}
        if args.case == 'amba':
            config = {'mask_generator': {'type': 'NormalMaskGenerator'}, 'fake_sparse': {'type': 'FakeSparse'},
                      'scheduler': {'type': 'AmbaLevelPruneScheduler', 'kwargs': {'total_iters': 4,
                      'sparsity_table': [30], 'prun_algo': 1, 'prun_algo_tuning': .5, 'do_sparse_analysis': False,
                      'output_dir': str(args.output/'analysis'), 'save_dir': str(args.output/'checkpoints'),
                      'default_input_shape': [1, 3, 8, 8]}}}
        scheduler = (ops.build_scheduler if routed else build_sparse_scheduler)(copy.deepcopy(config))
        save('pruning.build_scheduler', scheduler)
        model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8)).cuda()
        model = ops.prepare(scheduler, model) if routed else scheduler.prepare_sparse_model(model)
        save('pruning.prepare', model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        (ops.init_optimizer(scheduler, optimizer) if routed else scheduler.init_optimizer(optimizer))
        assert any('mask' in name for name, _ in model.named_buffers())
        save('pruning.init_optimizer', {'model': model.state_dict(), 'optimizer': optimizer.state_dict()})
        value = model(torch.randn(4, 32, device='cuda'))
        value.square().mean().backward()
        optimizer.step()
        sparse_rate = ops.sparse_rate(scheduler, model) if routed else scheduler.compute_sparse_rate(model)
        save('pruning.sparse_rate', {'rate': sparse_rate, 'actual_zero_elements':
            {name: int((parameter == 0).sum()) for name, parameter in model.named_parameters()}})
        if routed:
            ops.set_iteration(scheduler, 3)
        else:
            scheduler.iteration = 3
        save('pruning.set_iteration', scheduler.iteration)
        # Amba-only prune/export/LR/step semantics are deliberately not replaced
        # by Ampere's inherited no-op Scheduler methods.
    else:
        raise ValueError(args.case)
    return values, errors


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        values, errors = execute(args)
    except Exception as error:
        values, errors = {}, {'pipeline': {'type': type(error).__name__, 'message': str(error),
                                          'traceback': traceback.format_exc()}}
    torch.save(values, args.output/'values.pt')
    (args.output/'summary.json').write_text(json.dumps({'case': args.case, 'route': args.route,
        'entries': list(values), 'errors': errors, 'scope': 'Per-entry snapshots within the named actual dependency pipeline'}, indent=2)+'\n')


def summarize(args):
    rows = []
    for case in ('sparse', 'quantization', 'qdrop', 'pruning', 'base_export', 'amba'):
        roots = [args.results/(case+'-'+route) for route in ('direct', 'routed')]
        reports = [json.loads((root/'summary.json').read_text()) for root in roots]
        values = [torch.load(root/'values.pt', weights_only=False, map_location='cpu') for root in roots]
        for name in sorted(set(values[0]) | set(values[1])):
            row = {'interface': '.'.join(name.split('/')[:1]), 'case': case+'/'+name,
                   'reference_kind': 'original_dependency_API_pipeline_snapshot'}
            if name in values[0] and name in values[1]:
                row['comparison'] = compare_results(values[0][name], values[1][name])
                row['status'] = 'within_tolerance' if row['comparison']['passed'] else 'difference'
            else:
                row['status'] = 'execution_error'
            rows.append(row)
        if any(report['errors'] for report in reports):
            rows.append({'interface': 'optional.'+case, 'case': case, 'status': 'execution_error',
                         'errors': {route: report['errors'] for route, report in zip(('direct', 'routed'), reports)}})
    with args.output.open('x') as handle:
        json.dump({'records': rows, 'rtol': 1e-4, 'atol': 1e-5}, handle, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('sparse', 'quantization', 'qdrop', 'pruning', 'base_export', 'amba'))
    parser.add_argument('--route', choices=('direct', 'routed'))
    parser.add_argument('--results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args) if args.results else worker(args)
