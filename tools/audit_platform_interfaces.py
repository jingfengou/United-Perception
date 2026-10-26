"""Per-interface vendor contract comparisons with independent Torch references.

Run only on allocated devices. This tool never changes model/runtime source,
relaxes tolerances, or treats missing dependencies as successful comparisons.
Native binaries and optional/distributed interfaces have separate workers.
"""
import argparse
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import traceback

import torch

from platform_interface_fixtures import (DISTRIBUTED_NORMALIZATION, TORCH_CATEGORIES,
    fixture, interface_methods, reference_platform)
from platform_operator_cases import compare_results, copy_tree, execute_case

RTOL, ATOL = 1e-4, 1e-5


class Recorder:
    def __init__(self, output, group, purpose, *, backend=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.group, self.purpose = group, purpose
        self.backend = backend
        self.records, self.raw = [], {}

    def pair(self, interface, case, reference, routed, *, reference_kind='declared_Torch_API'):
        row = {'interface': interface, 'case': case, 'reference_kind': reference_kind}
        values, errors = {}, {}
        for name, function in (('reference', reference), ('reference_repeat', reference), ('routed', routed),
                               ('routed_repeat', routed)):
            try:
                values[name] = copy_tree(function())
            except Exception as error:
                errors[name] = {'type': type(error).__name__, 'message': str(error),
                                'traceback': traceback.format_exc()}
        if errors:
            same_error = len(errors) == 4 and len({(item['type'], item['message']) for item in errors.values()}) == 1
            row.update(status='matching_error' if same_error else 'execution_error', errors=errors)
        else:
            row.update(comparison=compare_results(values['reference'], values['routed']),
                       reference_repeat=compare_results(values['reference'], values['reference_repeat']),
                       routed_repeat=compare_results(values['routed'], values['routed_repeat']))
            row['status'] = 'within_tolerance' if row['comparison']['passed'] else 'difference'
        self.records.append(row)
        self.raw[interface+'/'+case] = values
        self.flush()
        print(json.dumps({k: row[k] for k in ('interface', 'case', 'status')}), flush=True)
        return row

    def blocked(self, interface, reason, variant='required_dependency'):
        self.records.append({'interface': interface, 'case': variant,
                             'status': 'blocked_dependency', 'reason': reason})
        self.flush()

    def flush(self):
        record = {'group': self.group, 'purpose': self.purpose, 'backend': self.backend, 'rtol': RTOL, 'atol': ATOL,
                  'records': self.records, 'torch': str(torch.__version__),
                  'scope': 'Finite per-interface fixtures; not every dtype/shape/variant or dataset accuracy'}
        temporary = self.output/'summary.json.tmp'
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(self.output/'summary.json')

    def finish(self):
        raw_path = self.output/'values.pt'
        torch.save(self.raw, raw_path)
        (self.output/'values-sha256.json').write_text(json.dumps({
            'file': raw_path.name, 'bytes': raw_path.stat().st_size,
            'sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest()}, indent=2)+'\n')
        self.flush()


def basic(platform, recorder):
    reference = reference_platform(platform)
    for category in TORCH_CATEGORIES:
        for name in sorted(interface_methods(category)):
            if name in DISTRIBUTED_NORMALIZATION | {'batch_norm_module_forward'}:
                continue
            interface = category+'.'+name
            for variant, dtype, layout, amp in (
                    ('fp32', torch.float32, 'contiguous', False),
                    ('fp32_strided', torch.float32, 'strided', False),
                    ('fp64', torch.float64, 'contiguous', False),
                    ('amp_fp16', torch.float32, 'contiguous', True)):
                payload = fixture(interface, dtype=dtype, layout=layout,
                                  training=platform.purpose == 'training', amp=amp)
                recorder.pair(interface, variant,
                    lambda: replay(reference, payload), lambda: replay(platform, payload))
    def batch_norm_forward(function):
        torch.manual_seed(131)
        module = torch.nn.BatchNorm2d(4).to('cuda')
        module.train(platform.purpose == 'training')
        value = torch.randn(2, 4, 6, 6, device='cuda', requires_grad=True)
        output = function(module, value)
        gradients = torch.autograd.grad(output, (value, *module.parameters()), torch.ones_like(output))
        return {'output': output, 'gradients': gradients, 'state': module.state_dict()}
    recorder.pair('normalization.batch_norm_module_forward', 'state_and_backward',
        lambda: batch_norm_forward(torch.nn.BatchNorm2d.forward),
        lambda: batch_norm_forward(platform.category('normalization').batch_norm_module_forward))


def replay(platform, payload):
    result, metadata = execute_case(platform, payload)
    return {'result': result, 'rng': metadata}


def additional(platform, recorder):
    reference = reference_platform(platform)
    inplace_names = {'ReLU', 'ReLU6', 'Hardswish', 'SiLU', 'SELU', 'LeakyReLU',
                     'relu', 'relu6', 'hardtanh', 'silu'}
    for category in TORCH_CATEGORIES:
        for name in sorted(interface_methods(category)):
            if name in DISTRIBUTED_NORMALIZATION | {'batch_norm_module_forward'}:
                continue
            interface = category+'.'+name
            empty = fixture(interface, layout='empty', training=platform.purpose == 'training')
            recorder.pair(interface, 'empty_batch', lambda: replay(reference, empty), lambda: replay(platform, empty))
            if category == 'activation' and name in inplace_names:
                payload = fixture(interface, inplace=True, training=platform.purpose == 'training')
                recorder.pair(interface, 'inplace_state_and_gradient', lambda: replay(reference, payload), lambda: replay(platform, payload))
            if name[:1].isupper():
                payload = fixture(interface, training=platform.purpose == 'training')
                aa, kk = payload['constructor']
                direct = getattr(reference.category(category), name)
                routed = getattr(platform.category(category), name)
                def constructor(function):
                    torch.manual_seed(131)
                    module = function(*aa, **kk).cuda()
                    state = module.state_dict(keep_vars=True)
                    return {'state': state, 'cpu_rng_after': torch.get_rng_state(),
                            'gpu_rng_after': torch.cuda.get_rng_state(),
                            'parameter_keys': list(dict(module.named_parameters())),
                            'parameter_identity': all(state[key] is value for key, value in module.named_parameters())}
                recorder.pair(interface, 'constructor_state_rng_parameter_identity',
                    lambda: constructor(direct), lambda: constructor(routed))
    def self_attention(factory):
        torch.manual_seed(131)
        module = factory(8, 2, batch_first=True, dropout=.2).cuda()
        module.train(platform.purpose == 'training')
        value = torch.randn(2, 3, 8, device='cuda', requires_grad=True)
        with torch.set_grad_enabled(platform.purpose == 'training'):
            output, weights = module(value, value, value)
        gradients = torch.autograd.grad(output, (value, *module.parameters()), torch.ones_like(output)) if output.requires_grad else ()
        return {'output': output, 'attention_weights': weights, 'gradients': gradients, 'state': module.state_dict()}
    recorder.pair('attention.MultiheadAttention', 'aliased_self_attention',
        lambda: self_attention(torch.nn.MultiheadAttention),
        lambda: self_attention(platform.category('attention').MultiheadAttention))


def device_cases(platform, recorder):
    runtime = platform.device
    cuda = torch.cuda
    simple = {'available': cuda.is_available, 'device_count': cuda.device_count,
              'current_device': cuda.current_device, 'device': lambda: torch.device('cuda', 0),
              'environment': lambda: {'vendor': platform.vendor, 'device_type': 'cuda',
                                     'torch': torch.__version__, 'cuda_api_version': torch.version.cuda}}
    for name, reference in simple.items():
        def normalize(function):
            value = function()
            return str(value) if isinstance(value, torch.device) else value
        recorder.pair('device.'+name, 'runtime_value', lambda: normalize(reference),
                      lambda: normalize(getattr(runtime, name)))
    def properties(function):
        value = function(0)
        return {k: getattr(value, k) for k in ('name', 'total_memory', 'major', 'minor', 'multi_processor_count')}
    recorder.pair('device.properties', 'physical_device', lambda: properties(cuda.get_device_properties),
                  lambda: properties(runtime.properties))
    def bind(function):
        function(0)
        return {'current': cuda.current_device(), 'allocated_device': str(torch.empty(1, device='cuda').device)}
    recorder.pair('device.set_device', 'bind_0', lambda: bind(cuda.set_device), lambda: bind(runtime.set_device))
    for selected in ('cpu', 'cuda'):
        value = torch.ones(2, device=selected)
        recorder.pair('device.is_accelerator', selected, lambda: value.is_cuda,
                      lambda: runtime.is_accelerator(value))
    from collections import OrderedDict, namedtuple
    Pair = namedtuple('Pair', 'value label')
    def moved(function):
        torch.manual_seed(131)
        module = torch.nn.Linear(3, 2)
        values = OrderedDict(a=torch.arange(6.).reshape(2, 3), b=[torch.ones(2), Pair(torch.ones(1), 'p')],
                             module=module)
        result = function(values)
        return {'types': [type(result).__name__, type(result['b']).__name__, type(result['b'][1]).__name__],
                'values': (result['a'], result['b'][0], result['b'][1].value),
                'devices': [str(x.device) for x in (result['a'], result['b'][0], result['b'][1].value)],
                'module_state': result['module'].state_dict(),
                'same_module': result['module'] is module}
    def original_move(value):
        if isinstance(value, (torch.Tensor, torch.nn.Module)):
            return value.cuda(0)
        if isinstance(value, dict):
            return type(value)((k, original_move(v)) for k, v in value.items())
        if isinstance(value, tuple):
            return type(value)(*(original_move(x) for x in value)) if hasattr(value, '_fields') else tuple(original_move(x) for x in value)
        if isinstance(value, list):
            return [original_move(x) for x in value]
        return value
    recorder.pair('device.move', 'nested_module_tensor', lambda: moved(original_move),
                  lambda: moved(runtime.move), reference_kind='original_cuda_calls_and_container_contract')
    def sync(function):
        value = torch.arange(32., device='cuda').square()
        function()
        return value
    recorder.pair('device.synchronize', 'pending_kernel', lambda: sync(cuda.synchronize), lambda: sync(runtime.synchronize))
    def cache(function):
        gc.collect()
        value = torch.ones(128, device='cuda')
        before = cuda.memory_allocated()
        function()
        return {'live_unchanged': before == cuda.memory_allocated(), 'value': value}
    recorder.pair('device.empty_cache', 'retain_live_storage', lambda: cache(cuda.empty_cache), lambda: cache(runtime.empty_cache))
    for name in ('memory_allocated', 'max_memory_allocated', 'reset_peak_memory_stats'):
        def memory(function, method=name):
            gc.collect()
            cuda.reset_peak_memory_stats()
            before = cuda.memory_allocated()
            value = torch.empty(1024, device='cuda')
            if method == 'reset_peak_memory_stats':
                del value
                function()
                return cuda.max_memory_allocated() == cuda.memory_allocated()
            return function()-before
        recorder.pair('device.'+name, 'allocator_counter', lambda n=name: memory(getattr(cuda, n)),
                      lambda n=name: memory(getattr(runtime, n)))
    for name in ('get_rng_state', 'set_rng_state', 'manual_seed'):
        def rng(function, method=name):
            torch.manual_seed(131)
            cuda.manual_seed(131)
            before = torch.get_rng_state().clone()
            if method == 'get_rng_state':
                result = function()
            elif method == 'set_rng_state':
                state = cuda.get_rng_state()
                torch.rand(7, device='cuda')
                function(state)
                result = torch.rand(7, device='cuda')
            else:
                function(991)
                result = torch.rand(7, device='cuda')
            return {'result': result, 'cpu_rng_unchanged': torch.equal(before, torch.get_rng_state())}
        recorder.pair('device.'+name, 'generator_state', lambda n=name: rng(getattr(cuda, n)),
                      lambda n=name: rng(getattr(runtime, n)))
    def flags(function):
        old = {name: getattr(torch.backends.cudnn, name) for name in ('enabled', 'benchmark', 'deterministic', 'allow_tf32')}
        try:
            return function(enabled=True, benchmark=False, deterministic=True, allow_tf32=False)
        finally:
            for name, value in old.items():
                setattr(torch.backends.cudnn, name, value)
    def direct_flags(**kwargs):
        for name, value in kwargs.items():
            setattr(torch.backends.cudnn, name, value)
        return {name: getattr(torch.backends.cudnn, name) for name in kwargs}
    recorder.pair('device.convolution_flags', 'set_restore', lambda: flags(direct_flags),
                  lambda: flags(runtime.convolution_flags))


def amp_case(platform, recorder):
    name = platform.purpose+'.autocast'
    for enabled in (False, True):
        def run(context):
            torch.manual_seed(131)
            x = torch.randn(4, 8, device='cuda', requires_grad=True)
            weight = torch.randn(8, 8, device='cuda', requires_grad=True)
            with context(enabled=enabled, dtype=torch.float16):
                value = torch.nn.functional.linear(x, weight)
            gradients = torch.autograd.grad(value, (x, weight), torch.ones_like(value))
            return {'output': value, 'gradients': gradients}
        recorder.pair(name, 'enabled_'+str(enabled),
            lambda: run(lambda **kwargs: torch.amp.autocast('cuda', **kwargs)),
            lambda: run(platform.execution.autocast))


def training_cases(platform, recorder):
    amp_case(platform, recorder)
    for name in ('SGD', 'AdamW'):
        def optimizer(factory):
            value = torch.nn.Parameter(torch.arange(8., device='cuda'))
            optim = factory(name, params=[value], lr=.01)
            for _ in range(2):
                optim.zero_grad()
                value.square().sum().backward()
                optim.step()
            state = copy.deepcopy(optim.state_dict())
            optim.load_state_dict(state)
            return {'parameter': value, 'state': optim.state_dict()}
        recorder.pair('training.optimizer', name,
            lambda: optimizer(lambda n, **kwargs: getattr(torch.optim, n)(**kwargs)),
            lambda: optimizer(platform.execution.optimizer))
    def scaler(factory):
        scaler = factory(init_scale=32., growth_interval=1)
        value = torch.nn.Parameter(torch.tensor([2.], device='cuda'))
        optim = torch.optim.SGD([value], lr=.1)
        for coefficient in (1., float('inf'), 1.):
            optim.zero_grad()
            scaler.scale(value.square().sum()*coefficient).backward()
            scaler.step(optim)
            scaler.update()
        state = copy.deepcopy(scaler.state_dict())
        scaler.load_state_dict(state)
        return {'value': value, 'scaler': scaler.state_dict(), 'optimizer': optim.state_dict()}
    recorder.pair('training.grad_scaler', 'finite_overflow_restore',
        lambda: scaler(lambda **kwargs: torch.amp.GradScaler('cuda', **kwargs)),
        lambda: scaler(platform.execution.grad_scaler))
    def clip(function):
        value = torch.nn.Parameter(torch.arange(8., device='cuda'))
        value.grad = torch.arange(8., device='cuda')
        total = function([value], max_norm=1., norm_type=2.)
        return {'norm': total, 'gradient': value.grad}
    recorder.pair('training.clip_grad_norm_', 'l2', lambda: clip(torch.nn.utils.clip_grad_norm_),
                  lambda: clip(platform.execution.clip_grad_norm_))
    from contextlib import contextmanager
    @contextmanager
    def reference_validation(module):
        modes = [(child, child.training) for child in module.modules()]
        try:
            module.eval()
            with torch.no_grad():
                yield module
        finally:
            for child, mode in modes:
                child.training = mode
    def validate(context):
        torch.manual_seed(131)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4), torch.nn.Dropout()).cuda()
        model.train()
        model[2].eval()
        before = copy.deepcopy(model.state_dict())
        with context(model):
            output = model(torch.ones(3, 4, device='cuda'))
            assert not torch.is_grad_enabled()
            assert all(not child.training for child in model.modules())
            assert platform.purpose == 'training'
        return {'output': output, 'unchanged': all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()),
                'modes': [child.training for child in model.modules()]}
    recorder.pair('training.validation', 'mixed_modes_no_state_update',
        lambda: validate(reference_validation), lambda: validate(platform.execution.validation),
        reference_kind='explicit_eval_no_grad_and_entry_mode_restore')


def inference_cases(platform, recorder):
    amp_case(platform, recorder)
    from compare_platform_exports import graph_signature
    def export(function):
        torch.manual_seed(131)
        module = torch.nn.Linear(4, 2).cuda().eval()
        output = io.BytesIO()
        function(module, (torch.ones(2, 4, device='cuda'),), output, opset_version=11, do_constant_folding=False)
        return graph_signature(output.getvalue())
    recorder.pair('inference.export_onnx', 'legacy_graph',
        lambda: export(lambda *args, **kwargs: torch.onnx.export(*args, dynamo=False, **kwargs)),
        lambda: export(platform.execution.export_onnx), reference_kind='original_legacy_ONNX_export_semantics')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('nvidia', 'metax'), default='nvidia')
    parser.add_argument('--group', choices=('basic', 'additional', 'device', 'training', 'inference'), required=True)
    parser.add_argument('--purpose', choices=('training', 'inference'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from up_platform import configure_platform
    p = configure_platform({'backend': args.backend, 'device': 'cuda:0'}, purpose=args.purpose)
    p.device.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    audit = Recorder(args.output, args.group, args.purpose, backend=args.backend)
    try:
        {'basic': basic, 'additional': additional, 'device': device_cases, 'training': training_cases,
         'inference': inference_cases}[args.group](p, audit)
    finally:
        audit.finish()


if __name__ == '__main__':
    main()
