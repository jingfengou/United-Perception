"""CPU contracts; distributed device execution lives in fsdp_smoke.py."""
import importlib.util
import inspect
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from up_platform.backends.cpu import CpuDevice
from up_platform.training.cpu import CpuTraining
from up_platform.training.fsdp import _named_optimizer_state


ROOT = Path(__file__).resolve().parents[2]


def adapter():
    spec = importlib.util.spec_from_file_location('fsdp_up_adapter', ROOT/'up/utils/env/platform.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fsdp_vendor_bindings_and_template_contracts_are_explicit():
    from up_platform.training.base import TrainingBackend
    from up_platform.training.nvidia import NvidiaTraining
    from up_platform.training.metax import MetaxTraining
    for cls in (NvidiaTraining, MetaxTraining, CpuTraining):
        assert not inspect.isabstract(cls)
        for name in TrainingBackend.__abstractmethods__:
            assert name in cls.__dict__
            assert inspect.signature(getattr(cls, name)) == inspect.signature(getattr(TrainingBackend, name))


def test_cpu_cannot_silently_replace_fsdp_gpu_execution():
    with pytest.raises(RuntimeError, match='cpu/training: fsdp:.*accelerator'):
        CpuTraining(CpuDevice()).wrap_fsdp(torch.nn.Linear(2, 2))


def test_fsdp_capabilities_never_claim_cpu_or_template_verification():
    from up_platform.backends.cpu import build
    platform = build({}, 'training')
    cap = platform.capabilities()['training.wrap_fsdp']
    assert not cap.registered and not cap.dependency_available and not cap.verified and not cap.executed
    with pytest.raises(RuntimeError, match='training.wrap_fsdp'):
        platform.require(['training.wrap_fsdp'])
    assert 'training.wrap_fsdp' not in build({}, 'inference').capabilities()


def test_fsdp_config_does_not_repurpose_vendor_or_communication_backend():
    options = adapter().fsdp_options
    kwargs = dict(backend='dist', purpose='training', runner_type='base')
    assert options({}, **kwargs) is None
    config = {'runtime': {'platform': {'backend': 'metax'}, 'distributed': {
        'strategy': 'fsdp2', 'shard_modules': ['backbone.blocks.0']}}}
    assert options(config, **kwargs) == {'shard_modules': ['backbone.blocks.0']}
    assert options(config, **{**kwargs, 'purpose': 'inference'}) is None
    with pytest.raises(ValueError, match='backend dist'):
        options(config, **{**kwargs, 'backend': 'linklink'})
    with pytest.raises(ValueError, match='base runner'):
        options(config, **{**kwargs, 'runner_type': 'quant'})


@pytest.mark.parametrize('update,error', [
    ({'hooks': [{'type': 'auto_checkpoint'}]}, 'signal'),
    ({'trainer': {'optimizer': {'type': 'FusedFP16AdamW'}}}, 'SGD/Adam/AdamW'),
    ({'runtime': {'distributed': {'strategy': 'fsdp2', 'cpu_offload': True}}}, 'Unknown FSDP'),
    ({'runtime': {'distributed': {'strategy': 'fsdp1'}}}, 'ddp or fsdp2'),
    ({'runtime': {'distributed': {'strategy': 'ddp', 'shard_modules': []}}}, 'require strategy'),
    ({'runtime': {'distributed': {'strategy': 'fsdp2'}, 'async_norm': True}}, 'async_norm'),
    ({'runtime': {'distributed': {'strategy': 'fsdp2'}, 'model_helper': {'type': 'mimic'}}}, 'base ModelHelper'),
])
def test_fsdp_rejects_unsupported_options_without_changing_algorithms(update, error):
    config = {'runtime': {'distributed': {'strategy': 'fsdp2'}}, **update}
    with pytest.raises(ValueError, match=error):
        adapter().fsdp_options(config, backend='dist', purpose='training', runner_type='base')


def test_legacy_optimizer_ids_convert_to_names_and_keep_groups():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW([{'params': [model.weight], 'lr': .01},
                                   {'params': [model.bias], 'lr': .02}])
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    state = optimizer.state_dict()
    named = _named_optimizer_state(model, optimizer, state)
    assert set(named['state']) == {'weight', 'bias'}
    assert [g['lr'] for g in named['param_groups']] == [.01, .02]
    assert named['state']['weight']['exp_avg'] is state['state'][0]['exp_avg']
    assert state['param_groups'][0]['params'] == [0]
    assert _named_optimizer_state(model, optimizer, named) == named
    named['param_groups'][0]['params'] = ['bias']
    with pytest.raises(ValueError, match='does not match'):
        _named_optimizer_state(model, optimizer, named)
    with pytest.raises(ValueError, match='group count'):
        _named_optimizer_state(model, optimizer, {'state': {}, 'param_groups': []})


def test_checkpoint_loading_preserves_caller_dicts_metadata_and_tensor_storage(monkeypatch):
    import torch.distributed.checkpoint.state_dict as dcp
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    weight = state['model']['weight']
    moment = state['optimizer']['state'][0]['exp_avg']
    metadata = dict(state['model']._metadata)
    def load_model(module, incoming, **kwargs):
        assert incoming._metadata == metadata
        assert incoming['weight'] is weight  # No full checkpoint tensor clone.
        incoming['weight'] = 'redistributed tensor'
    def load_optimizer(module, opt, incoming, **kwargs):
        incoming['state']['weight']['exp_avg'] = 'redistributed moment'
        incoming['param_groups'][0]['params'][0] = 'changed name'
    monkeypatch.setattr(dcp, 'set_model_state_dict', load_model)
    monkeypatch.setattr(dcp, 'set_optimizer_state_dict', load_optimizer)
    CpuTraining(CpuDevice()).load_full_state_dict(model, state, optimizer)
    assert state['model']['weight'] is weight
    assert state['optimizer']['state'][0]['exp_avg'] is moment
    assert state['optimizer']['param_groups'][0]['params'] == [0, 1]


def test_plain_validation_has_no_fsdp_import_or_collective_requirement():
    # Regression for the default DDP/single-device validation path.
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2)).train()
    model[1].eval()
    modes = [m.training for m in model.modules()]
    with CpuTraining(CpuDevice()).validation(model):
        assert not any(m.training for m in model.modules())
        model(torch.ones(2, 3))
    assert [m.training for m in model.modules()] == modes


@pytest.mark.parametrize('fsdp', [False, True])
def test_fsdp_evaluation_hook_failure_is_visible_and_ddp_keeps_legacy_behavior(fsdp):
    source = ast.parse((ROOT/'up/runner/base_runner.py').read_text(encoding='utf8'))
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'BaseRunner')
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_evaluate')
    comm = SimpleNamespace(broadcast_object_list=lambda *a, **k: None)
    namespace = dict(env=SimpleNamespace(is_master=lambda: True), json=json,
        logger=SimpleNamespace(info=lambda *a: None, warning=lambda *a: None),
        barrier=lambda: None, get_platform=lambda: SimpleNamespace(category=lambda n: comm))
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'base_runner_evaluate', 'exec'), namespace)
    def failure(*args):
        raise RuntimeError('checkpoint hook failed')
    runner = SimpleNamespace(use_fsdp=fsdp, inference=lambda: (None, None),
        data_loaders={'test': SimpleNamespace(dataset=SimpleNamespace(evaluate=lambda *a: {}))},
        _hooks=failure, set_cur_eval_iter=lambda: None)
    if fsdp:
        with pytest.raises(RuntimeError, match='checkpoint hook failed'):
            namespace['_evaluate'](runner)
    else:
        assert namespace['_evaluate'](runner) == {}


@pytest.mark.parametrize('assign', [False, True])
def test_up_checkpoint_loader_accepts_torch_assign_without_changing_default_identity(assign):
    source = ast.parse((ROOT/'up/models/model_helper.py').read_text(encoding='utf8'))
    original = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'ModelHelper')
    method = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'load_state_dict')
    wrapper = ast.parse('class ModelHelper(torch.nn.Module):\n    pass').body[0]
    wrapper.body = [method]
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), 'up_checkpoint_loader', 'exec'), namespace)
    model = namespace['ModelHelper']()
    parameter = torch.nn.Parameter(torch.zeros(3))
    model.register_parameter('weight', parameter)
    result = model.load_state_dict({'weight': torch.ones(3), 'model_cfg': {}}, strict=True, assign=assign)
    assert not result.missing_keys and not result.unexpected_keys
    assert torch.equal(model.weight, torch.ones(3))
    if not assign:
        assert model.weight is parameter
