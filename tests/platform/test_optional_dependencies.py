import copy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from up_platform.dependencies import available, dependency
from up_platform.ops.pruning import torch_impl


def recipe(fake_sparse='FakeSparse'):
    return {'scheduler': {'type': 'AmbaLevelPruneScheduler', 'kwargs': {'total_iters': 4}},
            'fake_sparse': {'type': fake_sparse}, 'mask_generator': {'type': 'NormalMaskGenerator'}}


def test_missing_pruning_policy_fails_before_scheduler_construction():
    config = recipe()
    before = copy.deepcopy(config)
    prepare = SimpleNamespace(FakeSparseTable={'DefaultFakeSparse': object()},
                              MaskGeneratorTable={'NormalMaskGenerator': object()})
    with patch.object(torch_impl, 'dependency', return_value=prepare) as load:
        with pytest.raises(RuntimeError, match="no fake_sparse 'FakeSparse'"):
            torch_impl.build_scheduler(config)
    load.assert_called_once_with('msbench.prepare')
    assert config == before


def test_supported_pruning_policy_keeps_config_identity_and_result():
    config = recipe('DefaultFakeSparse')
    prepare = SimpleNamespace(FakeSparseTable={'DefaultFakeSparse': object()},
                              MaskGeneratorTable={'NormalMaskGenerator': object()})
    scheduler = SimpleNamespace(build_sparse_scheduler=Mock(return_value=object()))
    with patch.object(torch_impl, 'dependency', side_effect=[prepare, scheduler]):
        result = torch_impl.build_scheduler(config)
    assert result is scheduler.build_sparse_scheduler.return_value
    assert scheduler.build_sparse_scheduler.call_args.args[0] is config


def test_ampere_does_not_require_fake_sparse_policy():
    config = {'scheduler': {'type': 'AmpereScheduler'}}
    scheduler = SimpleNamespace(build_sparse_scheduler=Mock())
    with patch.object(torch_impl, 'dependency', return_value=scheduler) as load:
        torch_impl.build_scheduler(config)
    load.assert_called_once_with('msbench.scheduler')
    scheduler.build_sparse_scheduler.assert_called_once_with(config)


def test_optional_sdk_exit_is_an_explicit_error_not_a_process_exit():
    with patch('up_platform.dependencies.import_module', side_effect=SystemExit(1)):
        with pytest.raises(RuntimeError, match="msbench.scheduler.*exited during import") as caught:
            dependency('msbench.scheduler')
    assert isinstance(caught.value.__cause__, SystemExit)


def test_missing_sdk_keeps_original_error():
    with patch('up_platform.dependencies.import_module', side_effect=ImportError('missing vendor library')):
        with pytest.raises(RuntimeError, match='missing vendor library') as caught:
            dependency('optional_vendor_sdk')
    assert isinstance(caught.value.__cause__, ImportError)


def test_capability_probe_does_not_exit_when_parent_sdk_exits():
    with patch('up_platform.dependencies.util.find_spec', side_effect=SystemExit(1)):
        assert available('optional_vendor_sdk') is False


def test_metax_sparse_type_uses_the_actual_vendor_base_class(monkeypatch):
    from up_platform.ops.sparse import metax
    class VendorSparseConvolution:
        pass
    class VendorInverseConvolution(VendorSparseConvolution):
        pass
    loader = Mock(return_value=SimpleNamespace(SparseConvolution=VendorSparseConvolution))
    monkeypatch.setattr(metax, 'dependency', loader)
    assert metax.MetaxSparseOps.is_sparse_convolution(VendorInverseConvolution())
    assert not metax.MetaxSparseOps.is_sparse_convolution(object())
    assert all(call.args == ('spconv.pytorch.modules.macaconv',) for call in loader.call_args_list)
