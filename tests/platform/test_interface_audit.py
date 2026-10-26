"""Coverage and independence guards for the finite hardware-interface audit."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip('torch')
from torch.utils._pytree import tree_flatten
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from platform_interface_fixtures import (TORCH_CATEGORIES, fixture, interface_methods,
    reference_platform, resolve_torch_reference)
from platform_operator_cases import compare_results, execute_case, restore_tensor


@pytest.mark.parametrize('backend', [None, 'nvidia', 'metax'])
def test_interface_audit_selects_the_requested_vendor(monkeypatch, tmp_path, backend):
    import audit_platform_interfaces as audit
    import up_platform
    platform = SimpleNamespace(device=SimpleNamespace(set_device=Mock()))
    configure = Mock(return_value=platform)
    recorder = SimpleNamespace(finish=Mock())
    make_recorder = Mock(return_value=recorder)
    run = Mock()
    monkeypatch.setattr(up_platform, 'configure_platform', configure)
    monkeypatch.setattr(audit, 'Recorder', make_recorder)
    monkeypatch.setattr(audit, 'basic', run)
    monkeypatch.setattr(sys, 'argv', ['audit', *(['--backend', backend] if backend else []), '--group', 'basic',
                                    '--purpose', 'inference', '--output', str(tmp_path/'audit')])
    audit.main()
    configure.assert_called_once_with({'backend': backend or 'nvidia', 'device': 'cuda:0'}, purpose='inference')
    make_recorder.assert_called_once_with(tmp_path/'audit', 'basic', 'inference', backend=backend or 'nvidia')
    run.assert_called_once_with(platform, recorder)
    recorder.finish.assert_called_once()


def test_default_training_audit_does_not_schedule_legacy_sdk_requirements():
    from audit_platform_interfaces import training_cases
    pairs = []
    legacy = Mock(side_effect=ImportError('not installed'))
    recorder = SimpleNamespace(pair=lambda name, *a, **k: pairs.append(name),
                               blocked=Mock(side_effect=AssertionError('unexpected legacy requirement')))
    training_cases(SimpleNamespace(purpose='training', execution=SimpleNamespace(fused_optimizer=legacy)), recorder)
    legacy.assert_not_called()
    recorder.blocked.assert_not_called()
    assert 'training.optimizer' in pairs and 'training.grad_scaler' in pairs


def test_every_non_distributed_basic_interface_has_a_fixture():
    deferred = {'normalization.batch_norm_module_forward', 'normalization.sync_batch_norm',
                'normalization.sync_batch_norm_forward'}
    covered = set()
    for category in TORCH_CATEGORIES:
        for name in interface_methods(category):
            operation = category+'.'+name
            if operation in deferred:
                continue
            payload = fixture(operation)
            assert payload['operation'] == operation
            assert payload['arguments'][0]
            infos = tree_flatten(payload['argument_info'], is_leaf=lambda x: isinstance(x, dict) and 'device_type' in x)[0]
            assert all(info['device_type'] == 'cuda' for info in infos if isinstance(info, dict) and 'device_type' in info)
            covered.add(operation)
    assert len(covered) == 61


def test_reference_does_not_resolve_through_shared_implementation(monkeypatch):
    from up_platform.ops.matrix import torch_impl
    monkeypatch.setattr(torch_impl, 'linear', lambda *args: torch.zeros(1))
    reference, source = resolve_torch_reference('matrix', 'linear')
    assert reference is torch.nn.functional.linear
    assert source == 'torch.nn.functional.linear'


def test_noncontiguous_fixture_preserves_actual_input_layout():
    payload = fixture('matrix.linear', layout='strided')
    value = payload['arguments'][0][0]
    info = payload['argument_info'][0][0]
    restored = restore_tensor(value, info, 'cpu')
    assert not restored.is_contiguous()
    assert restored.storage_offset() == 1
    assert torch.equal(restored, value)


def test_real_operator_divergence_is_detected():
    from up_platform.backends.cpu import build
    platform = build({}, 'training')
    reference = reference_platform(platform)
    payload = fixture('activation.relu', device_type='cpu')
    expected, _ = execute_case(reference, payload)
    platform.category('activation').relu = torch.neg
    actual, _ = execute_case(platform, payload)
    assert not compare_results(expected, actual)['passed']


def test_batch_norm_and_gradient_state_are_actually_compared():
    from up_platform.backends.cpu import build
    platform = build({}, 'training')
    payload = fixture('normalization.BatchNorm2d', dtype=torch.float64, device_type='cpu')
    expected, _ = execute_case(reference_platform(platform), payload)
    actual, _ = execute_case(platform, payload)
    assert expected['state_after']['num_batches_tracked'].item() == 1
    assert expected['parameter_gradients']['weight'] is not None
    assert compare_results(expected, actual)['passed']
    actual['state_after']['running_mean'].add_(.01)
    assert not compare_results(expected, actual)['passed']


def test_matching_rejection_is_not_counted_as_numeric_equivalence(tmp_path):
    from audit_platform_interfaces import Recorder
    audit = Recorder(tmp_path/'matching', 'test', 'training')
    def reject():
        raise RuntimeError('unsupported dtype')
    row = audit.pair('loss.binary_cross_entropy', 'rejection', reject, reject)
    assert row['status'] == 'matching_error'
    assert 'comparison' not in row


def test_asymmetric_exception_cannot_be_a_pass(tmp_path):
    from audit_platform_interfaces import Recorder
    audit = Recorder(tmp_path/'asymmetric', 'test', 'training')
    def reject():
        raise RuntimeError('unsupported dtype')
    row = audit.pair('matrix.linear', 'asymmetric', reject, lambda: torch.zeros(1))
    assert row['status'] == 'execution_error'


@pytest.mark.parametrize('boundary', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_boundary_is_reported_without_hiding_finite_error(boundary):
    from summarize_interface_audit import nonfinite_agreement
    a = {'state': torch.tensor([boundary, 1.])}
    b = {'state': torch.tensor([boundary, 1.])}
    assert nonfinite_agreement(a, b)['matching_nonfinite_elements'] == 1
    assert not compare_results(a, b)['passed']
    b['state'][1] = 1.1
    assert nonfinite_agreement(a, b) is None


def test_nonfinite_positions_and_signs_must_match():
    from summarize_interface_audit import nonfinite_agreement
    assert nonfinite_agreement(torch.tensor([float('inf')]), torch.tensor([-float('inf')])) is None
    assert nonfinite_agreement(torch.tensor([float('nan'), 1.]), torch.tensor([1., float('nan')])) is None


def test_executed_boundary_state_is_distinct_from_unexecuted_and_finite_pass():
    from collections import Counter
    from summarize_interface_audit import interface_status
    assert interface_status(Counter(matching_nonfinite=2)) == 'matching_nonfinite_state'
    assert interface_status(Counter()) == 'not_executed'
    assert interface_status(Counter(matching_nonfinite=2, difference=1)) == 'unresolved_difference_or_execution'


def test_vendor_summary_keeps_missing_failed_and_rejected_interfaces_separate():
    from summarize_vendor_interfaces import aggregate
    rows = [{'interface': 'matrix.linear', 'status': 'within_tolerance', 'evidence': 'a.json'},
            {'interface': 'matrix.mm', 'status': 'matching_error', 'evidence': 'b.json'},
            {'interface': 'matrix.bmm', 'status': 'within_tolerance', 'evidence': 'c.json'},
            {'interface': 'matrix.bmm', 'status': 'difference', 'evidence': 'd.json', 'repeat_difference': True}]
    result = {r['interface']: r for r in aggregate(rows, ['matrix.linear', 'matrix.mm', 'matrix.bmm', 'matrix.addmm'])}
    assert result['matrix.linear']['status'] == 'measured_within_tolerance'
    assert result['matrix.mm']['status'] == 'not_executed'
    assert result['matrix.addmm']['case_count'] == 0
    assert result['matrix.addmm']['max_abs'] is None
    assert result['matrix.bmm']['status'] == 'unresolved_difference_or_execution'
    assert result['matrix.bmm']['repeat_difference_cases'] == 1
    with pytest.raises(ValueError, match='Unknown interfaces'):
        aggregate(rows, ['matrix.linear'])
