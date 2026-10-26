"""Portable captures must retain state, layouts and the real vendor boundary."""
import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from platform_operator_cases import PlatformCalls, compare_results, execute_case, load_cases, restore_tensor, tensor_info
from up_platform.backends.cpu import build


@pytest.fixture
def platform():
    return build({}, 'training')


def payload(directory, index=0):
    return torch.load(directory/('case-%04d.pt' % index), weights_only=False)


def test_batch_norm_module_restores_common_state_before_each_replay(platform, tmp_path):
    with PlatformCalls(platform, tmp_path) as observer:
        module = platform.category('normalization').BatchNorm2d(3)
        x = torch.arange(48, dtype=torch.float32).reshape(2, 3, 2, 4)/19
        with observer.observing('training'):
            expected = module(x)
    case = payload(tmp_path)
    assert case['state']['num_batches_tracked'].item() == 0
    first, _ = execute_case(platform, case)
    again, _ = execute_case(platform, case)
    direct, _ = execute_case(platform, case, direct=True)
    assert compare_results(first, again)['passed']
    assert compare_results(first, direct)['passed']
    assert torch.equal(first['output'], expected)
    assert first['state_after']['num_batches_tracked'].item() == 1
    assert len(first['input_gradients']) == 1
    assert set(first['parameter_gradients']) == {'weight', 'bias'}


def test_inplace_capture_keeps_original_values_and_nonzero_offset(platform, tmp_path):
    x = torch.tensor([99., -2., 3., 17.])[1:3]
    with PlatformCalls(platform, tmp_path) as observer, observer.observing('training'):
        platform.category('activation').relu(x, inplace=True)
    case = payload(tmp_path)
    assert torch.equal(case['arguments'][0][0], torch.tensor([-2., 3.]))
    assert case['argument_info'][0][0]['storage_offset'] == 1
    result, _ = execute_case(platform, case)
    assert torch.equal(result['output'], torch.tensor([0., 3.]))
    assert result['input_gradients'][0][0].item() == 0.
    assert result['input_gradients'][0][1].item() == pytest.approx(-47/97)


def test_transposed_offset_and_expanded_operands_keep_layout(platform, tmp_path):
    x = torch.arange(24., requires_grad=False).reshape(4, 6)[:, 1::2].t()
    y = torch.arange(20.).reshape(4, 5).requires_grad_()
    with PlatformCalls(platform, tmp_path) as observer, observer.observing('training'):
        expected = platform.category('matrix').matmul(x, y)
    case = payload(tmp_path)
    restored = restore_tensor(case['arguments'][0][0], case['argument_info'][0][0], 'cpu')
    assert restored.stride() == x.stride() and restored.storage_offset() == x.storage_offset()
    result, _ = execute_case(platform, case)
    assert torch.equal(result['output'], expected)
    assert len(result['input_gradients']) == 2
    expanded = torch.arange(3.).expand(4, 3)
    restored = restore_tensor(expanded.clone(), tensor_info(expanded), 'cpu')
    assert restored.stride() == expanded.stride() and torch.equal(restored, expanded)


def test_functional_batch_norm_reports_updated_running_statistics(platform, tmp_path):
    mean, variance = torch.zeros(3), torch.ones(3)
    x = torch.arange(24.).reshape(2, 3, 2, 2)
    with PlatformCalls(platform, tmp_path) as observer, observer.observing('training'):
        platform.category('normalization').batch_norm(x, mean, variance, training=True)
    result, _ = execute_case(platform, payload(tmp_path))
    assert torch.equal(result['state_after']['running_mean'], mean)
    assert torch.equal(result['state_after']['running_var'], variance)


def test_dropout_records_stochastic_contract_without_transferring_rng_bytes(platform, tmp_path):
    with PlatformCalls(platform, tmp_path) as observer:
        module = platform.category('activation').Dropout(.5)
        with observer.observing('training'):
            module(torch.ones(64))
    case = payload(tmp_path)
    assert case['stochastic'] and 'rng_before' not in case
    first, info = execute_case(platform, case)
    second, _ = execute_case(platform, case)
    assert compare_results(first, second)['passed']
    assert info['rng_before_sha256'] != info['rng_after_sha256']


def test_eval_preserves_no_grad_and_records_partial_variant_coverage(platform, tmp_path):
    with PlatformCalls(platform, tmp_path, cases_per_op=1) as observer:
        module = platform.category('matrix').Linear(3, 2)
        with torch.no_grad(), observer.observing('training_eval'):
            module(torch.ones(2, 3))
            module(torch.ones(4, 3))
        observer.completed = True
    report = observer.report()
    assert report['variant_counts']['training_eval/matrix.Linear'] == 2
    assert len(report['cases']) == 1
    result, _ = execute_case(platform, payload(tmp_path))
    assert result['input_gradients'] == [] and result['parameter_gradients'] == {}


def test_failed_observation_restores_original_method(platform, tmp_path):
    original = platform.category('matrix').matmul
    with pytest.raises(RuntimeError), PlatformCalls(platform, tmp_path):
        raise RuntimeError('diagnostic interrupted')
    assert platform.category('matrix').matmul is original


def test_corrupt_or_incomplete_cases_cannot_be_reported_as_pass(platform, tmp_path):
    with PlatformCalls(platform, tmp_path) as observer, observer.observing('training'):
        platform.category('matrix').matmul(torch.ones(2, 3), torch.ones(3, 4))
    (tmp_path/'observation.json').write_text(json.dumps(observer.report()))
    with pytest.raises(ValueError, match='Incomplete'):
        load_cases(tmp_path)
    observer.completed = True
    (tmp_path/'observation.json').write_text(json.dumps(observer.report()))
    load_cases(tmp_path)
    (tmp_path/'case-0000.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='changed'):
        load_cases(tmp_path)


def test_numeric_comparison_rejects_missing_gradients_dtype_and_nonfinite():
    expected = {'output': torch.ones(2), 'gradient': torch.zeros(2)}
    assert not compare_results(expected, {'output': torch.ones(2)})['passed']
    assert not compare_results(torch.ones(2), torch.ones(2, dtype=torch.float64))['passed']
    result = compare_results(torch.tensor([float('nan')]), torch.tensor([float('nan')]))
    assert not result['passed'] and result['leaves'][0]['max_abs'] is None
