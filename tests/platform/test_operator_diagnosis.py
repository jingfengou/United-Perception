import importlib
import json
from pathlib import Path

import pytest
import torch


@pytest.fixture
def diagnosis(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'tools'))
    return importlib.import_module('diagnose_platform_operator')


def test_singleton_strided_tensor_can_be_fingerprinted(diagnosis):
    view = torch.arange(5.)[2:3:5]
    result = diagnosis.signature(view)
    assert result['stride'] == [5]
    assert result['sha256'] == diagnosis.signature(torch.tensor([2.]))['sha256']


def test_replay_preserves_noncontiguous_values_and_strides(diagnosis):
    value = torch.arange(12.).reshape(3, 4).t()
    inputs = ((value, None), {'alpha': 2})
    restored = diagnosis.restore_inputs(diagnosis.snapshot(inputs), diagnosis.signature(inputs), 'cpu')
    assert torch.equal(restored[0][0], value)
    assert restored[0][0].stride() == value.stride()
    assert restored[1] == {'alpha': 2}


def test_unfinished_trace_is_not_reported_as_pass(diagnosis, tmp_path):
    trace = diagnosis.OperatorTrace(tmp_path)
    assert not trace.report()['matched_complete_trace']
    trace.error = 'fixture error'
    assert not trace.report()['matched_complete_trace']


def test_trace_keeps_outputs_and_detects_changed_inputs(diagnosis, tmp_path):
    original, repeat = tmp_path/'original', tmp_path/'repeat'
    original.mkdir()
    repeat.mkdir()
    x, y = torch.ones(2), torch.zeros(2)
    trace = diagnosis.OperatorTrace(original, device_type='cpu')
    with trace:
        output = torch.add(x, y)
    assert torch.equal(output, x)
    trace.completed = True
    (original/'trace.json').write_text(json.dumps(trace.report()))
    comparison = diagnosis.OperatorTrace(repeat, original, device_type='cpu')
    changed = torch.ones(2)
    with pytest.raises(diagnosis.FirstDifference), comparison:
        torch.add(x, changed)
    assert comparison.first['operation'] == 'aten.add.Tensor'
    assert comparison.first['inputs_bitwise_equal'] is False
    assert (repeat/'first-difference.pt').exists()


def test_trace_limit_is_not_a_complete_pass(diagnosis, tmp_path):
    trace = diagnosis.OperatorTrace(tmp_path, limit=0, device_type='cpu')
    x = torch.ones(2)
    with pytest.raises(diagnosis.FirstDifference), trace:
        x + x
    assert trace.report()['limit_reached']
    assert not trace.report()['matched_complete_trace']
