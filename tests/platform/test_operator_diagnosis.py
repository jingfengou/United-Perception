import importlib
import json
from pathlib import Path
from types import SimpleNamespace

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


def native_fixture(operation):
    return SimpleNamespace(roi_align=SimpleNamespace(forward_avg_cuda=operation, backward_avg_cuda=operation),
                           overlaps=SimpleNamespace(iou=operation), naive_nms=SimpleNamespace(gpu_nms=operation))


def test_native_trace_observes_mutated_buffer_and_restores_binding(diagnosis, tmp_path):
    def operation(*args):
        args[7].fill_(3)
        return 17
    extension = native_fixture(operation)
    trace = diagnosis.OperatorTrace(tmp_path, device_type='cpu')
    buffer = torch.zeros(2)
    with trace.native_boundaries(extension), trace:
        returned = extension.roi_align.backward_avg_cuda(False, 2, 2, 1., 2, torch.ones(2), torch.zeros(1, 5), buffer)
    assert returned == 17
    assert extension.roi_align.backward_avg_cuda is operation
    assert trace.entries[0]['operation'] == 'native.roi_align.backward_avg_cuda'
    assert torch.equal(torch.load(tmp_path/'00000.pt', weights_only=False), torch.full((2,), 3.))


def test_native_trace_detects_first_difference_with_identical_inputs(diagnosis, tmp_path):
    delta = [0.]
    def operation(*args):
        args[7].fill_(delta[0])
    extension = native_fixture(operation)
    original, repeat = tmp_path/'original', tmp_path/'repeat'
    original.mkdir()
    repeat.mkdir()
    args = (False, 2, 2, 1., 2, torch.ones(2), torch.zeros(1, 5), torch.zeros(2))
    trace = diagnosis.OperatorTrace(original, device_type='cpu')
    with trace.native_boundaries(extension):
        extension.roi_align.backward_avg_cuda(*args)
    (original/'trace.json').write_text(json.dumps(trace.report()))
    delta[0] = 1.
    trace = diagnosis.OperatorTrace(repeat, original, device_type='cpu')
    with pytest.raises(diagnosis.FirstDifference), trace.native_boundaries(extension):
        extension.roi_align.backward_avg_cuda(*args)
    assert trace.first['operation'] == 'native.roi_align.backward_avg_cuda'
    assert trace.first['inputs_bitwise_equal']
    assert extension.roi_align.backward_avg_cuda is operation


def test_nms_scratch_tail_is_not_compared(diagnosis, tmp_path):
    def operation(keep, count, *args):
        keep[:2] = torch.tensor([0, 2])
        count.fill_(2)
    trace = diagnosis.OperatorTrace(tmp_path, device_type='cpu')
    extension = native_fixture(operation)
    with trace.native_boundaries(extension):
        extension.naive_nms.gpu_nms(torch.tensor([99, 99, 99]), torch.tensor([99]), torch.zeros(3, 5), .5, 1)
    value = torch.load(tmp_path/'00000.pt', weights_only=False)
    assert torch.equal(value[0], torch.tensor([0, 2]))


def test_native_backward_counts_existing_gradient_as_input(diagnosis, tmp_path):
    def operation(*args):
        args[7].add_(1)
    extension = native_fixture(operation)
    original, repeat = tmp_path/'original', tmp_path/'repeat'
    original.mkdir()
    repeat.mkdir()
    args = (False, 2, 2, 1., 2, torch.ones(2), torch.zeros(1, 5), torch.zeros(2))
    trace = diagnosis.OperatorTrace(original, device_type='cpu')
    with trace.native_boundaries(extension):
        extension.roi_align.backward_avg_cuda(*args)
    (original/'trace.json').write_text(json.dumps(trace.report()))
    trace = diagnosis.OperatorTrace(repeat, original, device_type='cpu')
    with pytest.raises(diagnosis.FirstDifference), trace.native_boundaries(extension):
        extension.roi_align.backward_avg_cuda(*args)
    assert not trace.first['inputs_bitwise_equal']
