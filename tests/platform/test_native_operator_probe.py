"""The diagnostic must preserve the operation that the real model receives."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from probe_native_operators import RepeatPure, compare, input_hash


def test_replaying_inplace_index_add_does_not_add_twice(tmp_path):
    actual = torch.zeros(2)
    indices = torch.tensor([0, 0, 1])
    values = torch.tensor([1., 2., 4.])
    with RepeatPure(tmp_path) as probe:
        returned = actual.index_add_(0, indices, values)
    assert returned is actual
    assert torch.equal(actual, torch.tensor([3., 4.]))
    assert not probe.findings


def test_pure_reduction_returns_original_value(tmp_path):
    values = torch.arange(12.).reshape(3, 4)
    before = values.clone()
    with RepeatPure(tmp_path):
        result = values.sum(0)
    assert torch.equal(result, torch.tensor([12., 15., 18., 21.]))
    assert torch.equal(values, before)


def test_replay_fingerprint_detects_shape_dtype_and_layout():
    values = torch.arange(4.).reshape(2, 2)
    assert input_hash((values,)) != input_hash((values.flatten(),))
    assert input_hash((values,)) != input_hash((values.to(torch.float64),))
    assert input_hash((values,)) != input_hash((values.t(),))
    assert compare(values, values.clone())[0]['bitwise_equal']
