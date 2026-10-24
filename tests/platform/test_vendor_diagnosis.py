"""Guard diagnostic evidence against false tolerance/mask conclusions."""
import importlib.util
from pathlib import Path

import pytest
import torch


def tool(name):
    path = Path(__file__).resolve().parents[2] / 'tools' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_near_zero_uses_original_absolute_tolerance():
    compare = tool('compare_vendor_prefix').compare_tensor
    row = compare(torch.tensor([0.]), torch.tensor([2e-5]))
    assert not row['passed'] and row['outside_tolerance'] == 1


def test_small_roundoff_is_not_called_exact():
    compare = tool('compare_vendor_prefix').compare_tensor
    row = compare(torch.tensor([100.]), torch.tensor([100.001]))
    assert row['passed'] and not row['exact'] and row['outside_tolerance'] == 0


def test_random_mask_comparison_requires_exact_boolean_values():
    compare = tool('compare_vendor_prefix').compare_tensor
    row = compare(torch.tensor([True, False]), torch.tensor([False, False]))
    assert not row['passed'] and row['mismatched_elements'] == 1


def test_tree_comparison_rejects_truncated_captures():
    compare = tool('compare_vendor_prefix').compare_tree
    with pytest.raises(ValueError, match='lengths'):
        compare((torch.ones(1), torch.zeros(1)), (torch.ones(1),))


def test_read_saved_mask_when_zero_input_hides_dropout_pattern():
    saved_mask = tool('observe_vendor_prefix').saved_mask
    x = torch.zeros(2, requires_grad=True)
    actual_mask = torch.tensor([0., 1.])
    output = x * actual_mask
    recovered, source = saved_mask(output)
    assert torch.equal(recovered, actual_mask)
    assert source == 'MulBackward0._saved_other'
    assert not torch.equal(output != 0, actual_mask.bool())


def test_random_creation_counts_device_without_tensor_operands(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'tools'))
    observer = tool('observe_vendor_writes').Writes(tmp_path)

    class RandomCreation:
        def __str__(self):
            return 'aten.rand.default'

        def __call__(self, *args, **kwargs):
            return None  # Dispatch fixture: no CUDA context needed.

    observer.__torch_dispatch__(RandomCreation(), (), ([64, 1, 1],), {'device': torch.device('cuda:0')})
    assert observer.random_calls == [{'step': 0, 'operation': 'aten.rand.default'}]
