from types import SimpleNamespace

import pytest
import torch

from up_platform.ops.detection._roi_align import RoIAlignFunction


def test_deterministic_backward_rejects_stale_extension():
    context = SimpleNamespace(saved_tensors=(torch.zeros(1, 1, 2, 2), torch.zeros(1, 5)),
                              pooled_mode='AVG', _roi_align=SimpleNamespace())
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match='Rebuild.*deterministic RoIAlign'):
            RoIAlignFunction.backward(context, torch.ones(1, 1, 1, 1))
    finally:
        torch.use_deterministic_algorithms(previous)
