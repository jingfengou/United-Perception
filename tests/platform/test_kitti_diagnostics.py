import importlib
from pathlib import Path

import numpy as np


def load_tool(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'tools'))
    return importlib.import_module('compare_platform_kitti')


def test_kitti_comparison_rejects_broadcastable_wrong_shape(monkeypatch):
    tool = load_tool(monkeypatch)
    result = tool.comparison_record(np.ones((2, 2), np.float32), np.ones((1, 2), np.float32))
    assert not result['passed']
    assert result['reason'] == 'shape/dtype mismatch'


def test_kitti_reference_sanity_does_not_replace_reference(monkeypatch):
    tool = load_tool(monkeypatch)
    boxes = np.array([[0, 0, 2, 2, .3], [0, 0, 0, 0, 0]], np.float32)
    original = np.array([[.3333, 0.], [0., 0.]], np.float32)
    saved = original.copy()
    result = tool.self_overlap_sanity(boxes, boxes, original)
    assert result['identical_box_pairs'] == result['non_unit_self_iou'] == 1
    assert np.array_equal(original, saved)
    assert tool.comparison_record(original, saved)['passed']
