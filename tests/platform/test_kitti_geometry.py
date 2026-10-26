from types import SimpleNamespace

import numpy as np
import pytest

from up_platform.ops.pointcloud import _kitti_gpu


def test_empty_geometry_does_not_load_sdk(monkeypatch):
    def forbidden(*args):
        raise AssertionError('empty geometry must not load the extension')
    monkeypatch.setattr(_kitti_gpu, 'load_native', forbidden)
    for n, k in ((0, 3), (3, 0), (0, 0)):
        value = _kitti_gpu.rotate_iou_eval(np.zeros((n, 5)), np.zeros((k, 5)), vendor='metax')
        assert value.shape == (n, k)
        assert value.dtype == np.float32


@pytest.mark.parametrize('bad', [np.zeros((2, 4)), np.zeros(5),
                                np.array([[0, 0, -1, 1, 0]]), np.full((1, 5), np.nan)])
def test_invalid_geometry_is_rejected_before_sdk(bad):
    for boxes, queries in ((bad, np.zeros((0, 5))), (np.zeros((0, 5)), bad)):
        with pytest.raises(ValueError):
            _kitti_gpu.rotate_iou_eval(boxes, queries, vendor='metax')


def test_stale_binary_requires_rebuild(monkeypatch):
    monkeypatch.setattr(_kitti_gpu, 'load_native', lambda vendor: SimpleNamespace())
    with pytest.raises(RuntimeError, match='Rebuild.*UP_BUILD_BACKEND=metax'):
        _kitti_gpu.rotate_iou_eval(np.ones((1, 5)), np.ones((1, 5)), vendor='metax')


def test_vendors_bind_corrected_geometry_explicitly(monkeypatch):
    from up_platform.ops.pointcloud.nvidia import NvidiaPointcloudOps
    from up_platform.ops.pointcloud.metax import MetaxPointcloudOps
    calls = []
    monkeypatch.setattr(_kitti_gpu, 'rotate_iou_eval', lambda *args, **kwargs: calls.append((args, kwargs)))
    for implementation, vendor in ((NvidiaPointcloudOps, 'nvidia'), (MetaxPointcloudOps, 'metax')):
        implementation.rotate_iou_eval('boxes', 'queries', 2, 3)
        assert calls[-1] == (('boxes', 'queries', 2, 3), {'vendor': vendor})
