import numpy as np
import torch

from up_platform.ops._native.loader import load_native


def rotate_iou_eval(boxes, query_boxes, criterion=-1, device_id=0, *, vendor):
    boxes = np.ascontiguousarray(boxes, dtype=np.float32)
    queries = np.ascontiguousarray(query_boxes, dtype=np.float32)
    for name, values in (('boxes', boxes), ('query_boxes', queries)):
        if values.ndim != 2 or values.shape[1] != 5:
            raise ValueError(f'{name} must have shape [N,5]')
        if not np.isfinite(values).all() or (values[:, 2:4] < 0).any():
            raise ValueError(f'{name} must be finite with nonnegative dimensions')
    if not len(boxes) or not len(queries):
        return np.zeros((len(boxes), len(queries)), dtype=np.float32)
    native = load_native(vendor)
    if getattr(native, 'kitti_geometry_version', None) != 2:
        raise RuntimeError(f'Rebuild the corrected KITTI extension with UP_BUILD_BACKEND={vendor}')
    torch.cuda.set_device(device_id)
    device = torch.device('cuda', device_id)
    result = native.kitti_rotate_iou(torch.as_tensor(boxes, device=device),
                                     torch.as_tensor(queries, device=device), criterion)
    # The original evaluator explicitly returns host NumPy metrics.
    return result.cpu().numpy()
