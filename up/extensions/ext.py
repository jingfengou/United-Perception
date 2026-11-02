"""Delay loading the NVIDIA binary until an actual native operation is called."""
import importlib

from up.runtime import get_runtime


class _LazyExtension:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        # 只在真正使用专用算子时加载原二进制，普通分类模型导入 UP 不会触发加载。
        if get_runtime().platform != 'nvidia':
            raise NotImplementedError('{} requires UP\'s NVIDIA extension'.format(self.name))
        try:
            extension = importlib.import_module('up.extensions._C')
        except ModuleNotFoundError as error:
            # 兼容扩展的两种历史安装路径；内部依赖缺失则继续抛出原错误。
            if error.name != 'up.extensions._C':
                raise
            extension = importlib.import_module('up_extensions._C')
        return getattr(getattr(extension, self.name), name)


roi_align = _LazyExtension('roi_align')
psroi_align = _LazyExtension('psroi_align')
psroi_pooling = _LazyExtension('psroi_pooling')
naive_nms = _LazyExtension('naive_nms')
softer_nms = _LazyExtension('softer_nms')
deform_conv_v1 = _LazyExtension('deform_conv_v1')
focal_loss = _LazyExtension('focal_loss')
cross_focal_loss = _LazyExtension('cross_focal_loss')
overlaps = _LazyExtension('overlaps')
roipoint_pool3d = _LazyExtension('roipoint_pool3d')
roiaware_pool3d = _LazyExtension('roiaware_pool3d')
iou3d_nms = _LazyExtension('iou3d_nms')
