"""Vendor delivery interface; not automatically loaded by UP.

Copy into up/platform/<vendor>/native_ops.py and implement the required methods.
Each method returns the complete operation, including autograd where specified.
The framework maintainer wires completed methods before the legacy Function.apply.
See VENDOR_IMPLEMENTATION.md for NVIDIA references and contracts.
"""


class NativeOpsTemplate:
    def __init__(self, platform):
        self.platform = platform

    def _missing(self, operation):
        raise NotImplementedError("{}: {} is not implemented".format(self.platform, operation))

    def roi_align(self, features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode, *, aligned):
        """RoIAlign 区域特征采样

        输入: features[N,C,H,W]；rois[K,5]=[batch_idx,x1,y1,x2,y2]；pooled_h/pooled_w 为输出尺寸；spatial_scale 为坐标缩放；sampling_ratio 为采样数；pooled_mode="AVG"；aligned 为原 ALIGNED_FLAG.aligned。
        输出: Tensor[K,C,pooled_h,pooled_w]；与 features 同设备、同浮点 dtype。
        反向: 对 features 求梯度；rois 和标量参数返回 None。保存 rois、特征尺寸和采样参数。
        输入按原包装转为 contiguous，rois 转成 features.dtype。空输出直接返回零尺寸 Tensor。MAX 的 Python 分支存在，但原 pybind 未导出 MAX 内核；本轮不要求补 MAX。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roi_align.py#L30
        """
        return self._missing("roi_align")

    def psroi_align(self, features, rois, group_size, spatial_scale, output_dim, sampling_ratio):
        """PSRoIAlign 位置敏感采样

        输入: features[N,C,H,W]；rois[K,5]；group_size、spatial_scale、output_dim、sampling_ratio。外层按通道数计算 output_dim。
        输出: Tensor[K,output_dim,group_size,group_size]，同特征设备/dtype；内部 mapping_channel 为 int32、同设备。
        反向: 只对 features 求梯度；保存 rois、mapping_channel、输入尺寸和采样参数。
        保持位置到通道的映射，不替换成普通 RoIAlign。原包装没有专门的零 ROI 分支，空输入行为需与 NVIDIA 参考实测对齐。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/psroi_align.py#L20
        """
        return self._missing("psroi_align")

    def psroi_pool(self, features, rois, group_size, spatial_scale, output_dim):
        """PSRoIPool 位置敏感池化

        输入: features[N,C,H,W]；rois[K,5]；group_size、spatial_scale、output_dim。
        输出: Tensor[K,output_dim,group_size,group_size]，同设备/dtype；内部 mapping_channel 为 int32。
        反向: 只对 features 求梯度；保存 rois、mapping_channel 和输入尺寸。
        按原池化 bin 边界、取整和位置通道映射实现；与 PSRoIAlign 分开。空输入先测 NVIDIA 参考。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/psroi_pool.py#L25
        """
        return self._missing("psroi_pool")

    def deform_conv(self, input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups):
        """DeformConv V1 可变形卷积

        输入: input[N,Cin,H,W]、offset[N,2*deformable_groups*kh*kw,Hout,Wout]、weight[Cout,Cin/groups,kh,kw]；out_channels、kernel_size、stride、pad、dilations、groups、deformable_groups，二维参数沿用原 tuple。
        输出: Tensor[N,Cout,Hout,Wout]；Hout/Wout 按原 _output_size 公式；无 bias 参数。
        反向: 按 needs_input_grad 提供 input、offset、weight 三者梯度；保存必要 columns/ones 等缓冲区或厂商等效状态。
        这是 DCNv1，不是增加 modulation mask 的 DCNv2。保留参数键、采样顺序、groups 与 deformable_groups；小输入仍按原公式报错。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/deformable_conv.py#L31
        """
        return self._missing("deform_conv")

    def naive_nms(self, dets, thresh, *, offset):
        """UP Native NMS

        输入: dets[N,5]=[x1,y1,x2,y2,score]，按调用方约定降序排列；thresh 为 IoU 阈值；offset 为原 ALIGNED_FLAG.offset（0/1）。
        输出: 保留索引 Tensor[K]，int64，返回输入 dets 所在设备。
        反向: 无梯度要求。
        保留 > 阈值判断、坐标 offset 和索引顺序。原实现包含 CPU 位掩码筛选；这不是要求新增 CPU 模型代算。与连续坐标的 TorchVision NMS 路线分开。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/nms.py#L9
        """
        return self._missing("naive_nms")

    def sigmoid_focal(self, preds, targets, weight_pos, gamma, alpha, num_classes, reduction):
        """Sigmoid Focal Loss

        输入: preds[N,C] 为 logits；targets[N] 为 int32；背景0、前景1..C、忽略-1；weight_pos 为单元素 normalizer Tensor；gamma、alpha、num_classes=C、reduction。
        输出: reduction="none" 返回 [N,C]，其他值返回 [1] 的求和结果；原 Python 包装输出 float32。
        反向: 只对 preds 求梯度，乘上传入 grad_output；targets/weight_pos/标量无梯度。
        分母 max(weight_pos,1)。已有 up.platform.ops.sigmoid_focal_loss 可复用，但原生公开入口、极值、dtype 和 reduction 需对照验收。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/focal_loss.py#L11
        """
        return self._missing("sigmoid_focal")

    def softmax_focal(self, preds, targets, weight_pos, gamma, alpha, num_classes):
        """Softmax Focal Loss

        输入: preds[N,C] logits；targets[N] int32，类别0是背景、负值忽略；weight_pos、gamma、alpha、num_classes=C。
        输出: float32 Tensor[1]，求和 loss；该接口没有 reduction 参数。
        反向: 只对 preds 求梯度；保存 softmax priors、targets、normalizer，原实现还用 [N] 缓冲区。
        先做逐行稳定 softmax；背景与前景使用不同 alpha 权重；分母 max(weight_pos,1)。不能用 Sigmoid Focal 替代。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/focal_loss.py#L82
        """
        return self._missing("softmax_focal")

    def cross_focal(self, preds, targets, weight_pos, gamma, alpha, num_classes, gt_class_to_avoid, reduction):
        """Cross Sigmoid Focal Loss

        输入: preds/targets/weight_pos/gamma/alpha/num_classes 同 Sigmoid Focal；另有 gt_class_to_avoid：int32 打包类别屏蔽表，格式按 kernel 的 neg_map 解码；保留 reduction 参数。
        输出: 原实现实际始终返回 float32 Tensor[1]：reduction 分支位于提前 return 之后，当前不可达。
        反向: 只对 preds 求梯度；保存 targets、normalizer 和屏蔽表。
        照原 neg_map 规则屏蔽对应负类项。此次交付保持原实际输出；如修 reduction 行为，另列语义修复。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/cross_focal_loss.py#L12
        """
        return self._missing("cross_focal")

    def overlap(self, b1, b2, mode='IoU'):
        """二维框 IoU / IoF / IoS

        输入: b1[N,>=4]、b2[M,>=4]，只取前4列 x1/y1/x2/y2；mode="IoU" / "IoF" / "IoS"。
        输出: 非空输入返回 [N,M]，同 b1 设备/dtype；任一输入为空时原接口返回 [0]，不是 [N,M]。
        反向: 原扩展没有注册几何反向；不用给该匹配/评估入口新增梯度。
        +1 闭区间；IoU 分母为并集，IoF 为第一组面积，IoS 为第二组面积；各分母至少1。共享 box_overlap 目前只覆盖 IoU/IoF，不能把 IoS 传给它冒充实现。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/iou_overlap.py#L5
        """
        return self._missing("overlap")

    def bev_iou(self, boxes_a, boxes_b):
        """旋转框 BEV IoU

        输入: boxes_a[N,7]、boxes_b[M,7]，float32；[x,y,z,dx,dy,dz,heading]，heading 为原旋转约定。
        输出: float32 Tensor[N,M]，应分配在输入设备，去掉原硬编码默认 CUDA 卡。
        反向: 原 BEV 扩展无注册反向。
        保持旋转方向、框中心/尺寸、几何边界与分母规则；空输入先运行 NVIDIA 参考确认。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/iou3d_nms_utils.py#L31
        """
        return self._missing("bev_iou")

    def iou3d(self, boxes_a, boxes_b):
        """三维框 IoU

        输入: boxes_a[N,7]、boxes_b[M,7]，float32；同 BEV 框格式。
        输出: float32 Tensor[N,M]；BEV 交集面积 × 高度交集 / max(体积并集,1e-6)。
        反向: 无完整自定义几何反向。原 BEV 缓冲区不求导，但外围高度/体积 Torch 计算可产生部分梯度；保持该图结构，不直接换成具有不同完整梯度的厂商 IoU3D。
        保留当前 Python 高度/体积组合，只替换 BEV overlap 内核最稳妥。主要用于 no_grad 匹配/评估；若训练配置需要梯度，单测原图行为。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/iou3d_nms_utils.py#L48
        """
        return self._missing("iou3d")

    def nms3d(self, boxes, scores, thresh, pre_maxsize=None, **kwargs):
        """旋转 3D NMS

        输入: boxes[N,7] float32、scores[N]、thresh、pre_maxsize=None；**kwargs 原函数不使用。
        输出: (indices[K], None)，indices 为输入设备上的 int64；必须映射回原输入索引。
        反向: 无梯度要求。
        先按 scores 降序，再按 pre_maxsize 截断；保持旋转框重叠和排序规则，输出的 None 不能删。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/iou3d_nms_utils.py#L82
        """
        return self._missing("nms3d")

    def nms3d_normal(self, boxes, scores, thresh, **kwargs):
        """普通 3D NMS

        输入: boxes[N,7] float32、scores[N]、thresh；**kwargs 原函数不使用。
        输出: (indices[K], None)，indices 为输入设备上的 int64。
        反向: 无梯度要求。
        按该 kernel 的轴对齐重叠规则实现，不复用旋转 NMS 的角度计算。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/iou3d_nms_utils.py#L100
        """
        return self._missing("nms3d_normal")

    def points_in_boxes(self, points, boxes):
        """点属于哪个三维框

        输入: points[B,M,3]、boxes[B,T,7]，float32、contiguous，batch 数一致。
        输出: int32 Tensor[B,M]，同设备；未命中为 -1。
        反向: 无梯度要求。
        保持边界判断、旋转坐标转换、多个框命中时原 kernel 的选择顺序。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roiaware_pool3d_utils.py#L34
        """
        return self._missing("points_in_boxes")

    def roiaware(self, rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        """RoIAwarePool3d 体素池化

        输入: rois[N,7]、pts[P,3]、pts_feature[P,C] float32；out_size 为 int 或 (x,y,z)；max_pts_each_voxel；pool_method="max" / "avg"。
        输出: float32 Tensor[N,out_x,out_y,out_z,C]，同设备。
        反向: 只对 pts_feature 返回 [P,C] 梯度；保存 int32 点索引、argmax、pool_method 和输入尺寸。
        保留体素坐标、每体素最大点数、max/avg 的空体素与截断规则。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roiaware_pool3d_utils.py#L63
        """
        return self._missing("roiaware")

    def roipoint(self, points, point_features, boxes3d, pool_extra_width, num_sampled_points=512):
        """RoIPointPool3d 区域点采样

        输入: points[B,N,3]、point_features[B,N,C]、boxes3d[B,M,7] float32；pool_extra_width；num_sampled_points=512。
        输出: (pooled[B,M,S,3+C], empty_flag[B,M])；S 为采样数；pooled float32，empty_flag int32，同设备。
        反向: 原版 backward 未实现；不要宣称该操作支持训练反向，需要反向时明确报不支持。
        先按原 enlarge_box3d 扩框；保留点顺序、不足采样数时的重复策略和空框标志。
        NVIDIA: https://github.com/jingfengou/United-Perception/blob/1208bfd9c3aeef3921db8ade83f7dbf85f583e5d/up/extensions/python/roipoint_pool3d_utils.py#L32
        """
        return self._missing("roipoint")
