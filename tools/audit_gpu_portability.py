"""Index GPU-facing Python calls against three frozen UP revisions (stdlib only).

This is a source audit, not a device execution test. Tensor receiver types and
configuration-selected branches remain explicit validation tasks. CPU preprocessing
and demos are retained in a separate scope rather than counted as tested GPU work.
"""
import argparse
import ast
import collections
import csv
import hashlib
import json
import pathlib
import re


REVISIONS = {
    'original': '52aaeb9ed27a490820696d7754d19f98cd6ecd11',
    'slim': '40795a621699abf44be55aff5aea2c4e73e7419c',
    'multivendor': 'a78f9734fe1c2ec717369d76eefc2d366990f98c',
}
SOURCES = {
    'dispatch': ('PyTorch 设备派发', 'https://docs.pytorch.org/docs/stable/notes/extending.html'),
    'accuracy': ('PyTorch 数值与随机性', 'https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html'),
    'cudnn': ('PyTorch cuDNN 控制项', 'https://docs.pytorch.org/docs/2.10/backends.html#torch.backends.cudnn'),
    'metax': ('MetaX 厂商 PyTorch', 'https://github.com/MetaX-MACA/mcPytorch/tree/2.4'),
    'macalib': ('MACA 官方文档中心', 'https://developer.metax-tech.com/doc?category_ids=59'),
    'mlu': ('Cambricon PyTorch / SDK 依赖', 'https://github.com/Cambricon/torch_mlu/tree/r1.22_pt2.1.0'),
    'cnnl': ('CNNL 控制项实际实现（历史公开版本）', 'https://github.com/Cambricon/torch_mlu/blob/r1.22_pt2.1.0/torch_mlu/backends/cnnl/__init__.py'),
    'npu': ('Ascend torch_npu', 'https://github.com/Ascend/pytorch'),
    'cann': ('CANN 官方文档', 'https://www.hiascend.com/cann/document'),
}
# Stable review units: a family is an owner work package, not one GPU kernel.
FAMILIES = [
    ('device', '设备与迁移', 'runtime', 'device / set_device / to_device',
     '设备查询、绑定、Tensor/Module.to、CPU↔设备拷贝；不能把 cuda 字符串当厂商身份。'),
    ('memory', '显存与同步', 'runtime', 'memory_available / module.memory_* / synchronize',
     '显存单位为 byte；按输入设备查询，保留 UP 原有预留量；同步不改变计算。'),
    ('rng', '随机数与状态', 'runtime', 'manual_seed / get_rng_state / set_rng_state',
     'CPU 与设备 RNG 分开保存；每个 rank 独立恢复；相同种子不要求跨厂商掩码相同。'),
    ('dnn', 'cuDNN 与替代库策略', 'runtime', 'configure_dnn(options, legacy_nocudnn=None)',
     '默认保留厂商原生策略；benchmark、deterministic、enabled、TF32 逐项核实；没有对应开关就报不支持。'),
    ('amp', 'AMP 与梯度缩放', 'runtime', 'amp.autocast / amp.GradScaler',
     '确认 FP16/BF16 适用范围，unscale/step/update 顺序及 scaler 保存恢复；AMP 不是整网强转 FP16。'),
    ('distributed', '通信、DDP 与 SyncBN', 'runtime', 'dist_backend / torch.distributed / DistributedDataParallel',
     '先绑卡再建组；all_reduce/reduce/gather/broadcast/barrier 的同步/异步、组、dtype 和 stream 语义不变。SyncBN 需双卡反向。'),
    ('matrix', '矩阵与 Linear', 'torch', 'torch.matmul / Tensor @ / nn.Linear',
     '保留权重形状和 state_dict 键；检查转置/非连续输入、FP32/AMP、输入和权重梯度。'),
    ('convolution', '卷积', 'torch', 'nn.Conv* / F.conv*',
     '核大小、padding、stride、groups 和 dilation 不变；核对前向及输入/权重/bias 梯度。'),
    ('attention', 'Attention 组合', 'torch', 'UP Attention.forward → Linear / matmul / softmax / dropout',
     'UP 自写 QKV 和 mask 算法保留；不直接替换为另一个 fused attention；逐层核对现有计算。'),
    ('normalization', '归一化', 'torch', 'BatchNorm / LayerNorm / GroupNorm / functional norm',
     '训练更新 running_mean/var 和 num_batches_tracked；eval 不更新；eps、momentum 和统计轴不变。'),
    ('activation', '激活', 'torch', 'ReLU / SiLU / GELU / sigmoid / tanh / softmax',
     '原地行为、GELU 公式/近似方式与梯度不变；UP 自写 GELU 仍用原公式。'),
    ('pool', '池化、插值与采样', 'torch', 'pool / interpolate / upsample / grid_sample',
     '核对 mode、align_corners、边界与反向；不要因名字相同更换采样规则。'),
    ('loss', '标准损失', 'torch', 'cross_entropy / BCE / L1 / smooth_l1 / mse',
     '标签编码、ignore_index、reduction、归一化和空正样本行为不变。专用 Focal 另列。'),
    ('tensor', '基础 Tensor 运算与归约', 'torch', 'factory / arithmetic / cat / index / reshape / reduce',
     '继承输入设备；保留广播、dtype、布局和索引语义；输入在 GPU 上时由厂商 Torch 执行。'),
    ('autograd', '自动求导与优化器', 'torch', 'backward / autograd.grad / optimizer.step / clip_grad_norm',
     '核对累积、zero_grad、参数组、优化器状态和恢复；不替换优化算法。'),
    ('vision_nms', 'TorchVision NMS 路线', 'integration', 'up.platform.ops.nms(boxes, scores, threshold)',
     'NVIDIA 调用 TorchVision；其他厂商已有同设备 Torch 实现。连续坐标、排序、相同分数与空输入需验收；区别于 UP Native NMS。'),
    ('export', '导出与权重', 'integration', 'torch.onnx.export / symbolic / load / state_dict',
     '权重先在 CPU 读入再迁移；ONNX 节点、属性和自定义 symbolic 保持；导出能力另测。'),
    ('build', '原生扩展构建', 'integration', 'setup.py / CUDAExtension / _C',
     'NVIDIA 保留原 C++/CUDA；其他设备不加载 NVIDIA 二进制。移植完整操作及分配/反向，再构建厂商扩展。'),
    ('sparse', '稀疏卷积与体素', 'optional', 'spconv / SparseConvTensor / SparseConv* / replace_feature',
     '原版点云分支使用；瘦身任务已移除。若恢复，需同版本厂商构建、indices/indice_dict 和反向验证。'),
    ('quantization', '量化', 'optional', 'MQBench prepare / fake-quant / calibration / export',
     '原版可选分支；瘦身已移除。若恢复，需校准、QAT、观察器状态和目标导出逐项验证。'),
    ('pruning', '剪枝', 'optional', 'MSBench / Amba / mask / scheduler',
     '原版可选分支；瘦身已移除。若恢复，需实际依赖、掩码更新、训练恢复和导出验证。'),
    ('numba', 'Numba CUDA 几何评估器', 'optional', 'rotate_iou_gpu_eval / numba.cuda',
     '原版 KITTI GPU 评估内核；瘦身点云任务已移除。若恢复需厂商内核或同契约 GPU 实现，单换 Torch 无效。'),
    ('legacy', '历史 spring / linklink', 'excluded', '历史分支，不列入本轮厂商任务',
     '按用户决定不再接入，也不把缺少旧 SDK 列为当前平台阻塞。原版位置仅用于对照。'),
]
TENSOR_METHODS = set('cuda cpu to type type_as half float double bfloat16 new new_tensor new_zeros new_ones new_empty new_full '
                     'sum mean max min prod norm sqrt exp log pow sigmoid sigmoid_ tanh relu softmax log_softmax '
                     'matmul mm bmm add add_ mul mul_ div div_ sub sub_ clamp clamp_ abs neg square '
                     'reshape view permute transpose flatten contiguous squeeze unsqueeze expand expand_as repeat '
                     'gather scatter scatter_ scatter_add_ index_put_ index_select masked_fill masked_fill_ '
                     'nonzero topk sort argsort argmax argmin eq ne gt ge lt le any all where '
                     'normal_ uniform_ bernoulli_ random_ zero_ fill_ copy_ detach clone backward pin_memory record_stream'.split())
DEVICE_ATTRS = {'is_cuda', 'device', 'dtype', 'shape'}


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + '.' + node.attr
    if isinstance(node, ast.Call):
        return dotted(node.func) + '()'
    return '<tensor>'


def family(api, path, function):
    low = api.lower()
    if 'link' in low or low.startswith('spring.'):
        return 'legacy'
    if 'spconv' in low or 'voxel' in low:
        return 'sparse'
    if 'mqbench' in low:
        return 'quantization'
    if 'msbench' in low or 'amba' in low:
        return 'pruning'
    if low.startswith('numba.cuda'):
        return 'numba'
    if any(k in low for k in ('cudnn', 'cnnl', 'configure_dnn')):
        return 'dnn'
    if any(k in low for k in ('cudaextension', 'cpp_extension', 'load_library')):
        return 'build'
    if any(k in low for k in ('autocast', 'gradscaler', '.amp.')):
        return 'amp'
    if low.startswith('torchvision.') and 'nms' in low:
        return 'vision_nms'
    if any(k in low for k in ('memory_', 'empty_cache', 'synchronize', 'mem_get_info')):
        return 'memory'
    if any(k in low for k in ('rng_state', 'manual_seed', 'initial_seed')):
        return 'rng'
    if any(k in low for k in ('distributed', 'syncbatchnorm', 'ddp', 'dist.', 'all_reduce', 'broadcast', 'new_group')):
        return 'distributed'
    if any(k in low for k in ('onnx', 'symbolic', 'state_dict', 'torch.load', 'torch.save')):
        return 'export'
    if 'runtime' in low or any(part in ('cuda', 'npu', 'mlu', 'device', 'cpu', 'to', 'is_cuda') for part in low.split('.')):
        if not low.startswith(('torch.nn.', 'torch.nn.functional.')):
            return 'device'
    if any(k in low for k in ('conv1', 'conv2', 'conv3', '.conv', 'convolution')):
        return 'convolution'
    if any(k in low for k in ('linear', 'matmul', '.mm', '.bmm')):
        return 'matrix'
    if any(k in low for k in ('attention', 'multihead')):
        return 'attention'
    if any(k in low for k in ('batchnorm', 'layernorm', 'groupnorm', 'instancenorm', 'batch_norm', 'layer_norm', 'group_norm', 'instance_norm')):
        return 'normalization'
    if any(k in low for k in ('relu', 'gelu', 'silu', 'sigmoid', 'tanh', 'softmax', 'hardswish', 'dropout')):
        return 'activation'
    if any(k in low for k in ('pool', 'interpolate', 'upsample', 'grid_sample', 'pad(')):
        return 'pool'
    if any(k in low for k in ('loss', 'cross_entropy', 'binary_cross_entropy')):
        return 'loss'
    if any(k in low for k in ('autograd', 'backward', 'optimizer', 'optim.', 'clip_grad', '.step', 'zero_grad')):
        return 'autograd'
    return 'tensor'


class Visitor(ast.NodeVisitor):
    def __init__(self, path, text):
        self.path, self.text = path, text
        self.stack, self.aliases, self.calls = [], {}, []
        self.seen = set()
        # Definitions do not prove execution; conditional device paths remain annotated.
        self.demo_depth = 0

    def resolve(self, node):
        name = dotted(node)
        head, *tail = name.split('.')
        return '.'.join([self.aliases.get(head, head)] + tail)

    def add(self, node, api, kind='call'):
        key = (node.lineno, node.col_offset, api)
        if key in self.seen:
            return  # a Call and its callable Attribute are one source use
        self.seen.add(key)
        scope = 'model_or_runtime'
        if self.demo_depth or '/tests/' in self.path or '/test_' in self.path:
            scope = 'demo_or_test'
        elif any(s in self.path for s in ('/data/', '/datasets/', '/metrics/')):
            scope = 'data_or_evaluator_check_device'
        elif '/extensions/' in self.path or self.path == 'setup.py':
            scope = 'optional_extension_or_build'
        self.calls.append(dict(api=api, line=node.lineno, column=node.col_offset, function='.'.join(self.stack) or '<module>',
                               category=family(api, self.path, '.'.join(self.stack)), scope=scope, kind=kind,
                               expression=(ast.get_source_segment(self.text, node) or '')[:500]))

    def visit_Import(self, node):
        for item in node.names:
            self.aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]

    def visit_ImportFrom(self, node):
        module = node.module or ''
        for item in node.names:
            self.aliases[item.asname or item.name] = module + '.' + item.name
        if module.startswith(('spconv', 'mqbench', 'msbench', 'spring')):
            self.add(node, module, 'dependency')

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node):
        demo = '__name__' in ast.unparse(node.test) and '__main__' in ast.unparse(node.test)
        self.demo_depth += int(demo)
        self.generic_visit(node)
        self.demo_depth -= int(demo)

    def visit_Call(self, node):
        api = self.resolve(node.func)
        if api.startswith(('torch.', 'torchvision.', 'spconv.', 'mqbench.', 'msbench.', 'spring.', 'numba.cuda', 'up.extensions.')):
            self.add(node, api)
        elif isinstance(node.func, ast.Attribute) and node.func.attr in TENSOR_METHODS:
            self.add(node, 'Tensor.' + node.func.attr, 'tensor_candidate')
        elif isinstance(node.func, ast.Attribute) and node.func.attr == 'apply':
            self.add(node, api, 'custom_autograd_candidate')
        elif any(s in api for s in ('get_runtime', 'setup_runtime', 'self.runtime.', 'link.', 'linklink.',
                                    'roi_align.', 'deform_conv_v1.', 'overlaps.', 'focal_loss.', 'iou3d_nms.',
                                    'self.optimizer.', 'self.scaler.')):
            self.add(node, api)
        elif api.startswith('self.') and self.stack and self.stack[-1] == 'forward':
            self.add(node, 'Module.call:' + api, 'module_dispatch_candidate')
        self.generic_visit(node)

    def visit_Attribute(self, node):
        api = self.resolve(node)
        if (api.startswith(('torch.cuda.', 'torch.backends.')) or node.attr in DEVICE_ATTRS) and not isinstance(node.ctx, ast.Store):
            self.add(node, api, 'attribute')
        elif api.startswith('torch.backends.') and isinstance(node.ctx, ast.Store):
            self.add(node, api, 'policy_write')
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            self.add(node, 'Tensor.matmul', 'operator')
        self.generic_visit(node)


def scan(root, version):
    files, calls, functions = [], [], {}
    paths = sorted(root.glob('up/**/*.py')) + [root / 'setup.py']
    for file in paths:
        if not file.exists():
            continue
        path = file.relative_to(root).as_posix()
        raw = file.read_bytes()
        text = raw.decode('utf-8-sig')
        tree = ast.parse(text, filename=path)  # parse failures fail the audit, never disappear
        files.append(dict(path=path, sha256=hashlib.sha256(raw).hexdigest()))
        visitor = Visitor(path, text)
        visitor.visit(tree)
        for row in visitor.calls:
            row.update(path=path, baseline=version)
            row['url'] = 'https://github.com/{}/United-Perception/blob/{}/{}#L{}'.format(
                'ModelTC' if version == 'original' else 'jingfengou', REVISIONS[version], path, row['line'])
            calls.append(row)
        # Store full native function signatures, docstrings and forward/backward companions.
        class Definitions(ast.NodeVisitor):
            def __init__(self):
                self.stack = []
            def visit_ClassDef(self, node):
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()
            def visit_FunctionDef(self, node):
                name = '.'.join(self.stack + [node.name])
                functions[(path, name)] = dict(name=name, path=path, line=node.lineno,
                    signature=name + '(' + ast.unparse(node.args) + ')', doc=ast.get_docstring(node) or '')
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()
        Definitions().visit(tree)
    native = [dict(path=p.relative_to(root).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in sorted(root.glob('up/extensions/csrc/**/*')) if p.is_file()]
    return dict(files=files, native_sources=native, calls=calls), functions


def vendor_decision(category, vendor, retained=True):
    if category == 'legacy':
        return dict(action='excluded', text='不列入本轮接入', pending='历史分支，仅保留原版定位')
    if category in ('sparse', 'quantization', 'pruning', 'numba') and not retained:
        return dict(action='out_of_slim', text='瘦身版已移除', pending='仅恢复该任务时接入，先核对第三方库版本/内核')
    if category in ('quantization', 'pruning'):
        return dict(action='adapt', text='仅可选冻结 BN 引用保留', pending='量化/剪枝 Runner 已移除；使用该 norm 配置时核对厂商依赖和 BN 状态，不安排整套 QAT/剪枝')
    if category == 'vision_nms':
        return dict(action='adapt', text='已有同设备 Torch 实现', pending='核对空输入、阈值相等、分数并列和索引；性能另测')
    if vendor == 'metax' and category in ('device', 'memory', 'rng', 'amp', 'distributed'):
        return dict(action='reuse', text='复用厂商 CUDA 兼容服务', pending={
            'device': '确认厂商 Torch 的 cuda 可用、卡号、迁移与 stream 语义；不能因同叫 cuda 加载 NVIDIA 二进制',
            'memory': '确认显存 byte 单位、查询设备、同步与峰值统计；原来的分配估计规则保留',
            'rng': '确认设备 seed 与状态恢复；跨厂商不要求 RNG 字节和掩码相同',
            'amp': '核厂商 Torch AMP、FP16/BF16、溢出跳步、unscale 与 scaler 恢复',
            'distributed': 'NCCL 兼容 API 由 mcCL 执行；双卡验证实际用到的通信、DDP 和 SyncBN',
        }[category])
    if category in ('device', 'memory', 'rng', 'amp', 'distributed', 'dnn', 'export', 'build'):
        return dict(action='adapt', text='集中修改/核实后端', pending={
            'dnn': '逐项核实真实计算库和控制项；CNNL/CANN 不照搬 cuDNN 开关',
            'build': '禁止加载 NVIDIA _C；专用操作按完整契约移植/编译',
            'amp': '实测 dtype、溢出/跳步及 scaler 恢复',
            'distributed': '实测双卡集合通信、DDP、SyncBN、异步句柄',
        }.get(category, '在目标设备执行并核对状态、边界和恢复；本轮仅源码/配置测试'))
    return dict(action='reuse', text='复用原算法与厂商 Torch',
                pending='目标版本逐 dtype/shape 核对前向、所需梯度和状态；未覆盖的算子不能凭支持 Torch 推定通过')


# Native semantic operations. Signatures and implementation locations come from the baseline AST.
NATIVE = [
    ('roi_align', 'roi_align.py', 'RoIAlignFunction', 'RoIAlign', 'feature[N,C,H,W] + rois[K,5] → [K,C,pooled_h,pooled_w]；AVG、sampling_ratio、坐标偏移、空 ROI；反向到 feature；保留 symbolic。原 pybind 未导出 MAX 内核，MAX 不列为已支持变体'),
    ('psroi_align', 'psroi_align.py', 'PSRoIAlignFunction', 'PSRoIAlign', 'feature + rois → pooled；group_size/output_dim/spatial_scale/sampling_ratio；保存映射；反向到 feature；保留 symbolic'),
    ('psroi_pool', 'psroi_pool.py', 'PSRoIPoolFunction', 'PSRoIPool', 'feature + rois → pooled；位置敏感通道映射与 group_size；保存映射；反向到 feature；保留 symbolic'),
    ('deform_conv', 'deformable_conv.py', 'DeformableConvFunction', 'DeformConv', 'input/offset/weight + stride/pad/dilation/groups/deformable_groups → output；分别提供三者梯度；保留缓冲区、参数名和 symbolic'),
    ('naive_nms', 'nms.py', 'naive_nms', 'UP Native NMS', 'dets[N,5] + thresh → 保留索引；按原扩展坐标/阈值/排序契约；无反向；不同于 TorchVision 连续坐标 NMS'),
    ('softer_nms', 'nms.py', 'softer_nms', 'Softer NMS', 'dets[N,9] + sigma/阈值/投票方式 → boxes 与索引；保留方差投票及分数更新；无反向'),
    ('sigmoid_focal', 'focal_loss.py', 'SigmoidFocalLossFunction', 'Sigmoid Focal', 'preds[N,C]、target[N]、weight_pos/gamma/alpha/reduction → loss；背景0、前景1..C、忽略-1；反向到 logits'),
    ('softmax_focal', 'focal_loss.py', 'SoftmaxFocalLossFunction', 'Softmax Focal', 'preds/targets/weight_pos/gamma/alpha/num_classes → loss；保存 softmax 状态；反向到 logits'),
    ('cross_focal', 'cross_focal_loss.py', 'CrossSigmoidFocalLossFunction', 'Cross Focal', '额外 gt_class_to_avoid；保留跨类屏蔽与归一化；反向到 logits'),
    ('overlap', 'iou_overlap.py', 'gpu_iou_overlap', 'GPU IoU / IoF / IoS', 'b1[N,>=4],b2[M,>=4],mode → overlaps；+1 闭区间、分母至少1；保留原空输入 shape；无注册反向'),
    ('bev_iou', 'iou3d_nms_utils.py', 'boxes_iou_bev', 'BEV IoU', 'boxes[N,7],boxes[M,7] → [N,M]；旋转角、坐标系、精度和设备分配；无注册反向'),
    ('iou3d', 'iou3d_nms_utils.py', 'boxes_iou3d_gpu', '3D IoU', '旋转框 BEV overlap + 高度交集 → [N,M]；保留尺寸/旋转/空输入契约；无注册反向'),
    ('nms3d', 'iou3d_nms_utils.py', 'nms_gpu', '3D NMS', 'boxes/scores/thresh/pre_maxsize → 索引及原返回结构；排序后映射回输入；无反向'),
    ('nms3d_normal', 'iou3d_nms_utils.py', 'nms_normal_gpu', '普通 3D NMS', 'boxes/scores/thresh → 原索引及返回结构；不拿旋转 NMS 冒充该变体；无反向'),
    ('points_in_boxes', 'roiaware_pool3d_utils.py', 'points_in_boxes_gpu', 'GPU 点在框内', 'points[B,N,3],boxes[B,M,7] → box_idx[B,N]；原整数 dtype、未命中值、设备和边界；无反向'),
    ('roiaware', 'roiaware_pool3d_utils.py', 'RoIAwarePool3dFunction', 'RoIAwarePool3d', 'rois/pts/pts_feature/out_size/max_pts_each_voxel/pool_method → voxel features；保存 argmax/点索引；反向到点特征'),
    ('roipoint', 'roipoint_pool3d_utils.py', 'RoIPointPool3dFunction', 'RoIPointPool3d', 'points/point_features/boxes3d → pooled_features + empty_flag；采样数和扩框规则；原 backward 不支持，不伪造梯度'),
]


def build(snapshots):
    data, definitions = {}, {}
    for version in REVISIONS:
        data[version], definitions[version] = scan(snapshots / version, version)
    catalog = []
    for key, title, kind, interface, contract in FAMILIES:
        sites = {v: [r for r in data[v]['calls'] if r['category'] == key] for v in REVISIONS}
        retained = bool(sites['slim'])
        catalog.append(dict(id=key, title=title, kind=kind, interface=interface, contract=contract,
            apis=sorted({r['api'] for rows in sites.values() for r in rows}), sites=sites,
            vendors={v: vendor_decision(key, v, retained) for v in ('metax', 'cambricon', 'ascend')},
            evidence=['dispatch', 'accuracy', 'metax', 'mlu', 'npu'] + (['cudnn', 'macalib', 'cnnl', 'cann'] if key == 'dnn' else []),
            scope='保留' if retained else '瘦身已移除/原版可选', hardware='源码核查；目标卡执行待验收'))
    for key, filename, symbol, title, contract in NATIVE:
        path = 'up/extensions/python/' + filename
        signatures = {}
        for v, funcs in definitions.items():
            signatures[v] = [f for (p, n), f in funcs.items() if p == path and (n == symbol or n.startswith(symbol + '.'))]
        # A wrapper retained for compatibility is different from an active model consumer.
        consumers = {}
        for v in REVISIONS:
            names = {symbol, symbol.replace('Function', ''), symbol.replace('Function', 'Pool')}
            hits = []
            for file in sorted((snapshots / v / 'up').rglob('*.py')):
                rel = file.relative_to(snapshots / v).as_posix()
                if rel.startswith('up/extensions/'):
                    continue
                for number, line in enumerate(file.read_text(encoding='utf-8-sig').splitlines(), 1):
                    if not line.lstrip().startswith(('#', 'from ', 'import ')) and any(re.search(r'\b' + re.escape(n) + r'\b', line) for n in names):
                        hits.append(dict(path=rel, line=number, expression=line.strip()))
            consumers[v] = hits
        decisions = {v: dict(action='implement', text='移植/绑定完整专用操作',
            pending='先核厂商库是否存在同契约算子；有则显式绑定并测前后向，无则 SDK 内核实现；不能加载 NVIDIA 二进制')
            for v in ('metax', 'cambricon', 'ascend')}
        if key in ('sigmoid_focal', 'overlap'):
            for v in decisions:
                decisions[v] = dict(action='adapt', text='已有共享 Torch 路线，核完整变体',
                    pending='当前业务封装已部分接入；原生公开入口/IoS、空输入、梯度与所有变体仍需逐项验证')
        if key == 'softer_nms':
            # The original wrapper explicitly transfers to CPU and calls cpu_softer_nms.
            # It is an extension packaging boundary, not a missing vendor GPU kernel.
            contract = '原实现 dets.cpu() → cpu_softer_nms → 结果搬回输入设备；保留方差投票/分数更新。只需解耦 CPU 扩展的打包与加载，不派 GPU 内核移植任务。'
            decisions = {v: dict(action='cpu_only', text='CPU 计算，GPU 任务排除',
                pending='原 CPU 实现可保留；加载不应再依赖 NVIDIA CUDA 二进制；设备传输仍用 Tensor.to')
                for v in decisions}
        catalog.append(dict(id=key, title=title, kind='cpu_boundary' if key == 'softer_nms' else 'native', interface=symbol,
            contract=contract, path=path, signatures=signatures, consumers=consumers,
            sites={v: [r for r in data[v]['calls'] if r['path'] == path] for v in REVISIONS},
            apis=[symbol], vendors=decisions, evidence=['metax', 'mlu', 'npu'],
            scope='有瘦身业务引用' if consumers['slim'] else '扩展包装保留，瘦身模型未见直接引用',
            hardware='源码已核查；当前分支的目标卡验收待负责人补齐'))
    counts = {v: dict(python_files=len(d['files']), native_files=len(d['native_sources']),
                     call_sites=len(d['calls']), api_spellings=len({r['api'] for r in d['calls']}),
                     functions=len({(r['path'], r['function']) for r in d['calls']})) for v, d in data.items()}
    # Match functions by path and qualified name, not by shifted line numbers.
    inventory = collections.defaultdict(lambda: dict(sites={v: [] for v in REVISIONS}))
    for version, dataset in data.items():
        for row in dataset['calls']:
            key = row['path'] + '::' + row['function']
            item = inventory[key]
            item.update(id=key, path=row['path'], function=row['function'])
            item['sites'][version].append(row)
    for item in inventory.values():
        item['presence'] = {v: ((item['path'], item['function']) in definitions[v]
                               or (item['function'] == '<module>' and
                                   any(f['path'] == item['path'] for f in data[v]['files'])))
                            for v in REVISIONS}
        item['categories'] = sorted({r['category'] for rows in item['sites'].values() for r in rows})
    same_native = data['original']['native_sources'] == data['slim']['native_sources'] == data['multivendor']['native_sources']
    return dict(revisions=REVISIONS, generated='2026-11-03', summary=counts, families=catalog,
                native_sources_identical=same_native, function_inventory=list(inventory.values()),
                sources={k: dict(title=t, url=u) for k, (t, u) in SOURCES.items()},
                files={v: dict(python=d['files'], native=d['native_sources']) for v, d in data.items()},
                notes=['调用位置数、函数数、接口家族数是不同单位。',
                       'Tensor 方法是静态候选；接收者设备需真机 trace 确认。CPU 数据处理与示例已单列。',
                       '标准 Torch 数学复用，不新增逐算子包装。厂商版本、布局、梯度和状态仍逐项验收。'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = build(args.snapshots)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'gpu-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with (args.output / 'gpu-call-sites.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['baseline', 'path', 'line', 'column', 'function', 'api', 'category', 'scope', 'kind', 'expression', 'url']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for version in REVISIONS:
            rows, _ = scan(args.snapshots / version, version)
            writer.writerows(rows['calls'])
    print(json.dumps(dict(summary=result['summary'], families=len(result['families']),
        native_gpu_operations=sum(f['kind'] == 'native' for f in result['families']),
        cpu_extension_boundaries=sum(f['kind'] == 'cpu_boundary' for f in result['families'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
