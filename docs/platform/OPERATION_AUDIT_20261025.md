# 硬件接口逐项差异清单

截至 2026-10-25，当前父类定义 **156 项接口、18 类**。这是平台公开方法数，不是 GPU 内部全部内核数，也不是已通过真机测试的数量。
其中设备/训练/推理入口 27 项，分类能力入口 129 项；分类能力包含通信、量化和剪枝的状态接口。普通张量存储、autograd、索引、reshape、基础加减仍属厂商 PyTorch 设备桥接前置能力，不重复计入。SDK 导入、扩展构建/加载辅助函数也不按新的数学算子重复计数。

计数按公开接口：例如 `Conv2d` 层工厂与 `conv2d` 函数分别计一项；完整操作的前向/反向不再拆成两项，训练/推理复用同一接口也不重复计数。

[完整 CSV](operation_audit_20261025.csv) · [机器可读 JSON 与源证据哈希](operation_audit_20261025.json) · [全部签名和契约](INTERFACES.md)

这些接口的组件归属、训练/推理如何共用算子与设备，以及两家厂商的实现路径见[平台架构详图](ARCHITECTURE_20261025.md)。

## 三种比较必须分开

- **原 UP→新版，同一厂商**：19 个专用操作有独立小样本路由对照。参考提取原 UP Python/autograd 体，双方使用同一厂商编译扩展；C500 的 PSRoIPool 整数 scratch 明确补 contiguous。这验证路由迁移，不能据此证明原始内核与所有后续修正内核无差异。
- **新版平台→直接厂商 Torch，同一厂商**：16 类实际模型操作、每家 98 样本。两家各自全部前向 max_abs=0；这不是原 UP 单算子对照，不能把两个轴互换。
- **新版 NVIDIA↔新版 C500，同一公共操作数**：98 个模型样本中 19 个训练样本超差，64 个 eval/独立推理样本均在原容差内。另有修正后 KITTI 24 组数组跨卡最大差 0。其他接口没有配对测量时写“未测”，不填 0。

浮点容差固定 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`；索引/整数精确比较。最大绝对差不等于精度百分点，也不是任意输入上的误差上限。

分类训练采用 AMP FP16，YOLO 采用 FP32；各操作实际 dtype/shape/stride 和失败字段保留在 [19 个失败样本明细](validation/20261025/model-ops/nvidia-complete-02/findings.json)。参数梯度使用相同固定上游数列，不是整网损失梯度。

本轮 GPU 为 RTX 4090 / C500，Torch 分别为 `2.10.0+cu126` / `2.10.0+metax3.8.1.0`。厂商 PyTorch、DNN/BLAS 运行库不同，因此“跨卡差异”包含软件栈差异，不能只归于芯片；具体环境见[训练摘要](validation/20261025/fixed-500/completed-01/summary.json)。

## 分类计数与全部操作名

| 类别 | 项数 | 操作名（省略类别前缀） |
| --- | ---: | --- |
| 设备、RNG 与显存 / `device` | 18 | `available`, `convolution_flags`, `current_device`, `device`, `device_count`, `empty_cache`, `environment`, `get_rng_state`, `is_accelerator`, `manual_seed`, `max_memory_allocated`, `memory_allocated`, `move`, `properties`, `reset_peak_memory_stats`, `set_device`, `set_rng_state`, `synchronize` |
| 训练执行 / `training` | 7 | `autocast`, `clip_grad_norm_`, `fused_optimizer`, `grad_scaler`, `optimizer`, `validation`, `wrap_ddp` |
| 独立推理与导出 / `inference` | 2 | `autocast`, `export_onnx` |
| 激活与随机丢弃 / `activation` | 22 | `Dropout`, `Dropout2d`, `GELU`, `Hardswish`, `LeakyReLU`, `ReLU`, `ReLU6`, `SELU`, `SiLU`, `Sigmoid`, `Softmax`, `Tanh`, `gelu`, `hardtanh`, `log_softmax`, `relu`, `relu6`, `sigmoid`, `silu`, `softmax`, `softplus`, `tanh` |
| Attention / `attention` | 1 | `MultiheadAttention` |
| 通信 / `communication` | 17 | `all_gather`, `all_gather_object`, `all_reduce`, `barrier`, `broadcast`, `broadcast_object_list`, `destroy_process_group`, `gather`, `get_backend`, `get_rank`, `get_world_size`, `init_process_group`, `is_available`, `is_initialized`, `new_group`, `sum_op`, `world_group` |
| 卷积 / `convolution` | 4 | `Conv2d`, `ConvTranspose2d`, `conv2d`, `conv_transpose2d` |
| 可变形卷积 / `deformable` | 1 | `deform_conv` |
| 2D 检测专用操作 / `detection` | 6 | `iou_overlap`, `naive_nms`, `psroi_align`, `psroi_pool`, `roi_align`, `softer_nms` |
| Focal Loss / `focal` | 3 | `cross_sigmoid_focal_loss`, `sigmoid_focal_loss`, `softmax_focal_loss` |
| 通用损失 / `loss` | 10 | `BCEWithLogitsLoss`, `CosineSimilarity`, `CrossEntropyLoss`, `KLDivLoss`, `L1Loss`, `binary_cross_entropy`, `binary_cross_entropy_with_logits`, `cross_entropy`, `kl_div`, `l1_loss` |
| 矩阵运算 / `matrix` | 7 | `Linear`, `addmm`, `bmm`, `einsum`, `linear`, `matmul`, `mm` |
| 归一化 / `normalization` | 10 | `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `batch_norm`, `batch_norm_module_forward`, `group_norm`, `layer_norm`, `normalize`, `sync_batch_norm`, `sync_batch_norm_forward` |
| 点云与 3D 几何 / `pointcloud` | 10 | `boxes_bev_iou_cpu`, `boxes_iou3d`, `boxes_iou_bev`, `nms3d`, `nms3d_normal`, `points_in_boxes_cpu`, `points_in_boxes_gpu`, `roiaware_pool3d`, `roipoint_pool3d`, `rotate_iou_eval` |
| 池化、插值与填充 / `pooling` | 10 | `AdaptiveAvgPool2d`, `AvgPool2d`, `MaxPool2d`, `Upsample`, `UpsamplingBilinear2d`, `ZeroPad2d`, `adaptive_avg_pool2d`, `interpolate`, `max_pool2d`, `pad` |
| 剪枝工具链 / `pruning` | 10 | `build_scheduler`, `export`, `init_optimizer`, `last_prune_step`, `module_type`, `prepare`, `prune`, `set_iteration`, `set_lr_scheduler`, `sparse_rate` |
| 量化工具链 / `quantization` | 7 | `backend_types`, `convert_deploy`, `enable_calibration`, `enable_quantization`, `module_type`, `prepare`, `reconstruct` |
| 稀疏张量与卷积 / `sparse` | 11 | `SparseConv3d`, `SparseConvTensor`, `SparseInverseConv3d`, `SparseSequential`, `SubMConv3d`, `dense`, `features`, `is_sparse_convolution`, `module_instance`, `replace_feature`, `voxelize_cpu` |

## 16 类真实模型操作的跨卡数值

下表为这批样本的聚合最大值；“全部字段”包括输出、输入/参数梯度和运行状态。小于某个绝对值并非自动通过，还需按每个元素的相对容差判断。

| 操作 | 样本 | 跨卡前向 max_abs | 跨卡全部字段 max_abs | 超差样本 | 原因证据 |
| --- | ---: | ---: | ---: | ---: | --- |
| `activation.Dropout` | 6 | 5.73008347 | 5.73008347 | 2 | R |
| `activation.ReLU` | 6 | 0 | 0 | 0 | C |
| `activation.SiLU` | 6 | 9.53674316e-07 | 9.53674316e-07 | 0 | C |
| `activation.log_softmax` | 2 | 9.53674316e-07 | 6.10351562e-05 | 2 | F |
| `activation.sigmoid` | 6 | 1.1920929e-07 | 1.1920929e-07 | 0 | C |
| `activation.softmax` | 7 | 5.58793545e-09 | 3.81469727e-06 | 0 | C |
| `activation.tanh` | 3 | 1.1920929e-07 | 1.1920929e-07 | 0 | C |
| `convolution.Conv2d` | 15 | 0.00390625 | 2.5 | 5 | F |
| `loss.binary_cross_entropy_with_logits` | 2 | 9.53674316e-07 | 9.53674316e-07 | 0 | C |
| `matrix.Linear` | 9 | 0.0078125 | 0.0625 | 3 | F |
| `matrix.matmul` | 6 | 0.0078125 | 0.0078125 | 2 | F |
| `normalization.BatchNorm2d` | 12 | 0.00390625 | 0.00419998169 | 4 | F |
| `normalization.LayerNorm` | 3 | 9.53674316e-07 | 3.43322754e-05 | 1 | F |
| `pooling.AdaptiveAvgPool2d` | 3 | 1.43051147e-06 | 1.43051147e-06 | 0 | C |
| `pooling.MaxPool2d` | 6 | 0 | 0 | 0 | C |
| `pooling.Upsample` | 6 | 0 | 0 | 0 | C |

本批 7 类操作有训练样本超差：Conv2d、BatchNorm2d、Linear、matmul、LayerNorm、log_softmax、Dropout。不是 19 种不同函数。
两家各自前向与本厂商 Torch 逐位一致；NVIDIA 2 个、C500 5 个卷积权重梯度样本在直接参考及自身重复中都超差，失败样本集合相同。已确认存在原生执行差异，但没有量化每项对最终精度的贡献。
完整阶段/字段解释见[算子报告](MODEL_OPERATOR_PARITY_20261025.md)；新旧 NVIDIA 的 ViT 500 步 loss 与预测一致，ResNet/YOLO native 轨迹仍有差异，不能把本表当作全部训练差异的唯一解释。见[任务指标](FIXED_STEPS_20261025.md)。

## 原因与证据图例

- **U**：未逐接口隔离测量；没有数值或定位证据，不能填 0 或推断原因。 [证据](INTERFACES.md)
- **P**：所测原 UP Python/autograd 体与平台入口复用同一厂商编译扩展，支持路由兼容；不证明所有形状、旧/新内核或跨厂商相同。 [证据](VALIDATION.md#结果矩阵)
- **D**：DeformConv 原实现自身重复的输入梯度最大差也为 9.536743e-7；新旧差小于该重复差且在容差内，包含原生浮点/执行波动，未进一步证明某个指令是原因。 [证据](VALIDATION.md#结果矩阵)
- **F**：同厂商的平台前向与直接 Torch 逐位相同；跨厂商差异也存在于厂商 Torch 路径。涉及 AMP/归约，但未审阅厂商内部内核，不能断言某一方算错。 [证据](MODEL_OPERATOR_PARITY_20261025.md#跨厂商结果)
- **R**：同 seed 不保证跨设备随机流消耗/线程网格相同。既有纯 Torch 重放及真实掩码诊断复现差异；本轮未强制统一掩码。 [证据](VENDOR_TRAINING_DIAGNOSIS_20261024.md)
- **C**：这组公共输入样本全部在原容差内；若有非零低位差，同样存在于直接厂商 Torch，未定位到具体内部算法；不能推广全部变体。 [证据](MODEL_OPERATOR_PARITY_20261025.md#跨厂商结果)
- **A**：RoIAlign 原子反向的浮点加法顺序随调度变化，原版重复也有差异。新版仅在显式确定性开关开启时走固定顺序；小样本路由差为 0 不等于 native 训练始终稳定。 [证据](FRCNN_DETERMINISM_20261024.md)
- **G**：KITTI 旧几何边界错误经用户授权单独纠正：IoU/归一化重叠的新旧最大差为 1，原始交面积模式最大差为 13.39433956（面积单位）。两家修后同输入最大差为 0；不同单位不能合并解释为 IoU 差。 [证据](KITTI_GEOMETRY_20261024.md)
- **M**：设备拷贝逻辑值相同，但既有纯 Torch 探针中 NVIDIA 为 NCHW、C500 为 channels-last；这是实测布局差，布局可影响后续内核选择，不是值被平台改写。 [证据](VENDOR_TRAINING_DIAGNOSIS_20261024.md)
- **S**：设备元信息、句柄、同步、RNG、显存计数或执行上下文按语义核对，不应统一当作精度误差。没有本接口隔离的新旧/跨卡 max_abs。 [证据](VALIDATION.md)
- **T**：该类别已有同厂商直接依赖/路由组合流程证据；未输出本接口独立误差，不能把组合通过拆成所有成员逐项通过或跨厂商相等。 [证据](VALIDATION.md#结果矩阵)
- **B**：spring.linklink 变体缺 SDK（含 SyncBN/FP16 和三个融合优化器），相关变体未执行；不得用普通实现替代，数值差异未知。 [证据](LINKLINK.md)
- **V**：存在可选变体缺口：剪枝 2:4 permutation=False 流程已执行，Amba FakeSparse 缺依赖；不把全部剪枝变体标为通过。 [证据](VALIDATION.md#结果矩阵)
- **Q**：MQBench 校准/QAT/恢复/ONNX 与 QDrop 组合已执行；NART/部署引擎未验证，未隔离每个入口的 max_abs。 [证据](VALIDATION.md#结果矩阵)
- **X**：原有显式 CPU/cumm voxel 分支尚未做真机用例验收；不能将 PointPillar 执行成功作为该分支通过。 [证据](VALIDATION.md#结果矩阵)
- **H**：原版与新版 SegFormer MultiheadAttention 组合模块在两家各有 11 个 Tensor 比较通过；报告未保存最大差，不能填写为 0，也不能扩展为所有 Attention 变体。 [证据](VALIDATION.md#结果矩阵)

## 156 项逐项账本

“未做逐项原 UP 对照”不表示不支持，也不表示没有差异。量化/剪枝/稀疏/通信已有组合执行证据，但没有每个成员的独立误差；状态型接口不强行填写一个浮点误差。
下面的新旧数字仅指对应记录的小样本原 Python/autograd 体与平台路由对照，不覆盖所有 dtype/布局/形状。没有实际 Tensor 梯度时写“未测/不适用”，不伪造 0。

### 设备、RNG 与显存（18 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 1 | `device.available` | 未测 | 未测 | 未测 | S |
| 2 | `device.convolution_flags` | 未测 | 未测 | 未测 | S |
| 3 | `device.current_device` | 未测 | 未测 | 未测 | S |
| 4 | `device.device` | 未测 | 未测 | 未测 | S |
| 5 | `device.device_count` | 未测 | 未测 | 未测 | S |
| 6 | `device.empty_cache` | 未测 | 未测 | 未测 | S |
| 7 | `device.environment` | 未测 | 未测 | 未测 | S |
| 8 | `device.get_rng_state` | 未测 | 未测 | 未测 | S |
| 9 | `device.is_accelerator` | 未测 | 未测 | 未测 | S |
| 10 | `device.manual_seed` | 未测 | 未测 | 未测 | S |
| 11 | `device.max_memory_allocated` | 未测 | 未测 | 未测 | S |
| 12 | `device.memory_allocated` | 未测 | 未测 | 未测 | S |
| 13 | `device.move` | 未测 | 未测 | 逻辑值相同；既有探针布局不同 | M；既有逻辑值/stride 配对探针，不是本轮模型最终精度 |
| 14 | `device.properties` | 未测 | 未测 | 未测 | S |
| 15 | `device.reset_peak_memory_stats` | 未测 | 未测 | 未测 | S |
| 16 | `device.set_device` | 未测 | 未测 | 未测 | S |
| 17 | `device.set_rng_state` | 未测 | 未测 | 未测 | S |
| 18 | `device.synchronize` | 未测 | 未测 | 未测 | S |

### 训练执行（7 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 19 | `training.autocast` | 未测 | 未测 | 未测 | S |
| 20 | `training.clip_grad_norm_` | 未测 | 未测 | 未测 | S |
| 21 | `training.fused_optimizer` | 未测 | 未测 | 未测 | B |
| 22 | `training.grad_scaler` | 未测 | 未测 | 未测 | S |
| 23 | `training.optimizer` | 未测 | 未测 | 未测 | S |
| 24 | `training.validation` | 未测 | 未测 | 未测 | S |
| 25 | `training.wrap_ddp` | 未测 | 未测 | 未测 | S |

### 独立推理与导出（2 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 26 | `inference.autocast` | 未测 | 未测 | 未测 | S |
| 27 | `inference.export_onnx` | 未测 | 未测 | 未测 | S |

### 激活与随机丢弃（22 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 28 | `activation.Dropout` | 未测 | 未测 | 前向 5.73008347；全部字段 5.73008347；超差 2/6 | R |
| 29 | `activation.Dropout2d` | 未测 | 未测 | 未测 | U |
| 30 | `activation.GELU` | 未测 | 未测 | 未测 | U |
| 31 | `activation.Hardswish` | 未测 | 未测 | 未测 | U |
| 32 | `activation.LeakyReLU` | 未测 | 未测 | 未测 | U |
| 33 | `activation.ReLU` | 未测 | 未测 | 前向 0；全部字段 0；超差 0/6 | C |
| 34 | `activation.ReLU6` | 未测 | 未测 | 未测 | U |
| 35 | `activation.SELU` | 未测 | 未测 | 未测 | U |
| 36 | `activation.SiLU` | 未测 | 未测 | 前向 9.53674316e-07；全部字段 9.53674316e-07；超差 0/6 | C |
| 37 | `activation.Sigmoid` | 未测 | 未测 | 未测 | U |
| 38 | `activation.Softmax` | 未测 | 未测 | 未测 | U |
| 39 | `activation.Tanh` | 未测 | 未测 | 未测 | U |
| 40 | `activation.gelu` | 未测 | 未测 | 未测 | U |
| 41 | `activation.hardtanh` | 未测 | 未测 | 未测 | U |
| 42 | `activation.log_softmax` | 未测 | 未测 | 前向 9.53674316e-07；全部字段 6.10351562e-05；超差 2/2 | F |
| 43 | `activation.relu` | 未测 | 未测 | 未测 | U |
| 44 | `activation.relu6` | 未测 | 未测 | 未测 | U |
| 45 | `activation.sigmoid` | 未测 | 未测 | 前向 1.1920929e-07；全部字段 1.1920929e-07；超差 0/6 | C |
| 46 | `activation.silu` | 未测 | 未测 | 未测 | U |
| 47 | `activation.softmax` | 未测 | 未测 | 前向 5.58793545e-09；全部字段 3.81469727e-06；超差 0/7 | C |
| 48 | `activation.softplus` | 未测 | 未测 | 未测 | U |
| 49 | `activation.tanh` | 未测 | 未测 | 前向 1.1920929e-07；全部字段 1.1920929e-07；超差 0/3 | C |

### Attention（1 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 50 | `attention.MultiheadAttention` | SegFormer MHA 组合 11 项在容差内；max_abs 未记录 | SegFormer MHA 组合 11 项在容差内；max_abs 未记录 | 未测 | H |

### 通信（17 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 51 | `communication.all_gather` | 未测 | 未测 | 未测 | S |
| 52 | `communication.all_gather_object` | 未测 | 未测 | 未测 | S |
| 53 | `communication.all_reduce` | 未测 | 未测 | 未测 | S |
| 54 | `communication.barrier` | 未测 | 未测 | 未测 | S |
| 55 | `communication.broadcast` | 未测 | 未测 | 未测 | S |
| 56 | `communication.broadcast_object_list` | 未测 | 未测 | 未测 | S |
| 57 | `communication.destroy_process_group` | 未测 | 未测 | 未测 | S |
| 58 | `communication.gather` | 未测 | 未测 | 未测 | S |
| 59 | `communication.get_backend` | 未测 | 未测 | 未测 | S |
| 60 | `communication.get_rank` | 未测 | 未测 | 未测 | S |
| 61 | `communication.get_world_size` | 未测 | 未测 | 未测 | S |
| 62 | `communication.init_process_group` | 未测 | 未测 | 未测 | S |
| 63 | `communication.is_available` | 未测 | 未测 | 未测 | S |
| 64 | `communication.is_initialized` | 未测 | 未测 | 未测 | S |
| 65 | `communication.new_group` | 未测 | 未测 | 未测 | S |
| 66 | `communication.sum_op` | 未测 | 未测 | 未测 | S |
| 67 | `communication.world_group` | 未测 | 未测 | 未测 | S |

### 卷积（4 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 68 | `convolution.Conv2d` | 未测 | 未测 | 前向 0.00390625；全部字段 2.5；超差 5/15 | F |
| 69 | `convolution.ConvTranspose2d` | 未测 | 未测 | 未测 | U |
| 70 | `convolution.conv2d` | 未测 | 未测 | 未测 | U |
| 71 | `convolution.conv_transpose2d` | 未测 | 未测 | 未测 | U |

### 可变形卷积（1 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 72 | `deformable.deform_conv` | 训练前向 0 / 梯度 4.76837158e-07；推理 0 | 训练前向 0 / 梯度 5.96046448e-07；推理 0 | 未测 | D |

### 2D 检测专用操作（6 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 73 | `detection.iou_overlap` | 训练前向 0 / 梯度 未测/不适用；推理 0 | 训练前向 0 / 梯度 未测/不适用；推理 0 | 未测 | P |
| 74 | `detection.naive_nms` | 训练前向 0 / 梯度 未测/不适用；推理 0 | 训练前向 0 / 梯度 未测/不适用；推理 0 | 未测 | P |
| 75 | `detection.psroi_align` | 训练前向 0 / 梯度 0；推理 0 | 训练前向 0 / 梯度 0；推理 0 | 未测 | P |
| 76 | `detection.psroi_pool` | 训练前向 0 / 梯度 0；推理 0 | 训练前向 0 / 梯度 0；推理 0 | 未测 | P, T |
| 77 | `detection.roi_align` | 训练前向 0 / 梯度 0；推理 0 | 训练前向 0 / 梯度 0；推理 0 | 未测 | P, A |
| 78 | `detection.softer_nms` | 训练前向 0 / 梯度 未测/不适用；推理 0 | 训练前向 0 / 梯度 未测/不适用；推理 0 | 未测 | P |

### Focal Loss（3 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 79 | `focal.cross_sigmoid_focal_loss` | 训练前向 0 / 梯度 0；推理 0 | 训练前向 0 / 梯度 0；推理 0 | 未测 | P |
| 80 | `focal.sigmoid_focal_loss` | 训练前向 0 / 梯度 0；推理 0 | 训练前向 0 / 梯度 0；推理 0 | 未测 | P |
| 81 | `focal.softmax_focal_loss` | 训练前向 0 / 梯度 0；推理 0 | 训练前向 0 / 梯度 0；推理 0 | 未测 | P |

### 通用损失（10 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 82 | `loss.BCEWithLogitsLoss` | 未测 | 未测 | 未测 | U |
| 83 | `loss.CosineSimilarity` | 未测 | 未测 | 未测 | U |
| 84 | `loss.CrossEntropyLoss` | 未测 | 未测 | 未测 | U |
| 85 | `loss.KLDivLoss` | 未测 | 未测 | 未测 | U |
| 86 | `loss.L1Loss` | 未测 | 未测 | 未测 | U |
| 87 | `loss.binary_cross_entropy` | 未测 | 未测 | 未测 | U |
| 88 | `loss.binary_cross_entropy_with_logits` | 未测 | 未测 | 前向 9.53674316e-07；全部字段 9.53674316e-07；超差 0/2 | C |
| 89 | `loss.cross_entropy` | 未测 | 未测 | 未测 | U |
| 90 | `loss.kl_div` | 未测 | 未测 | 未测 | U |
| 91 | `loss.l1_loss` | 未测 | 未测 | 未测 | U |

### 矩阵运算（7 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 92 | `matrix.Linear` | 未测 | 未测 | 前向 0.0078125；全部字段 0.0625；超差 3/9 | F |
| 93 | `matrix.addmm` | 未测 | 未测 | 未测 | U |
| 94 | `matrix.bmm` | 未测 | 未测 | 未测 | U |
| 95 | `matrix.einsum` | 未测 | 未测 | 未测 | U |
| 96 | `matrix.linear` | 未测 | 未测 | 未测 | U |
| 97 | `matrix.matmul` | 未测 | 未测 | 前向 0.0078125；全部字段 0.0078125；超差 2/6 | F |
| 98 | `matrix.mm` | 未测 | 未测 | 未测 | U |

### 归一化（10 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 99 | `normalization.BatchNorm1d` | 未测 | 未测 | 未测 | U |
| 100 | `normalization.BatchNorm2d` | 未测 | 未测 | 前向 0.00390625；全部字段 0.00419998169；超差 4/12 | F |
| 101 | `normalization.LayerNorm` | 未测 | 未测 | 前向 9.53674316e-07；全部字段 3.43322754e-05；超差 1/3 | F |
| 102 | `normalization.batch_norm` | 未测 | 未测 | 未测 | U |
| 103 | `normalization.batch_norm_module_forward` | 未测 | 未测 | 未测 | U |
| 104 | `normalization.group_norm` | 未测 | 未测 | 未测 | U |
| 105 | `normalization.layer_norm` | 未测 | 未测 | 未测 | U |
| 106 | `normalization.normalize` | 未测 | 未测 | 未测 | U |
| 107 | `normalization.sync_batch_norm` | 未测 | 未测 | 未测 | T, B；Torch SyncBN 双卡路径已执行；linklink 变体缺 SDK |
| 108 | `normalization.sync_batch_norm_forward` | 未测 | 未测 | 未测 | T, B；Torch SyncBN 双卡路径已执行；linklink 变体缺 SDK |

### 点云与 3D 几何（10 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 109 | `pointcloud.boxes_bev_iou_cpu` | 训练前向 0 / 梯度 未测/不适用；推理 0 | 训练前向 0 / 梯度 未测/不适用；推理 0 | 未测 | P |
| 110 | `pointcloud.boxes_iou3d` | 训练前向 0 / 梯度 0；推理 0 | 训练前向 0 / 梯度 0；推理 0 | 未测 | P |
| 111 | `pointcloud.boxes_iou_bev` | 训练前向 0 / 梯度 未测/不适用；推理 0 | 训练前向 0 / 梯度 未测/不适用；推理 0 | 未测 | P |
| 112 | `pointcloud.nms3d` | 训练前向 0 / 梯度 未测/不适用；推理 0 | 训练前向 0 / 梯度 未测/不适用；推理 0 | 未测 | P |
| 113 | `pointcloud.nms3d_normal` | 训练前向 0 / 梯度 未测/不适用；推理 0 | 训练前向 0 / 梯度 未测/不适用；推理 0 | 未测 | P |
| 114 | `pointcloud.points_in_boxes_cpu` | 训练前向 0 / 梯度 未测/不适用；推理 0 | 训练前向 0 / 梯度 未测/不适用；推理 0 | 未测 | P |
| 115 | `pointcloud.points_in_boxes_gpu` | 训练前向 0 / 梯度 未测/不适用；推理 0 | 训练前向 0 / 梯度 未测/不适用；推理 0 | 未测 | P |
| 116 | `pointcloud.roiaware_pool3d` | 训练前向 0 / 梯度 0；推理 0 | 训练前向 0 / 梯度 0；推理 0 | 未测 | P |
| 117 | `pointcloud.roipoint_pool3d` | 训练前向 0 / 梯度 未测/不适用；推理 0 | 训练前向 0 / 梯度 未测/不适用；推理 0 | 未测 | P |
| 118 | `pointcloud.rotate_iou_eval` | 有意修正；IoU 1 / 交面积 13.3943396 | 有意修正；IoU 1 / 交面积 13.3943396 | 前向 0；全部字段 0；超差 0/24 | G |

### 池化、插值与填充（10 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 119 | `pooling.AdaptiveAvgPool2d` | 未测 | 未测 | 前向 1.43051147e-06；全部字段 1.43051147e-06；超差 0/3 | C |
| 120 | `pooling.AvgPool2d` | 未测 | 未测 | 未测 | U |
| 121 | `pooling.MaxPool2d` | 未测 | 未测 | 前向 0；全部字段 0；超差 0/6 | C |
| 122 | `pooling.Upsample` | 未测 | 未测 | 前向 0；全部字段 0；超差 0/6 | C |
| 123 | `pooling.UpsamplingBilinear2d` | 未测 | 未测 | 未测 | U |
| 124 | `pooling.ZeroPad2d` | 未测 | 未测 | 未测 | U |
| 125 | `pooling.adaptive_avg_pool2d` | 未测 | 未测 | 未测 | U |
| 126 | `pooling.interpolate` | 未测 | 未测 | 未测 | U |
| 127 | `pooling.max_pool2d` | 未测 | 未测 | 未测 | U |
| 128 | `pooling.pad` | 未测 | 未测 | 未测 | U |

### 剪枝工具链（10 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 129 | `pruning.build_scheduler` | 未测 | 未测 | 未测 | V |
| 130 | `pruning.export` | 未测 | 未测 | 未测 | V |
| 131 | `pruning.init_optimizer` | 未测 | 未测 | 未测 | V |
| 132 | `pruning.last_prune_step` | 未测 | 未测 | 未测 | V |
| 133 | `pruning.module_type` | 未测 | 未测 | 未测 | V |
| 134 | `pruning.prepare` | 未测 | 未测 | 未测 | V |
| 135 | `pruning.prune` | 未测 | 未测 | 未测 | V |
| 136 | `pruning.set_iteration` | 未测 | 未测 | 未测 | V |
| 137 | `pruning.set_lr_scheduler` | 未测 | 未测 | 未测 | V |
| 138 | `pruning.sparse_rate` | 未测 | 未测 | 未测 | V |

### 量化工具链（7 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 139 | `quantization.backend_types` | 未测 | 未测 | 未测 | Q |
| 140 | `quantization.convert_deploy` | 未测 | 未测 | 未测 | Q |
| 141 | `quantization.enable_calibration` | 未测 | 未测 | 未测 | Q |
| 142 | `quantization.enable_quantization` | 未测 | 未测 | 未测 | Q |
| 143 | `quantization.module_type` | 未测 | 未测 | 未测 | Q |
| 144 | `quantization.prepare` | 未测 | 未测 | 未测 | Q |
| 145 | `quantization.reconstruct` | 未测 | 未测 | 未测 | Q |

### 稀疏张量与卷积（11 项）

| # | 接口 | 原 UP→新版 NVIDIA | 原 UP→新版 C500 | 新 NVIDIA↔新 C500 | 原因/其他验证范围 |
| ---: | --- | --- | --- | --- | --- |
| 146 | `sparse.SparseConv3d` | 未测 | 未测 | 未测 | T |
| 147 | `sparse.SparseConvTensor` | 未测 | 未测 | 未测 | T |
| 148 | `sparse.SparseInverseConv3d` | 未测 | 未测 | 未测 | T |
| 149 | `sparse.SparseSequential` | 未测 | 未测 | 未测 | T |
| 150 | `sparse.SubMConv3d` | 未测 | 未测 | 未测 | T |
| 151 | `sparse.dense` | 未测 | 未测 | 未测 | T |
| 152 | `sparse.features` | 未测 | 未测 | 未测 | T |
| 153 | `sparse.is_sparse_convolution` | 未测 | 未测 | 未测 | T |
| 154 | `sparse.module_instance` | 未测 | 未测 | 未测 | T |
| 155 | `sparse.replace_feature` | 未测 | 未测 | 未测 | T |
| 156 | `sparse.voxelize_cpu` | 未测 | 未测 | 未测 | X |

## 两项有意改变及尚存缺口

1. **RoIAlign 确定性反向**：显式确定性开关开启时改为固定累加顺序；关闭时保持原子路径。优化前后同平台 8 组梯度逐位相同，不代表两厂商梯度逐位相同，见[性能与数值报告](ROI_ALIGN_PERFORMANCE_20261024.md)。
2. **KITTI rotate_iou_eval**：修正旧几何边界错误，新旧在既有参考的 64 对框上发生有意变化：默认 IoU 及 criterion=0/1 的归一化重叠最大差 1；criterion=2 的原始交面积最大差 13.39433956，单位是面积，不能混为 IoU 差。修后两家 24 组数组最大差 0，与 GEOS 的最大误差两家均 9.5367432e-7。不是模型算法变更或复刻旧错误值。
3. **linklink / Amba / 其他变体**：缺 SDK、FakeSparse 或尚未执行的分支保留为阻塞/未验证；不能用 Torch DDP、普通优化器或类别组合通过代替。
4. **缺少的逐项数据**：除表中明确给出的同输入数值外，其他接口未获单操作原 UP 四路径的完整笛卡尔积对照；不把已有模型/CPU 契约通过改写成这些数据已经存在。

本清单只汇总已保存证据，没有新增训练、GPU 作业、模型修复、容差调整或数值策略。原始输入、配置、归档索引和历史失败均保留。JSON 内记录所读源报告的 LF 文本 SHA256，便于核对这张表的依据。
