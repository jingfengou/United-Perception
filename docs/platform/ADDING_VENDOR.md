# 厂商接入指南

`RuntimeBackend` 提供设备与执行服务，`NativeOps` 提供专用操作。标准 Torch 操作直接复用厂商实现；不兼容部分在厂商后端处理，不复制模型或 Runner。

## 厂商接入

### 1. 准备环境

| 平台配置 | 设备 API | 配套库 |
|---|---|---|
| `nvidia` | `torch.cuda` | NVIDIA Torch / TorchVision、CUDA、cuDNN、cuBLAS、NCCL |
| `metax_cuda` | 厂商 Torch 的 `torch.cuda` 兼容 API | MACA Torch / TorchVision、mcDNN、mcBLAS、mcCL 兼容通信接口 |
| `ascend_npu` | `torch.npu` | Torch、torch_npu、CANN、HCCL |
| `cambricon_mlu` | `torch.mlu` | Torch、torch_mlu、CNNL、CNRT、CNCL |

先安装匹配的厂商 Torch、SDK 和驱动，再安装 UP。C500 使用 cuda 名称不表示使用 NVIDIA 二进制。插件只导入当前厂商 SDK，不修改其他环境。

### 2. 适配运行时

父类位于 `up/runtime/backend.py::RuntimeBackend`。设备 API 兼容时直接复用；不兼容时继承父类，只覆盖差异。

| 方法 / 属性 | 接口职责 | NVIDIA 参考 |
|---|---|---|
| `set_device(local_rank)` | 绑定当前进程；不可用或越界报错 | `torch.cuda.set_device` |
| `device` | 返回当前进程的 `torch.device` | `torch.device('cuda', current_device())` |
| `module` | 设备、同步、内存和 RNG 服务 | `torch.cuda` |
| `amp` | autocast、GradScaler 及其状态保存与加载 | `torch.cuda.amp` |
| `memory_available(device, needed_memory, reserve_bytes)` | 输入 byte 数，返回容量是否满足要求 | 总容量减已分配量与保留量 |
| `configure_dnn(options, legacy_nocudnn)` | 映射显式计算库策略；不支持时报错 | cuDNN 与 matmul TF32 开关 |
| `sync_batch_norm(input, weight, bias, mean, variance, eps, momentum, group)` | 归一化、状态更新、反向和指定组通信 | Torch SyncBatchNorm |

`module` 应提供以下服务，内存单位为 byte：

- 设备：`is_available`、`device_count`、`current_device`、`set_device`、`get_device_properties`、`get_device_name`。
- 内存与同步：`memory_allocated`、`max_memory_allocated`、`reset_peak_memory_stats`、`empty_cache`、`synchronize`。
- 随机状态：`manual_seed`、`get_rng_state`、`set_rng_state`。状态只在同厂商环境保存与加载。
- AMP：`autocast(enabled=...)` 与 GradScaler 的 scale / step / update / state_dict / load_state_dict。

NPU / MLU 的显式 DNN 选项目前需要厂商映射；不支持的选项直接报错。不能把空函数或关闭 AMP 当作兼容实现。

插件可放在独立包，通过 `PLUGINPATH` 在运行时初始化前加载：

```python
# vendor_plugins/ascend_runtime.py
from up.runtime import register_backend
from up.runtime.backend import RuntimeBackend

class AscendRuntime(RuntimeBackend):
    def configure_dnn(self, options=None, legacy_nocudnn=None):
        # Replace with the verified CANN policy mapping.
        raise NotImplementedError('Implement the requested CANN DNN policy')

register_backend(
    'ascend_npu', device_type='npu', dist_backend='hccl',
    extension='torch_npu', backend_class=AscendRuntime, replace=True,
)
```

```bash
PLUGINPATH=vendor_plugins.ascend_runtime python -m up train --config configs/det/yolox/yolox_small.yaml --opts runtime.platform ascend_npu
```

模板中的报错须替换为真实策略实现。注册在 `setup_runtime()` 前完成，不能在训练途中切换。

### 3. 绑定专用操作

契约位于 `up/platform/native.py::NativeOps`；实现位于 `up/platform/<vendor>/native_ops.py`。方法 docstring 列出参数、dtype、布局、坐标、空输入和梯度要求。NVIDIA 参考位于 `up/platform/nvidia/native_ops.py` 与 `up/extensions/csrc/`。

| 操作 | 非 NVIDIA 当前绑定 |
|---|---|
| `naive_nms`、`overlap`、`sigmoid_focal` | 显式复用共享 Torch |
| `roi_align`、`psroi_align`、`psroi_pool`、`deform_conv` | 未绑定，调用时报未实现 |
| `softmax_focal`、`cross_focal` | 未绑定，调用时报未实现 |
| `bev_iou`、`iou3d`、`nms3d`、`nms3d_normal` | 未绑定，调用时报未实现 |
| `points_in_boxes`、`roiaware`、`roipoint` | 未绑定，调用时报未实现 |

只为实际使用的操作补适配；接口存在不等于当前模型会调用它。完整签名及源码位置可在 [接口浏览页](HANDOFF.html) 展开。

```python
from up.platform.native import NativeOps as Contract
from up.platform import torch_native

class NativeOps(Contract):
    naive_nms = staticmethod(torch_native.naive_nms)
    overlap = staticmethod(torch_native.overlap)
    sigmoid_focal = staticmethod(torch_native.sigmoid_focal)
```

SDK 或自写内核在同一类覆盖同名方法。可微操作必须提供实际反向及所需状态；NMS 等不可微操作不用伪造梯度。内核和构建脚本放在本厂商目录，不能覆盖 NVIDIA 扩展。

### 4. 增加模型或操作

- Torch 已支持的操作：复用，核对新形状、dtype、前向和反向。
- 已有平台接口：复用调用；未支持的变体再补厂商实现。
- 通用 Torch 可组合的操作：维护一份共享实现，各厂商显式绑定。
- 专用内核：先定义统一契约和参考结果，再实现厂商内核。

接入时核对前向、反向和状态，然后运行训练、训练内验证、保存加载、独立推理与多卡通信。保持原算子容差，缺失能力明确报错，不新增静默 CPU 代算。没有 PyTorch 设备后端的厂商还需完成设备注册、Tensor 存储、拷贝、派发、stream 与 autograd 桥接。

[架构说明](ARCHITECTURE.md) · [训练指南](TRAINING.md)
