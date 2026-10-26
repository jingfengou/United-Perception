# 新厂商接入

按四步完成：**准备环境 → 生成模板 → 实现接口 → 注册和真机验证**。
平台结构与接入流程图见[首页](../../README.md#如何调用)，每个函数的签名和要求见[完整接口](INTERFACES.md)。

当前 GPU 相关接口共 **155 项：设备 18 + 算子 125（15 类）+ 训练 10 + 推理 2**。
生成器保留全部 159 个必需函数契约，其中 4 个纯 CPU 兼容接口不计入 GPU 验收。

## 1. 准备厂商环境

准备显卡、驱动、SDK 和匹配的厂商 PyTorch，确认其设备名，如 `cuda`、`npu`、`mlu`。
后面的 `acme` 是示例厂商名，`privateuseone` 要替换成真实设备名。

已有厂商 PyTorch 时，复用兼容操作；缺失操作接 SDK 或自写内核。
没有 PyTorch 设备后端时，先完成文末的[设备桥接](#没有-pytorch-设备后端时)，再接入平台接口。

## 2. 生成模板

在仓库根目录执行，输出目录须尚不存在：

```bash
python tools/new_platform_vendor.py --vendor acme --device privateuseone --output acme_platform
```

若已确认厂商 Torch 兼容公共 API，可在命令末尾加 `--reuse-torch`。
它会绑定矩阵、卷积、注意力、归一化、激活、池化和损失这 7 类的 64 个公共入口，并复用训练/推理公共实现；其余 8 类算子仍需逐项填写。

生成的文件如下，15 类全部列出：

```text
acme_platform/
  __init__.py                 装配组件并注册厂商
  device.py                   设备接口、SDK 身份和版本检查
  training.py                 训练适配，包含训练内 eval
  inference.py                独立推理与 ONNX 导出
  ops/
    __init__.py
    matrix.py                 矩阵 / Linear
    convolution.py            卷积
    attention.py              注意力
    normalization.py          归一化
    activation.py             激活 / Dropout
    pooling.py                池化 / 插值 / Padding
    loss.py                   通用损失
    detection.py              NMS / RoIAlign 等检测操作
    deformable.py             可变形卷积
    focal.py                  Focal Loss
    pointcloud.py             点云 / 3D IoU / 点云池化
    sparse.py                 稀疏张量与稀疏卷积
    communication.py          通信与进程组
    quantization.py           量化
    pruning.py                剪枝
```

## 3. 实现接口

| 文件 | 需要完成的工作 |
| --- | --- |
| `device.py` | 延迟导入 SDK、检查厂商/版本；检查绑定、迁移、同步、显存、RNG 和卷积开关，覆盖不兼容方法 |
| `ops/*.py` | 按类别父类的签名，逐项绑定厂商 Torch、共享实现或 SDK/内核 |
| `training.py` | AMP、GradScaler、DDP/FSDP2、保存恢复、标准优化器、梯度裁剪、训练内验证 |
| `inference.py` | 推理 autocast、ONNX 导出；与训练实现分别装配 |
| `__init__.py` | 生成器已写入 `build(config, purpose)` 和注册；补齐组件后即可使用 |

### 兼容厂商 PyTorch：直接绑定

例如生成的 `ops/matrix.py` 可以使用以下完整绑定。`torch_impl` 调用当前安装的厂商 PyTorch：

```python
from up_platform.ops.matrix.base import MatrixOps
from up_platform.ops.matrix import torch_impl

class AcmeMatrixOps(MatrixOps):
    dependencies = ('torch',)
    linear = staticmethod(torch_impl.linear)
    matmul = staticmethod(torch_impl.matmul)
    mm = staticmethod(torch_impl.mm)
    bmm = staticmethod(torch_impl.bmm)
    addmm = staticmethod(torch_impl.addmm)
    einsum = staticmethod(torch_impl.einsum)
    Linear = staticmethod(torch_impl.Linear)
```

只需覆盖不兼容的方法。`linear` 函数和 `Linear` 层是独立入口，若两者都受影响，需要一起覆盖。
设备模板继承 `TorchDevice`；SDK 身份校验及不兼容的设备 API（例如卷积开关）由厂商补齐。

### 缺少算子：接 SDK 或自写内核

在插件内增加 `vendor_kernels.py`，调用真实 SDK/扩展。需要梯度的操作同时实现反向并保存必要状态。
[SDK Linear 示例](../../up_platform/templates/sdk_linear.py) 已示范前向和一阶反向接线，供应者需提供：

- `linear_forward(input, weight, bias)`。
- `linear_backward(grad_output, input, weight, has_bias, needs_input_grad)`，返回所需的输入、权重和 bias 梯度。

结果留在原 GPU 上；按[函数契约](INTERFACES.md)处理 dtype、布局、空输入、状态和导出。
示例的高阶梯度和 ONNX 适配需另行补充。使用共享内核源码时，用新厂商工具链独立编译，并在插件内加载。
尚未实现的函数保留明确报错。

## 4. 配置并验证

让生成的 `acme_platform` 包可导入：安装到厂商环境，或把其父目录加入 `PYTHONPATH`。
生成器已调用 `register_backend('acme', build)`，无需重复注册。在 UP YAML 中选择：

```yaml
runtime:
  platform:
    backend: acme
    device: privateuseone
    plugins: [acme_platform]
    requirements: [matrix.linear, convolution.Conv2d]
```

之后沿用 UP 的训练和推理命令，见[首页用法](../../README.md#选择与运行)。
训练和训练内 eval 使用 `training`；独立推理使用 `inference`。只支持一种用途时，另一种明确报不支持。

先检查模板和接口，再用厂商显卡验收：

```bash
python tools/platform_contracts.py
python tools/check_platform_boundaries.py
```

- **算子测试**：前向、所需梯度、状态更新、dtype、布局、空输入和导出；保持原容差。
- **模型测试**：训练、训练内 eval、权重恢复、独立推理；需要多卡时验证通信与 SyncBN。

记录所用环境、通过项和缺失项。Amba `FakeSparse` 等可选依赖的限制见[环境与依赖](ENVIRONMENT.md)。

## 没有 PyTorch 设备后端时

<details>
<summary>查看需要先补齐的设备桥接能力</summary>

UP 使用 PyTorch 张量、autograd 和模块系统，先为新设备补齐以下基础能力：

- 设备注册、上下文、设备绑定与 guard。
- 显存分配/释放、张量存储、引用计数和视图生命周期。
- CPU/GPU 及 GPU 间拷贝、异步与 pinned memory。
- 流、事件、同步、RNG 和错误传播。
- 张量构造、索引、reshape、转置、算术和归约派发。
- autograd、梯度累积、保存张量与原地修改检查。
- Parameter、state_dict、优化器状态、保存与恢复。
- AMP 和多卡通信后端。

桥接可用后，再按上述四步接入平台。

</details>
