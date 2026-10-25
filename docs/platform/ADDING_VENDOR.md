# 新厂商接入

先按接口交付能力，再用模型检查实际调用路径；不需要先猜厂商支持哪些模型。下面两张图分别说明平台组成和接入顺序。

![UP 平台组成、设备/共享算子/用途适配及当前 linklink 残留](assets/platform-overview.svg)

![新厂商接入：运行时前提、设备/算子/用途实现、注册配置与验收](assets/vendor-onboarding.svg)

图中 156 项是当前接口数量，不代表每项都要重新写内核。兼容的实现可以显式复用；未完成项必须保留为不支持。[完整函数清单](INTERFACES.md) · [当前逐项状态](OPERATION_AUDIT_20261025.md) · [带 11 处残留位置的架构详图](ARCHITECTURE_20261025.md)。

图里 linklink 的“缺 SDK”指缺少它的真实通信/训练加速库，不是缺少两家显卡的驱动或基础 CUDA/MACA。普通 Torch 路径已具备相应基础运行环境，详见 [linklink 是什么、需要哪些材料](LINKLINK.md#这里的-sdk-到底是什么)。

## 需要交付哪些文件

以已有厂商 PyTorch 的外部插件为例，生成器输出如下结构。`acme`、`privateuseone` 和包名仅为接入示例，不代表现有真实硬件支持。

```text
acme_torch_platform/
  __init__.py                 build(config, purpose) 与 register_backend
  device.py                   设备接口；导入/校验真实 SDK
  training.py                 训练接口，包含训练内 eval
  inference.py                独立推理及导出接口
  ops/
    matrix.py                 矩阵类别的厂商显式实现/绑定
    convolution.py            卷积类别的厂商显式实现/绑定
    ...                       全部 15 个类别，每个文件列出待实现方法
  vendor_kernels.py           可选：厂商自己提供的真实 SDK/内核绑定
```

| 要做的事情 | 对应文件/接口 | 已兼容厂商 PyTorch 时 | 不兼容或缺少时 |
| --- | --- | --- | --- |
| 设备管理 | `device.py` / `DeviceBackend`，18 项 | 继承 `TorchDevice`；补 SDK 身份和版本检查 | 覆盖不兼容方法；无设备后端先做桥接 |
| 算子/功能 | `ops/*.py` / 15 类父类，129 项 | 逐项显式绑定共享 `torch_impl` 或已确认兼容的实现 | 调 SDK/独立扩展，保持原签名、输出、所需反向和状态 |
| 训练适配 | `training.py` / `TrainingBackend`，7 项 | 显式复用 `TorchTraining` 的兼容方法 | 覆盖 AMP、DDP、优化器等差异；缺 SDK 不冒充可用 |
| 推理适配 | `inference.py` / `InferenceBackend`，2 项 | 显式复用 `TorchInference` | 补推理上下文和导出适配；只交付一种用途时拒绝另一种 |
| 装配和注册 | `__init__.py` | 生成器已经列出全部类别装配并注册厂商 | 补真实实现后再声明能力；不可留空后宣称完整支持 |

与厂商 SDK 兼容的 PyTorch/驱动仍需厂商提供。本项目模板不包含设备运行库，也不会自动把 NVIDIA wheel 变成其他厂商版本。

## 准备材料

1. 显卡、驱动、SDK、Python、PyTorch 和可选库的匹配版本及安装来源。
2. PyTorch 已注册的设备名，例如 cuda、npu、mlu 或 privateuseone；厂商名另行管理。
3. 按 [完整接口](INTERFACES.md) 逐项选择厂商 PyTorch、共享实现或自写内核，不需要先猜支持哪些模型。
4. 专用操作的前向、所需反向、状态、布局、导出实现，以及真实测试设备。

模型能否运行取决于它实际需要的接口。模板、导入或注册成功都不是完整适配。

## 生成全部待实现函数

```bash
# 纯模板：未实现项明确报错，不覆盖已有厂商
python tools/new_platform_vendor.py --vendor acme --device privateuseone --output acme_platform

# 已确认厂商 Torch 兼容公共 API 时，显式复用
python tools/new_platform_vendor.py --vendor acme --device privateuseone \
  --output acme_torch_platform --reuse-torch
```

生成 `device.py`、`training.py`、`inference.py` 和全部类别文件。
`--reuse-torch` 只复用基础类别与训练/推理的 Torch 入口；检测、点云、稀疏、通信、量化、剪枝等仍逐项填写。
它不是自动检测支持。生成器读取父类签名，新增接口会出现在下一次生成的模板中。

当前自动绑定的基础类别是 matrix、convolution、attention、normalization、activation、pooling、loss，共 **64 项**；另外 8 个类别、65 项仍生成明确报错的待实现入口。
设备模板继承了公共设备类，但 SDK 导入、身份/版本检查及不兼容方法必须实际补齐；训练/推理的公共绑定也不意味着所需 linklink 等 SDK 已存在。
上面两个命令是两种选择，输出目录必须尚不存在；配置中的 `plugins` 使用实际生成或安装的包名。

## 设备后端

在 `device.py` 中延迟导入 SDK，核对厂商 Torch 身份/版本，继承 `TorchDevice` 或实现 `DeviceBackend`。
检查设备绑定、迁移、同步、RNG、内存及卷积开关。同叫 cuda 也不能跳过身份校验。
共享卷积开关使用 `torch.backends.cudnn` 兼容 API；npu/mlu 等不兼容设备必须覆盖。
模板不提供任何厂商 SDK，不能改个名字就称为支持。

## 厂商 PyTorch 方式

与已实现的 `ops/matrix/metax.py` 相同：

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

`torch_impl.py` 明确绑定到 `torch.nn.functional.linear`、`torch.mm` 等，执行当前安装的厂商 Torch，而非 NVIDIA 专用二进制。
如果只有 linear 不兼容，就只覆盖 linear。完整的“复用公共实现、覆盖一个操作”示例可执行：

```bash
python -m pytest tests/platform/test_vendor_template.py -q
```

测试用 CPU 是明确的测试替身，不是实际 GPU 验证。

`linear` 函数和 `Linear` 层是两个独立接口；替换函数不会自动改写层内部的调用。若同一厂商问题同时影响两者，应一起提供兼容实现，不能只修函数后把整类 Linear 标为完成。

## SDK / 自写内核方式

[sdk_linear.py](../../up_platform/templates/sdk_linear.py) 包含完整的一阶 autograd 接线：

- 前向保存输入/权重并调用 `kernels.linear_forward`。
- 反向调用 `kernels.linear_backward` 返回输入、权重与可选 bias 梯度。
- 供应者负责批量维度、dtype、布局、设备流、空输入和 `needs_input_grad`。
- 示例只声明一阶梯度；未提供 ONNX 适配时明确报错。满足更完整契约需继续实现，不能冒充完整 Linear。
- CPU gradcheck 只证明接线，真实内核还需 GPU 前向/梯度/状态对照。

若已有 PyTorch 设备桥接，但 Linear 需要走厂商 SDK，可在上面的 `AcmeMatrixOps` 公共绑定基础上补充：

```python
import torch
from up_platform.templates.sdk_linear import bind_sdk_linear
from .. import vendor_kernels  # 厂商自己提供的真实 GPU 绑定，不是生成器自带实现

sdk_linear = bind_sdk_linear(vendor_kernels)

class AcmeLinear(torch.nn.Linear):
    # 继承原构造，保留 weight/bias 名称、初始化和 Parameter 结构。
    def forward(self, input):
        return sdk_linear(input, self.weight, self.bias)

AcmeMatrixOps.linear = staticmethod(sdk_linear)
AcmeMatrixOps.Linear = staticmethod(AcmeLinear)
```

`vendor_kernels.py` 需要交付 `linear_forward(input, weight, bias)` 和 `linear_backward(grad_output, input, weight, has_bias, needs_input_grad)`，返回相同设备上的结果与所需梯度，不允许转 CPU 代算。
该 Python 胶水模块应在需要时延迟加载真实 SDK/扩展；缺失时报告厂商、操作和原因，普通导入不初始化设备。
这个示例仍只声明一阶梯度；高阶导数和 ONNX 必须另外实现/验收，不能把示例包装当成完整生产 Linear。它同样适用于“厂商 Torch 已有、部分操作不兼容”的情况，不要求为单个缺失算子重新做完整设备桥接。

不能用同名操作冒充等价实现。RoIAlign、DeformConv 等的坐标、采样、offset、空输入与反向必须匹配。

若厂商工具链能够复用已有 C++/内核源码，也应单独编译成该厂商的插件扩展，并在插件中提供加载入口。当前内置 `load_native` 只登记了 NVIDIA/沐曦，新插件不会仅靠修改 vendor 字符串就自动复用它，更不能加载 `_nvidia_ops` 来充当新厂商二进制。

## 注册和用途

插件已调用 `register_backend('acme', build)`；`build(config, purpose)` 显式组合类别。
把插件放到 Python 导入路径后配置：

```yaml
runtime:
  platform:
    backend: acme
    device: privateuseone
    plugins: [acme_torch_platform]
    requirements: [matrix.linear, convolution.Conv2d]
```

只交付一种用途时，另一用途工厂应明确拒绝，不自动借用。
训练内 eval 使用 `TrainingBackend.validation`。融合优化器必须保留真实状态/master weights，无 SDK 时不能换普通 AdamW。

生成器已经在 `__init__.py` 写入 `register_backend('acme', build)`，不要再重复注册；重点是让 `build` 装配真实设备、全部所需类别和当前用途实现。
插件须可被当前解释器导入，可安装到该厂商环境，或将它的父目录加入 `PYTHONPATH`。

## 没有 PyTorch 适配时

仅填几个算子不够，UP 仍依赖 PyTorch 张量、autograd、模块和基础运算。先交付设备桥接：

| 层面 | 必须交付 |
| --- | --- |
| 注册 | Dispatch key、设备名、上下文、device guard |
| 存储 | 分配器、storage、引用计数、视图生命周期和释放 |
| 传输 | CPU/设备间及设备间拷贝、异步和 pinned memory |
| 执行 | 流、事件、同步、错误传播、RNG、多设备 |
| 基础派发 | 构造、索引、reshape、转置、拷贝、算术和归约 |
| 自动求导 | 反向派发、保存张量、in-place/version、梯度累积 |
| 模块/保存 | Parameter、state_dict、优化器状态、序列化和恢复 |
| 精度/多卡 | autocast、scaler、通信后端和同步语义 |

桥接完成后再用本项目父类、模板和 SDK 示例接入。
本分支没有重写一个脱离 PyTorch 的框架，不承诺缺少桥接时 UP 可运行。

## 验收

检查父类/模板/绑定/文档一致，缺失能力明确报错。
先测 dtype、布局、边界、前向、梯度、状态和导出，再测模型训练、训练内 eval、恢复、独立推理及多卡。
保留失败和同源重复；不放宽容差、不做静默 CPU 代算、不挑最好一次。
在已完成隔离的主流程中，新增厂商增加后端/插件与注册配置，已有模型算法和配方保持不变。
当前 11 处旧 linklink 分支仍需平台公共边界收尾；不能承诺新厂商只填模板就能运行这些尚未迁移的分支。
