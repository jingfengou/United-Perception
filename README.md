# UP Platform

<img src="up-logo.png" alt="United Perception" width="280" />

在原版 [United-Perception](https://github.com/ModelTC/United-Perception) 上重建的显式多厂商平台层。
模型、数据和训练配方留在 `up/`，设备、算子和厂商差异集中在 `up_platform/`。

**当前分支：`codex/up-platform-clean`。阶段性实现，尚未达到“两家完整适配”验收。**
NVIDIA 与沐曦已有真机记录；失败、缺口和复现步骤见 [验证报告](docs/platform/VALIDATION.md)。
本仓库是 UP 的正式 fork，`main` 保留原版 UP；平台改造位于本分支。
[旧实验仓库](https://github.com/jingfengou/United-Perception-Hardware) 的 `main`、Release 和历史证据保持原样。
本分支不包含旧数值修复或多随机种子任务。

[GPU 操作差异](#gpu-操作原版与新版是否一致) · [架构](docs/platform/README.md) · [完整接口](docs/platform/INTERFACES.md) · [接入厂商](docs/platform/ADDING_VENDOR.md) · [环境](docs/platform/ENVIRONMENT.md) · [验证](docs/platform/VALIDATION.md)

按责任拆分的提交及与已测试版本的对应关系见 [提交说明](docs/platform/HISTORY.md)。

## GPU 操作：原版与新版是否一致

在同一台 **NVIDIA RTX 4090** 上，使用相同输入、权重、配置和种子，对比原版 UP 与新版平台层的 GPU 计算输出、训练梯度及设备/训练状态。
专用扩展分别使用原版和新版各自编译的二进制，覆盖 16 个 GPU 操作、52 个训练/推理配对用例。

**清单内共 152 项 GPU 相关接口，已验证 148 项，另有 4 项缺依赖未验证。已测的 147 项新旧一致性检查通过；另 1 项 KITTI 评估器修正了原版错误，修复后通过独立几何参考检查。**

| 验证内容 | 接口数 | 结果 |
| --- | ---: | --- |
| 普通 GPU 操作、设备与训练状态 | 146 | **通过**：计算结果在原有容差内，设备与训练状态检查一致 |
| MQBench 量化初始化 | 1 | **通过**：量化观察器的初始状态与原版一致 |
| KITTI 旋转框 IoU 评估 | 1 | **修复后通过**：修正原版几何错误，结果与独立参考一致 |
| linklink 融合优化器 | 1 | **未验证**：环境缺少原版所需的 `spring.linklink` 实现库 |
| Amba 剪枝 | 3 | **未验证**：当前 MSBench 缺少 UP 配方所需的 `FakeSparse` 类，相关流程无法运行 |

浮点结果按原有 `rtol=1e-4, atol=1e-5` 验收。原版自身重复运行也有浮点波动；本次新旧 DeformConv 梯度最大差约 **`4.77e-7`**，`Upsample` / `interpolate` 最大差约 **`1.19e-7`**，均满足验收标准。

**KITTI 修复：** 原版在旋转框共边、端点重合等情况下会漏算交集顶点，部分完全相同的框被算出 `0.33333337` 或 `0` 的 IoU，正确值应为 `1`。
新版在框的局部坐标系内，沿边界裁剪求交集，保留边界上的点；中间计算使用双精度，最终仍输出 float32。在 7,100 对框上与独立几何库 GEOS 对照通过，相同框的 IoU 恢复为 `1`。这次修改的是评估器，模型算法保持原样。

统计范围包含 GPU 计算及设备、AMP、通信等配套接口，纯 CPU 操作和 CPU 测试分支已排除。
以上是 NVIDIA 的结果；沐曦已有的 GPU 算子测试见下方跨厂商记录，逐接口的新旧完整对照仍待补齐。

[GPU 逐项结果 CSV](docs/platform/validation/20261025/nvidia-interfaces/gpu-interfaces-20261026.csv) · [GPU 范围与差异详情](docs/platform/NVIDIA_INTERFACE_PARITY_20261025.md#gpu-相关口径2026-10-26-整理) · [跨厂商 GPU 算子记录](docs/platform/MODEL_OPERATOR_PARITY_20261025.md)

## 如何调用

<p align="center">
  <img src="docs/platform/assets/platform-overview.svg" width="100%" alt="UP 平台组成：统一入口选择厂商与用途，训练和推理共用设备接口及 15 类算子，底层绑定当前厂商实现" />
</p>

平台包含训练/推理适配，以及两者共用的设备和算子接口。模型直接调用分类算子，训练/推理适配提供 AMP、DDP 等执行能力。
15 类算子是：**矩阵、卷积、注意力、归一化、激活、池化/插值、损失、检测、可变形卷积、Focal 损失、点云、稀疏、通信、量化、剪枝**。
训练中的 eval 和 `train --evaluate` 使用训练适配；独立推理使用推理适配。详见[架构说明](docs/platform/ARCHITECTURE_20261025.md)。

**基础操作仍由厂商 PyTorch 负责。** 加减乘除、`cat/stack`、`sum/mean`、随机数和 autograd 等继续直接调用 PyTorch，按输入张量或指定设备调用相应实现，没有逐个包装进 `up_platform`。
PyTorch 已提供统一 API 和设备派发机制，平台层集中处理设备控制、厂商差异和专用算子，减少重复维护。
[LightX2V 的 RoPE](https://github.com/ModelTC/LightX2V/blob/19c19614b067f1746ac24350dfd097e1c0b48591/lightx2v_train/lightx2v_train/runtime/ops/rope.py#L17) 也直接使用 `cat/stack/reshape`，[训练损失](https://github.com/ModelTC/LightX2V/blob/19c19614b067f1746ac24350dfd097e1c0b48591/lightx2v_train/lightx2v_train/model_zoo/capability_adapters/common.py#L345-L368) 直接使用随机数、张量运算和归约。
新增厂商须同时提供这些基础 PyTorch 能力；某项不兼容时，再将该项纳入平台适配与测试。没有厂商 PyTorch 设备后端时，先完成[设备桥接](docs/platform/ADDING_VENDOR.md#没有-pytorch-设备后端时)。

## 安装

先进入匹配显卡、驱动的厂商 PyTorch 环境。沐曦必须保留厂商版 PyTorch，不能用 NVIDIA wheel 覆盖。
详细版本与可选依赖见 [环境说明](docs/platform/ENVIRONMENT.md)。原版 `requirements.txt` 是历史清单，不是现代环境锁文件。

```bash
git clone --branch codex/up-platform-clean https://github.com/jingfengou/United-Perception.git
cd United-Perception
python -m pip install --no-deps -e .

# 仅在匹配的厂商环境中编译专用扩展，二选一：
UP_BUILD_BACKEND=nvidia python setup.py build_ext --inplace
UP_BUILD_BACKEND=metax python setup.py build_ext --inplace
```

普通安装不探测 GPU、不编译扩展。无 PyTorch/CUDA 时可以导入 `up_platform` 公共接口；运行 UP 模型仍需完整环境。

## 选择与运行

在原有 YAML 的 `runtime` 中加入：

```yaml
runtime:
  platform:
    backend: metax   # NVIDIA 使用 nvidia
    device: cuda    # 设备名可以相同，厂商身份不同
```

配置好原版配方的数据和权重路径后，继续使用 UP 命令：

```bash
python -m up train --config configs/cls/resnet/res18.yaml --launch pytorch --ng 1 \
  --opts runtime.platform.backend metax

# 训练验证路径，不切换成部署推理后端
python -m up train --evaluate --config configs/cls/resnet/res18.yaml --launch pytorch --ng 1 \
  --opts runtime.platform.backend metax

# 独立推理路径；替换为实际的配置、图片和权重
python -m up inference --config model.yaml --img_path images/ --ckpt model.pth \
  --launch pytorch --ng 1 --opts runtime.platform.backend metax
```

`--backend dist/linklink` 仍表示通信方式，不是显卡厂商。
`linklink` 真实 SDK 缺失，详见 [依赖缺口](docs/platform/LINKLINK.md)，不会用 DDP 或普通优化器冒充。

## 新增厂商

<p align="center">
  <img src="docs/platform/assets/vendor-onboarding.svg" width="100%" alt="新增厂商四步：准备厂商环境，生成全部类别模板，实现或复用接口，注册配置并完成真机验证" />
</p>

**准备环境 → 生成模板 → 实现接口 → 注册和真机验证。**
兼容项直接复用厂商 PyTorch；缺失项接 SDK 或自写内核，并提供所需反向。
[接入指南](docs/platform/ADDING_VENDOR.md) 列全了 15 类文件、生成命令、实现示例和配置。昇腾、寒武纪当前为待实现和验证的模板。

## 来源

基线为 UP `52aaeb9e`，架构参考 LightX2V `19c19614`。保留上游历史与 [Apache-2.0 LICENSE](LICENSE)。
模型和配方的原始用法见 [UP 官方文档](https://modeltc-up.readthedocs.io/en/latest/index.html)。
