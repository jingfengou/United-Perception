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

```text
UP 模型 / 训练 / 独立推理
        ↓
PlatformBackend + 各算子类别父类
        ↓
ops/<类别>/nvidia.py 或 metax.py 的显式绑定
        ↓
对应厂商 PyTorch / 共享实现 / 厂商编译的内核
```

训练中的 eval 和 `train --evaluate` 使用训练适配；独立推理使用推理适配。
两者复用公共能力，不根据 `.eval()` 自动切换后端。

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

按 [接入指南](docs/platform/ADDING_VENDOR.md) 生成模板，实现设备后端、训练/推理差异和缺失算子。
兼容的厂商 PyTorch 操作可以显式复用；不兼容的操作必须提供完整实现及所需反向。
昇腾、寒武纪目前是逐项模板，不是已验证支持。

## 来源

基线为 UP `52aaeb9e`，架构参考 LightX2V `19c19614`。保留上游历史与 [Apache-2.0 LICENSE](LICENSE)。
模型和配方的原始用法见 [UP 官方文档](https://modeltc-up.readthedocs.io/en/latest/index.html)。
