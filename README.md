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

**NVIDIA 已测 GPU 操作总体保持原计算语义；存在微小浮点波动和单独修正的 KITTI 评估行为，不能概括为“全部逐位相同”。**
2026-10-26 按 GPU 范围整理已有真机证据，未新增训练；浮点容差仍为 `rtol=1e-4, atol=1e-5`。

| NVIDIA 新旧 UP 对照 | 接口数 | 实测结论 |
| --- | ---: | --- |
| 常规数值或状态匹配 | 146 | 所测结果在原容差内，或设备/状态语义一致；不要求所有浮点结果逐位相同 |
| MQBench 初始化状态匹配 | 1 | 未校准观察器的 `Inf/-Inf` 占位状态一致，单独记录 |
| KITTI 旋转框评估器有意修正 | 1 | 原版部分相同框的 IoU 为 `0.33333337` 或 `0`，修后与独立参考均为 `1`；不是模型精度下降 |
| 真实依赖缺失，尚未验收 | 4 | linklink 融合优化器 1 项、Amba 剪枝 3 项；未用替代实现冒充通过 |
| **GPU 相关口径合计** | **152** | 包含设备、AMP、通信及工具链配套接口，不是 152 个 GPU 计算内核 |

测得的普通非零差异为 DeformConv 梯度最大约 **`4.77e-7`**，`Upsample` / `interpolate` 最大约 **`1.19e-7`**；参考自身重复也有同量级波动，均未超原容差。
专用 GPU 扩展对照使用原版、新版各自二进制，覆盖 **16 个 GPU 操作、52 个训练/推理配对用例**。

GPU 统计已排除 4 个仅在 CPU 计算的接口，以及 `naive_nms` 的 CPU 测试分支；原功能和历史证据保留。
CPU 契约测试用于软件检查，不作为 GPU 数值一致性的证据。结论限于已测输入/变体，不代表完整模型收敛精度相同。
本表仅针对 NVIDIA；沐曦已有部分实测，尚未达到相同的逐接口新旧覆盖，不能直接沿用 NVIDIA 结论。

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
